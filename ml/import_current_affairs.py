"""Audit supplied data and build a local SQLite FTS index of official documents.

No downloaded Python, instructions, models, or pickle files are executed.
Unlabeled government pages are evidence documents, never fake/real examples.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from urllib.parse import urlparse
import zipfile

import pyarrow.parquet as pq

from backend.current_affairs import official_url


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def build(source_dir, output):
    source_dir, output = Path(source_dir), Path(output)
    files = [source_dir / f'train-{i:05d}.parquet' for i in range(7)]
    if any(not path.is_file() for path in files):
        raise ValueError('All seven supplied Parquet shards are required.')
    if output.exists():
        raise ValueError('Choose a new output path; existing indexes are preserved.')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.building.sqlite')
    if temporary.exists():
        raise ValueError('A previous build file exists. Use a new output path.')
    counts, hosts, dates = Counter(), Counter(), Counter()
    sources = []
    with sqlite3.connect(temporary) as db:
        db.execute('CREATE TABLE documents (id INTEGER PRIMARY KEY, content_hash TEXT UNIQUE, '
                   'title TEXT NOT NULL, text TEXT NOT NULL, url TEXT NOT NULL, publisher TEXT NOT NULL, fetched_at TEXT NOT NULL)')
        db.execute('CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
        for path in files:
            parquet = pq.ParquetFile(path)
            sources.append({'path': str(path.resolve()), 'sha256': digest(path), 'rows': parquet.metadata.num_rows})
            for batch in parquet.iter_batches(batch_size=128):
                for row in batch.to_pylist():
                    counts['input_rows'] += 1
                    url = row.get('url') or ''
                    try:
                        valid_url = official_url(url)
                    except ValueError:
                        valid_url = False
                    if not valid_url:
                        counts['excluded_non_https_or_non_government_url'] += 1
                        continue
                    body = row.get('text') or ''
                    if not isinstance(body, str) or len(body.strip()) < 100:
                        counts['excluded_short_or_empty'] += 1
                        continue
                    fetched = row.get('fetched_at')
                    try:
                        stamp = datetime.fromisoformat(fetched.replace('Z', '+00:00'))
                        if stamp.tzinfo is None or stamp > datetime.now(timezone.utc):
                            raise ValueError('Invalid snapshot time')
                    except (ValueError, AttributeError, TypeError):
                        counts['excluded_invalid_snapshot_date'] += 1
                        continue
                    original_body = body
                    body = body[:100000]
                    if len(original_body) > len(body):
                        counts['truncated_large_documents'] += 1
                    normalized = re.sub(r'\s+', ' ', original_body).strip().casefold()
                    identity = hashlib.sha256(normalized.encode()).hexdigest()
                    host = urlparse(url).hostname
                    cursor = db.execute('INSERT OR IGNORE INTO documents '
                        '(content_hash,title,text,url,publisher,fetched_at) VALUES (?,?,?,?,?,?)',
                        (identity, str(row.get('title') or host)[:1000], body, url, host, fetched))
                    if cursor.rowcount:
                        counts['indexed_documents'] += 1
                        hosts[host] += 1
                        dates[fetched[:10]] += 1
                    else:
                        counts['duplicate_documents'] += 1
            db.commit()
            print(f"Imported {path.name}: {counts['indexed_documents']:,} official documents so far", flush=True)
        db.execute("CREATE VIRTUAL TABLE documents_fts USING fts5(title, text, content='documents', content_rowid='id', tokenize='unicode61')")
        print('Building full-text search index...', flush=True)
        db.execute("INSERT INTO documents_fts(documents_fts) VALUES('rebuild')")
        db.execute("INSERT INTO documents_fts(documents_fts) VALUES('optimize')")
        db.execute("INSERT INTO documents_fts(documents_fts, rank) VALUES('integrity-check', 1)")
        metadata = {'format_version': '1', 'built_at': datetime.now(timezone.utc).isoformat(),
                    'document_count': str(counts['indexed_documents']),
                    'latest_snapshot': max(dates), 'earliest_snapshot': min(dates),
                    'method': 'SQLite FTS5 BM25 with query-matched source excerpts'}
        db.executemany('INSERT INTO metadata VALUES (?,?)', metadata.items())
        db.commit()
        assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    db.close()
    audit = {'sources': sources, 'counts': dict(counts), 'snapshot_dates': dict(dates),
             'official_host_count': len(hosts), 'metadata': metadata,
             'limitations': ['No fake/real labels are supplied; these documents were not used for classifier training.',
                 'A fetched_at date records collection, not publication or present-day truth.',
                 'Non-government and HTTP source URLs are excluded rather than trusted from dataset labels.',
                 'Source licensing was not supplied with the Parquet files; preserve provenance for local research.']}
    qa = source_dir / 'train_0.jsonl'
    if qa.is_file():
        qa_count = 0
        with qa.open(encoding='utf-8-sig') as handle:
            for line in handle:
                if line.strip():
                    json.loads(line)
                    qa_count += 1
        audit['qa_dataset'] = {'path': str(qa.resolve()), 'sha256': digest(qa), 'rows': qa_count,
            'decision': 'Audited only. Model-generated expansions and no original source URLs; not treated as authoritative evidence or fake/real labels.'}
    archive = source_dir / 'Dataset-main.zip'
    if archive.is_file():
        manifests = []
        with zipfile.ZipFile(archive) as zipped:
            names = set(zipped.namelist())
            for name in ['Dataset-main/data/manifests/v0.1.jsonl', 'Dataset-main/data/manifests/v0.2.jsonl']:
                rows = [json.loads(line) for line in zipped.read(name).decode('utf-8-sig').splitlines() if line.strip()]
                missing = sum('Dataset-main/' + row['file'] not in names for row in rows if row.get('file'))
                manifests.append({'name': name, 'records': len(rows), 'missing_referenced_files': missing})
        audit['fact_check_archive'] = {'path': str(archive.resolve()), 'sha256': digest(archive), 'manifests': manifests,
            'decision': 'Manifest-only release; full claim/evidence records are absent. No downloaded code executed.',
            'license_note': 'README and datasheet contain differing license statements; preserve both before redistribution.'}
    audit['incomplete_downloads'] = [{'name': path.name, 'bytes': path.stat().st_size,
                                    'decision': 'Excluded: incomplete download.'}
                                   for path in source_dir.glob('Unconfirmed*.crdownload')]
    temporary.replace(output)
    output.with_name('import-audit.json').write_text(json.dumps(audit, indent=2), encoding='utf8')
    print(json.dumps({'index': str(output), 'counts': dict(counts), 'qa': audit.get('qa_dataset'),
                      'archive': audit.get('fact_check_archive')}, indent=2), flush=True)
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', required=True)
    parser.add_argument('--output', default='models/current-affairs/evidence.sqlite')
    args = parser.parse_args()
    build(args.source_dir, args.output)
