"""Index supplied original source articles, without treating training labels as verdicts."""
import argparse
from collections import Counter
import hashlib
import json
import re
from pathlib import Path
import sqlite3
from urllib.parse import urlparse
import zipfile

from backend.store import now
from ml.prepare_immfnd import read_xlsx

SOURCE_HOSTS = {'boomlive.in', 'factly.in', 'altnews.in', 'vishvasnews.com', 'newsmeter.in',
                'factcheck.afp.com', 'reuters.com', 'apnews.com', 'bbc.com', 'bbc.co.uk', 'factcrescendo.com', 'newschecker.in',
                'thestatesman.com', 'tribuneindia.com', 'thehindu.com', 'indianexpress.com', 'abplive.com'}


def build(archive_path, output):
    archive_path, output = Path(archive_path), Path(output)
    if output.exists():
        raise ValueError('Choose a new output path; existing indexes are preserved.')
    output.parent.mkdir(parents=True, exist_ok=True)
    stamp = now()
    counts, documents = Counter(), {}
    with zipfile.ZipFile(archive_path) as archive:
        for member in sorted(archive.namelist()):
            if not member.endswith('_text.xlsx'):
                continue
            for row in read_xlsx(archive.read(member)):
                counts['input_rows'] += 1
                url = str(row.get('url') or '').strip()
                parsed = urlparse(url)
                host = (parsed.hostname or '').lower().removeprefix('www.')
                if (parsed.scheme != 'https' or parsed.username or parsed.password or parsed.port not in (None, 443)
                        or not any(host == domain or host.endswith('.' + domain) for domain in SOURCE_HOSTS)):
                    counts['excluded_source'] += 1
                    continue
                body = str(row.get('content') or '').strip()
                if len(body) < 100:
                    counts['excluded_missing_article_body'] += 1
                    continue
                title = str(row.get('title') or '').strip()
                # The source article is evidence. The spreadsheet fake/real label
                # is deliberately excluded: some supplied labels contradict the article.
                content = body[:40000]
                identity = hashlib.sha256((url + '\n' + content).encode()).hexdigest()
                if url in documents:
                    counts['duplicate_source_urls'] += 1
                    continue
                published = None
                date = str(row.get('date') or '').strip()
                if date and re.search(r'\b(?:19|20)\d{2}\b', date):
                    try:
                        from dateutil.parser import parse
                        published = parse(date, fuzzy=True, tzinfos={'IST': 19800}).isoformat()
                    except (ValueError, OverflowError):
                        counts['unparsed_publication_date'] += 1
                documents[url] = (identity, title, content, url, host, stamp, published)
    with sqlite3.connect(output) as db:
        db.execute('CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)')
        db.execute('CREATE TABLE documents (id INTEGER PRIMARY KEY, content_hash TEXT UNIQUE, '
                   'title TEXT, text TEXT, url TEXT, publisher TEXT, fetched_at TEXT, published_at TEXT)')
        db.execute("CREATE VIRTUAL TABLE documents_fts USING fts5(title,text,content='documents',content_rowid='id')")
        db.executemany('INSERT INTO documents(content_hash,title,text,url,publisher,fetched_at,published_at) VALUES (?,?,?,?,?,?,?)', documents.values())
        db.execute("INSERT INTO documents_fts(documents_fts) VALUES('rebuild')")
        db.executemany('INSERT INTO metadata VALUES (?,?)', [('name', 'Supplied original source articles'),
                       ('built_at', stamp), ('document_count', str(len(documents)))])
    with archive_path.open('rb') as handle:
        archive_sha = hashlib.file_digest(handle, 'sha256').hexdigest()
    audit = {'archive_sha256': archive_sha, 'documents': len(documents), 'counts': dict(counts),
             'label_policy': 'Dataset classification labels are not used as evidence verdicts.',
             'source_policy': 'Only original HTTPS articles from allowlisted publishers; runtime enabled-source controls still apply.'}
    output.with_name('import-audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--output', default='models/dataset-articles-v2/evidence.sqlite')
    args = parser.parse_args()
    print(json.dumps(build(args.archive, args.output), indent=2))
