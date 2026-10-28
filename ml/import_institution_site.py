"""Import selected GIST pages as source evidence, never classifier labels."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from urllib.parse import urljoin, urlparse

import requests

from backend.store import now
from backend.verification import _ArticleParser


HOME = 'https://gist.edu.in/gist/gist-home/'
IDENTITY = 'Geethanjali Institute of Science and Technology (GIST)'
SECTIONS = {'about us', 'programmes offered', 'contact us', 'vision-mission',
            'management', 'principal', 'director', 'central library',
            'placement record', 'admissions-policy'}


class InstitutionParser(_ArticleParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.link = None

    def handle_starttag(self, tag, attrs):
        super().handle_starttag(tag, attrs)
        if tag == 'a':
            self.link = [dict(attrs).get('href', ''), []]
        if tag in ('p', 'tr', 'h1', 'h2', 'h3', 'h4', 'li', 'br') and not self._ignored:
            self.visible_text.append('\n')

    def handle_data(self, data):
        super().handle_data(data)
        if self.link is not None:
            self.link[1].append(data)

    def handle_endtag(self, tag):
        super().handle_endtag(tag)
        if tag == 'a' and self.link is not None:
            self.links.append((self.link[0], ' '.join(self.link[1]).strip()))
            self.link = None

    def document(self):
        body = ' '.join(self.visible_text)
        # The site repeats its full navigation after this footer heading.
        body = re.split(r'Campus\s+Hub', body, maxsplit=1, flags=re.I)[0]
        body = '\n'.join(re.sub(r'\s+', ' ', line).strip() for line in body.splitlines() if line.strip())
        return IDENTITY + '\n' + body


def allowed(url):
    parsed = urlparse(url)
    return (parsed.scheme == 'https' and parsed.hostname == 'gist.edu.in'
            and not parsed.username and not parsed.password and parsed.port in (None, 443)
            and parsed.path.startswith('/gist/') and not parsed.query
            and not re.search(r'\.(pdf|zip|jpg|png|docx?)$', parsed.path, re.I))


def fetch(url):
    if not allowed(url):
        raise ValueError('Only selected HTTPS pages on gist.edu.in/gist/ may be imported.')
    with requests.get(url, timeout=(5, 25), stream=True, allow_redirects=False,
                      headers={'User-Agent': 'GISTLocalEvidenceImporter/1.0'}) as response:
        response.raise_for_status()
        if response.status_code != 200 or 'html' not in response.headers.get('Content-Type', '').lower():
            raise ValueError('Expected a direct HTML page.')
        chunks, size = [], 0
        for chunk in response.iter_content(32768):
            size += len(chunk)
            if size > 2 * 1024 * 1024:
                raise ValueError('Page exceeds the import size limit.')
            chunks.append(chunk)
        parser = InstitutionParser()
        parser.feed(b''.join(chunks).decode('utf-8', errors='replace'))
        return parser


def import_site(index_path, snapshot_dir):
    index_path, snapshot_dir = Path(index_path), Path(snapshot_dir)
    if not index_path.is_file():
        raise ValueError('An existing source-article index is required.')
    homepage = fetch(HOME)
    urls = [HOME]
    for href, label in homepage.links:
        url = urljoin(HOME, href).split('#')[0]
        if label.casefold() in SECTIONS and allowed(url) and url not in urls:
            urls.append(url)
    stamp = now()
    records = []
    for url in urls[:12]:
        page = homepage if url == HOME else fetch(url)
        text = page.document()
        if len(text) < 150:
            raise ValueError('Page has insufficient usable source text: ' + url)
        records.append({'title': IDENTITY + ' — ' + ' '.join(page.page_title),
                        'text': text, 'url': url, 'publisher': 'gist.edu.in',
                        'fetched_at': stamp, 'published_at': page.published_at,
                        'content_hash': hashlib.sha256(text.encode('utf-8')).hexdigest()})
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    # Preserve a recoverable copy before changing the existing corpus.
    backup = snapshot_dir / 'source-index-before-import.sqlite'
    if not backup.exists():
        with sqlite3.connect(index_path) as original, sqlite3.connect(backup) as saved:
            original.backup(saved)
    (snapshot_dir / 'source-pages.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    with sqlite3.connect(index_path) as db:
        for record in records:
            old = db.execute('SELECT id,title,text FROM documents WHERE url=?', (record['url'],)).fetchall()
            for row_id, title, text in old:
                db.execute("INSERT INTO documents_fts(documents_fts,rowid,title,text) VALUES('delete',?,?,?)", (row_id, title, text))
                db.execute('DELETE FROM documents WHERE id=?', (row_id,))
            cursor = db.execute('INSERT INTO documents(content_hash,title,text,url,publisher,fetched_at,published_at) VALUES (?,?,?,?,?,?,?)',
                                tuple(record[key] for key in ('content_hash', 'title', 'text', 'url', 'publisher', 'fetched_at', 'published_at')))
            db.execute('INSERT INTO documents_fts(rowid,title,text) VALUES (?,?,?)', (cursor.lastrowid, record['title'], record['text']))
        count = db.execute('SELECT count(*) FROM documents').fetchone()[0]
        db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)', ('document_count', str(count)))
        db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)', ('institution_imported_at', stamp))
        db.execute("INSERT INTO documents_fts(documents_fts,rank) VALUES('integrity-check',1)")
    audit = {'institution': IDENTITY, 'imported_at': stamp, 'page_count': len(records),
             'urls': urls[:12], 'total_documents': count, 'backup': str(backup),
             'policy': 'Original institutional statements with source URLs and collection dates; no fake/real training labels. Collection time is not publication time.'}
    (snapshot_dir / 'import-audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', default='models/dataset-articles-v2/evidence.sqlite')
    parser.add_argument('--snapshots', default='data/gist-source-import')
    args = parser.parse_args()
    print(json.dumps(import_site(args.index, args.snapshots), indent=2))
