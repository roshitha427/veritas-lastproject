import sqlite3

from backend.current_affairs import search_dataset_articles
from ml.import_institution_site import HOME, InstitutionParser, allowed, import_site


def test_imported_institution_pages_are_searchable_repeatable_and_source_controlled(tmp_path, monkeypatch):
    page = InstitutionParser()
    page.feed('<title>About GIST</title><script>Invented instructions</script>'
              '<h1>Geethanjali Institute of Science and Technology</h1>'
              '<p>GIST was established in 2008 by Ushodaya Educational Society. '
              'The college provides engineering education near Nellore in Andhra Pradesh.</p>'
              '<h2>Campus Hub</h2><a href="https://evil.example/">Ignored footer</a>')
    monkeypatch.setattr('ml.import_institution_site.fetch', lambda url: page)
    path = tmp_path / 'evidence.sqlite'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE documents(id INTEGER PRIMARY KEY,content_hash TEXT UNIQUE,title TEXT,text TEXT,url TEXT,publisher TEXT,fetched_at TEXT,published_at TEXT)')
        db.execute('CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT)')
        db.execute("CREATE VIRTUAL TABLE documents_fts USING fts5(title,text,content='documents',content_rowid='id')")
    snapshots = tmp_path / 'snapshots'
    assert import_site(path, snapshots)['page_count'] == 1
    assert import_site(path, snapshots)['total_documents'] == 1
    result = search_dataset_articles('GIST was established in 2008', [{'domain': 'edu.in'}], path)
    record = result['evidence'][0]
    assert record['url'] == HOME
    assert '2008' in record['snippet']
    assert 'Invented instructions' not in record['snippet']
    assert 'Ignored footer' not in record['snippet']
    assert record['reliability'] == 'academic_source'
    assert record['rating'] == ''
    assert not search_dataset_articles('GIST established 2008', [{'domain': 'gist.edu.in', 'enabled': False}], path)['evidence']
    with sqlite3.connect(snapshots / 'source-index-before-import.sqlite') as db:
        assert db.execute('SELECT count(*) FROM documents').fetchone()[0] == 0


def test_institution_import_stays_on_selected_https_site():
    assert allowed(HOME)
    for url in ['http://gist.edu.in/gist/about/', 'https://gist.edu.in.evil.example/gist/',
                'https://gist.edu.in/gist/download.zip', 'https://gist.edu.in/gist/?login=1',
                'https://user:password@gist.edu.in/gist/']:
        assert not allowed(url)
