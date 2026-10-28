from io import BytesIO
import sqlite3
import zipfile

from backend.current_affairs import search_dataset_articles
from backend.verification import verify_submission
from ml.import_factcheck_articles import build


def test_original_article_is_indexed_without_turning_training_label_into_verdict(tmp_path, monkeypatch):
    archive = tmp_path / 'dataset.zip'
    with zipfile.ZipFile(archive, 'w') as out:
        out.writestr('train/Fake/Fake_text.xlsx', b'fixture')
    article = 'The Riverford bridge opened to the public in 1985. Residents attended the opening ceremony. ' * 3
    monkeypatch.setattr('ml.import_factcheck_articles.read_xlsx', lambda payload: iter([{
        'url': 'https://www.boomlive.in/fact-check/bridge', 'title': 'Riverford bridge opening',
        'content': article, 'date': '21 April 2020', 'label': 'fake'}]))
    path = tmp_path / 'articles.sqlite'
    audit = build(archive, path)
    assert audit['documents'] == 1
    source = [{'domain': 'boomlive.in'}]
    found = search_dataset_articles('The Riverford bridge opened in 1985', source, path)
    assert article[:60] in found['evidence'][0]['snippet']
    assert found['evidence'][0]['rating'] == ''
    assert found['evidence'][0]['kind'] == 'fact_check_article'
    assert not search_dataset_articles('Riverford bridge', [{'domain': 'boomlive.in', 'enabled': False}], path)['evidence']
    monkeypatch.setattr('requests.get', lambda *a, **k: (_ for _ in ()).throw(AssertionError('Strong local coverage must remain offline')))
    combined = verify_submission('The Riverford bridge opened in 1985', 'en', source,
                                provider='gdelt', dataset_path=path)
    assert combined['evidence'][0]['url'].endswith('/bridge')
    assert combined['provider'] == 'Supplied source-article corpus'
