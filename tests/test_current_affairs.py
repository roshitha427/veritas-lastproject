"""Synthetic source fixtures; these are not a factual office-holder dataset."""
import sqlite3
from unittest.mock import patch

import pytest

pa = pytest.importorskip('pyarrow', reason='Parquet corpus preparation requires training dependencies.')
import pyarrow.parquet as pq

from backend.current_affairs import _office_context, official_url, search_current_affairs
from backend.verification import verify
from ml.import_current_affairs import build


def make_index(tmp_path):
    directory = tmp_path / 'input'
    directory.mkdir()
    # Changed, fictional names ensure the code reads evidence instead of memorizing a claim.
    good = {'title': 'Example Pradesh leadership', 'text':
            'Official leadership directory. ' * 50 + '\nExample Pradesh\n'
            'Aruna Example\nChief Minister\nBala Sample\nDeputy Chief Minister\n'
            'State leadership appointments and responsibilities.',
            'url': 'https://example.gov.in/leadership', 'fetched_at': '2026-10-01T00:00:00+00:00'}
    rows = [good, dict(good), dict(good, text='Untrusted political text. ' * 20,
                                   url='https://example.gov.in.evil.test/story'),
            dict(good, text='HTTP-only political text. ' * 20, url='http://example.gov.in/story')]
    for i in range(7):
        pq.write_table(pa.Table.from_pylist(rows if i == 0 else [] , schema=pa.Table.from_pylist(rows).schema),
                       directory / f'train-{i:05d}.parquet')
    output = tmp_path / 'model/evidence.sqlite'
    audit = build(directory, output)
    return output, audit


def test_import_deduplicates_and_keeps_original_provenance(tmp_path):
    path, audit = make_index(tmp_path)
    assert audit['counts']['input_rows'] == 4
    assert audit['counts']['indexed_documents'] == 1
    assert audit['counts']['duplicate_documents'] == 1
    assert audit['counts']['excluded_non_https_or_non_government_url'] == 2
    assert len(audit['sources']) == 7
    assert all(len(item['sha256']) == 64 for item in audit['sources'])
    with sqlite3.connect(path) as db:
        assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'


def test_official_snapshot_returns_role_context_and_distinguishes_collection_date(tmp_path):
    path, _ = make_index(tmp_path)
    result = search_current_affairs('Example Pradesh chief minister is Bala Sample', [{'domain': 'gov.in'}], path)
    evidence = result['evidence'][0]
    assert 'Aruna Example' in evidence['snippet']
    assert 'Deputy Chief Minister' in evidence['snippet']
    assert evidence['fetched_at'] == '2026-10-01T00:00:00+00:00'
    assert evidence['reviewed_at'] is None and evidence['rating'] == ''
    assert result['status'] == 'needs_review'
    assert 'not a publication date' in evidence['relationship']


def test_disabled_specific_source_overrides_enabled_government_root(tmp_path):
    path, _ = make_index(tmp_path)
    sources = [{'domain': 'gov.in'}, {'domain': 'example.gov.in', 'enabled': False}]
    assert not search_current_affairs('Example Pradesh chief minister Bala Sample', sources, path)['evidence']
    assert not search_current_affairs('Example Pradesh chief minister', [], path)['evidence']


def test_matching_corpus_avoids_network_but_none_mode_remains_disabled(tmp_path):
    path, _ = make_index(tmp_path)
    with patch('backend.current_affairs.DEFAULT_PATH', path), \
         patch('requests.get', side_effect=AssertionError('A matching local search must not use the network')):
        result = verify('Example Pradesh chief minister is Bala Sample', 'en',
                        [{'domain': 'gov.in'}], provider='gdelt')
        assert result['evidence'][0]['kind'] == 'official_source_snapshot'
        assert not verify('Example Pradesh chief minister', 'en',
                          [{'domain': 'gov.in'}], provider='none')['evidence']


def test_search_operators_are_data_and_missing_index_is_safe(tmp_path):
    path, _ = make_index(tmp_path)
    result = search_current_affairs('" OR NOT * : chief minister', [{'domain': 'gov.in'}], path)
    assert result['provider_status'] == 'searched'
    assert search_current_affairs('chief minister', [], tmp_path / 'absent')['provider_status'] == 'not_configured'
    assert not official_url('https://user:password@example.gov.in/private')
    assert not official_url('https://example.gov.in.evil.test/story')


def test_office_lookup_finds_holder_without_matching_the_alleged_name(tmp_path):
    path, _ = make_index(tmp_path)
    # Enough name-heavy distractors to crowd the original OR query's top 80.
    with sqlite3.connect(path) as db:
        for index in range(100):
            db.execute('INSERT INTO documents (content_hash,title,text,url,publisher,fetched_at) '
                       'VALUES (?,?,?,?,?,?)', (str(index), 'Example Pradesh Daya Wrong',
                       'Example Pradesh Daya Wrong agricultural programme. ' * 5,
                       f'https://example.gov.in/crops/{index}', 'example.gov.in', '2026-10-01T00:00:00Z'))
        db.execute("INSERT INTO documents_fts(documents_fts) VALUES('rebuild')")
    for claim in ('The current chief minister of Example Pradesh is Daya Wrong',
                  'Current Example Pradesh chief minister is Daya Wrong',
                  'current Example Pradesh cm is Daya Wrong',
                  'Current Example Pradesh cheif minister is Daya Wrong',
                  'The current cheif minister of Example Pradesh is Daya Wrong'):
        evidence = search_current_affairs(claim, [{'domain': 'gov.in'}], path)['evidence']
        assert evidence[0]['url'] == 'https://example.gov.in/leadership'
        assert 'Aruna Example' in evidence[0]['snippet']
        assert evidence[0]['relevance_score'] >= .7


def test_misspelled_chief_minister_preserves_the_office_and_not_the_alleged_holder():
    claim = 'current andhra pradesh cheif minister is pawan kalyan'
    assert _office_context(claim) == ('chief minister', ['andhra', 'pradesh'])
    assert _office_context('current AP cheif minister is Daya Wrong') == ('chief minister', ['andhra', 'pradesh'])
    assert _office_context('current AP chief minister is Aruna Example') == ('chief minister', ['andhra', 'pradesh'])
