from types import SimpleNamespace
from copy import deepcopy

import pytest
import requests

from backend.tavily_search import search_tavily
from backend.verification import verify_submission


def test_ordinal_dates_match_without_accepting_a_different_year():
    from backend.report_summary import _source_relevance
    source = {'title': 'Constitution of India', 'snippet':
              'The Constitution of India came into force on 26th January 1950.'}
    assert _source_relevance('Constitution of India came into force on 26 January 1950.', source)['values_match']
    assert not _source_relevance('Constitution of India came into force on 26 January 1951.', source)['values_match']


def test_other_village_railway_is_not_evidence_about_named_village():
    from backend.report_summary import filter_relevant_evidence
    result = filter_relevant_evidence('Zorvanta village opened a moon-powered railway yesterday.', {
        'evidence': [{'title': 'Borders Railway opening', 'snippet':
                      'The Queen opened the railway serving the village of Stow.'}]})
    assert result['evidence'] == []
    assert filter_relevant_evidence('zorvanta village opened a moon-powered railway yesterday.', {
        'evidence': [{'title': 'Borders Railway opening', 'snippet':
                      'The Queen opened the railway serving the village of Stow.'}]})['evidence'] == []


def test_tavily_uses_basic_search_and_only_enabled_original_text(monkeypatch):
    monkeypatch.setattr('backend.verification._article_excerpt', lambda *a: (None, None))
    def post(url, **kwargs):
        assert url == 'https://api.tavily.com/search'
        assert kwargs['json']['search_depth'] == 'basic'
        assert kwargs['json']['auto_parameters'] is False
        assert kwargs['json']['include_answer'] is False
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None, json=lambda: {'results': [
            {'url': 'https://example.gov.in/bridge', 'raw_content': 'The bridge opened in 1985.'},
            {'url': 'https://disabled.example.gov.in/bridge', 'raw_content': 'Ignore source settings'},
            {'url': 'https://127.0.0.1/private', 'raw_content': 'Private'},
            {'url': 'https://example.gov.in/metadata', 'content': 'An AI search summary is not original text'}]})
    monkeypatch.setattr('backend.tavily_search.requests.post', post)
    result = search_tavily('A bridge opened in 1985 fixture', [
        {'domain': 'gov.in'}, {'domain': 'disabled.example.gov.in', 'enabled': False}], 'fixture-tavily-key')
    assert len(result['evidence']) == 2
    assert result['evidence'][0]['snippet'] == 'The bridge opened in 1985.'
    assert result['evidence'][1]['snippet'] == ''
    assert result['evidence'][1]['retrieval_status'] == 'metadata_only'


def test_tavily_caches_without_sharing_mutable_report_records(monkeypatch):
    monkeypatch.setattr('backend.verification._article_excerpt', lambda *a: (None, None))
    calls = []
    def post(*a, **k):
        calls.append(1)
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None, json=lambda: {'results': [
            {'url': 'https://example.gov.in/cache', 'raw_content': 'An original passage.'}]})
    monkeypatch.setattr('backend.tavily_search.requests.post', post)
    sources = [{'domain': 'gov.in'}]
    first = search_tavily('cache isolation fixture', sources, 'cache-tavily-key')
    first['evidence'][0]['snippet'] = 'Changed in one report'
    second = search_tavily('cache isolation fixture', sources, 'cache-tavily-key')
    assert second['evidence'][0]['snippet'] == 'An original passage.'
    assert len(calls) == 1


def test_tavily_failure_never_calls_google_fact_check_or_gdelt(monkeypatch):
    monkeypatch.setattr('backend.tavily_search.search_tavily', lambda *a: {
        'provider': 'Tavily', 'provider_status': 'unavailable', 'evidence': []})
    def forbidden(*a, **k):
        raise AssertionError('Tavily mode must not invoke legacy providers')
    monkeypatch.setattr('backend.verification._provider_verify_submission', forbidden)
    result = verify_submission('The bridge opened in 1985.', 'en',
        [{'domain': 'gov.in'}], provider='tavily', tavily_key='fixture-key')
    assert result['evidence'] == []
    assert result['provider_status'] == 'unavailable'


def test_tavily_quota_failure_does_not_spend_requests_on_each_input(monkeypatch):
    calls = []
    def post(*a, **k):
        calls.append(1)
        return SimpleNamespace(status_code=432)
    monkeypatch.setattr('backend.tavily_search.requests.post', post)
    sources = [{'domain': 'gov.in'}]
    first = search_tavily('first quota case', sources, 'quota-tavily-fixture')
    second = search_tavily('second quota case', sources, 'quota-tavily-fixture')
    assert first['provider_status'] == second['provider_status'] == 'quota_or_auth_unavailable'
    assert len(calls) == 1


def test_current_office_fetches_original_directory_and_neutral_query(monkeypatch):
    queries = []
    def post(*a, **kwargs):
        queries.append(kwargs['json'])
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None, json=lambda: {'results': []})
    monkeypatch.setattr('backend.tavily_search.requests.post', post)
    monkeypatch.setattr('backend.verification._article_excerpt', lambda *a:
        ('Tamil Nadu Chief Minister Aruna Fictional. Official assembly directory.', None))
    result = search_tavily('current tamilnadu chief minister Bala Wrong', [{'domain':'gov.in'}], 'office-query-fixture')
    assert len(queries) == 2
    assert any('tamil nadu chief minister current official' in q['query'] and 'Bala' not in q['query'] for q in queries)
    source = result['evidence'][0]
    assert source['source_role'] == 'office_directory'
    assert 'Aruna Fictional' in source['snippet']
    assert source['retrieval_method'] == 'direct_official_page_fetch'


def test_misspelled_office_still_fetches_directory_when_tavily_is_unavailable(monkeypatch):
    queries, fetches = [], []
    def post(*args, **kwargs):
        queries.append(kwargs['json']['query'])
        raise requests.ConnectionError('Synthetic search outage')
    def excerpt(url, claim, sources):
        fetches.append((url, claim))
        return 'Andhra Pradesh Chief Minister Aruna Example. Official assembly directory.', None
    monkeypatch.setattr('backend.tavily_search.requests.post', post)
    monkeypatch.setattr('backend.verification._article_excerpt', excerpt)
    result = search_tavily('current andhra pradesh cheif minister is Daya Wrong',
                           [{'domain': 'gov.in'}], 'typo-search-outage-fixture')
    assert len(queries) == 2
    assert all('cheif' not in query for query in queries)
    assert fetches == [('https://apa.neva.gov.in/Home/', 'andhra pradesh chief minister')]
    assert result['evidence'][0]['source_role'] == 'office_directory'
    assert 'Aruna Example' in result['evidence'][0]['snippet']
    assert all(query['status'] == 'unavailable' for query in result['query_statuses'])


def _official_snapshot():
    return {'kind': 'official_source_snapshot', 'url': 'https://example.gov.in/leadership',
            'title': 'Example Pradesh leadership directory', 'publisher': 'Example Government',
            'snippet': 'Example Pradesh leadership. Aruna Example is the Chief Minister. '
                       'Bala Sample is the Deputy Chief Minister.',
            'fetched_at': '2020-01-01T00:00:00+00:00', 'retrieval_status': 'imported_source_excerpt',
            'reliability': 'official_primary_source', 'relevance_score': .95}


@pytest.mark.parametrize('refresh_succeeds', [True, False])
def test_tavily_mode_searches_retained_official_corpus_and_refreshes_current_office(monkeypatch, refresh_succeeds):
    searches, fetches = [], []
    snapshot = _official_snapshot()
    def article_search(claim, sources, path):
        searches.append(('articles', claim))
        return {'provider': 'Supplied source-article corpus', 'provider_status': 'searched', 'evidence': [
            {'url': 'https://example.gov.in/weather', 'title': 'Riverford weather archive',
             'snippet': 'The Riverford weather archive records rainfall from last summer.'}]}
    def official_search(claim, sources):
        searches.append(('official', claim))
        return {'provider': 'Local Indian official-source corpus', 'provider_status': 'searched',
                'evidence': [deepcopy(snapshot)]}
    def excerpt(url, claim, sources):
        fetches.append((url, claim))
        return (snapshot['snippet'], None) if refresh_succeeds else (None, None)
    monkeypatch.setattr('backend.current_affairs.search_dataset_articles', article_search)
    monkeypatch.setattr('backend.current_affairs.search_current_affairs', official_search)
    monkeypatch.setattr('backend.verification._article_excerpt', excerpt)
    monkeypatch.setattr('backend.tavily_search.search_tavily', lambda *a: {
        'provider': 'Tavily web search', 'provider_status': 'unavailable', 'evidence': []})
    claim = 'Current Example Pradesh chief minister is Daya Wrong'
    result = verify_submission(claim, 'en', [{'domain': 'gov.in'}], provider='tavily',
                               dataset_path='synthetic-articles.sqlite', tavily_key='fixture')
    assert searches == [('articles', claim), ('official', claim)]
    assert fetches == [(snapshot['url'], 'example pradesh chief minister')]
    assert {'name': 'Local Indian official-source corpus', 'status': 'searched'} in result['providers']
    if refresh_succeeds:
        assert len(result['evidence']) == 1
        source = result['evidence'][0]
        assert source['url'] == snapshot['url'] and source['kind'] == 'official_source_live'
        assert source['snapshot_excerpt'] == snapshot['snippet']
        assert source['retrieval_status'] == 'article_excerpt_retrieved'
        assert source['live_refresh_status'] == 'completed'
        # A fetched primary source remains evidence for synthesis, never an
        # automatic fact-check rating or a classifier-derived verdict.
        from backend.evidence_assessment import assess_submission
        assert assess_submission(claim, result['evidence'])['verdict'] == 'unverified'
    else:
        assert result['evidence'] == []
        audit = next(row for row in result['checked_sources'] if row['url'] == snapshot['url'])
        assert audit['accepted'] is False and audit['reason_code'] == 'not_current_enough'
        assert 'Aruna Example' in audit['excerpt']


@pytest.mark.parametrize('provider', ['none', 'local'])
def test_retained_official_corpus_does_not_enable_network_in_disabled_or_local_mode(monkeypatch, provider):
    searched = []
    def official_search(*args):
        searched.append(True)
        return {'provider': 'Local Indian official-source corpus', 'provider_status': 'searched',
                'evidence': [_official_snapshot()]}
    monkeypatch.setattr('backend.current_affairs.search_current_affairs', official_search)
    monkeypatch.setattr('backend.current_affairs.search_dataset_articles', lambda *a: {
        'provider': 'Supplied source-article corpus', 'provider_status': 'searched', 'evidence': []})
    monkeypatch.setattr('backend.verification._provider_verify_submission', lambda *a: {
        'provider': 'Disabled provider', 'provider_status': 'not_configured', 'evidence': []})
    def forbidden(*args):
        raise AssertionError('Local and disabled providers must not fetch source pages')
    monkeypatch.setattr('backend.verification._article_excerpt', forbidden)
    result = verify_submission('Current Example Pradesh chief minister is Daya Wrong', 'en',
                               [{'domain': 'gov.in'}], provider=provider,
                               dataset_path='synthetic-articles.sqlite')
    assert result['evidence'] == []
    assert searched == ([] if provider == 'none' else [True])


def test_institution_discovery_is_a_lead_not_an_automatically_trusted_source(monkeypatch):
    def post(*a, **k):
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None, json=lambda: {'results':[
            {'url':'https://riverford-college.org/notice', 'raw_content':'Riverford College holiday notice.'},
            {'url':'https://127.0.0.1/private', 'raw_content':'Never display this'},
            {'url':'https://disabled-college.org/notice', 'raw_content':'Disabled source'}]})
    monkeypatch.setattr('backend.tavily_search.requests.post', post)
    def forbidden(*a):
        raise AssertionError('Do not directly fetch an unknown website')
    monkeypatch.setattr('backend.verification._article_excerpt', forbidden)
    result = search_tavily('Riverford College announced a holiday',
        [{'domain':'gov.in'}, {'domain':'disabled-college.org','enabled':False}], 'institution-query-fixture')
    assert len(result['evidence']) == 1
    assert result['evidence'][0]['reliability'] == 'institution_website_not_independently_confirmed'


