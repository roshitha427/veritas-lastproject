from types import SimpleNamespace

from backend.grounded_search import search_grounded, _original_url


def test_grounded_discovery_only_keeps_fetched_enabled_sources(monkeypatch):
    response = SimpleNamespace(status_code=200, raise_for_status=lambda: None,
        json=lambda: {'candidates': [{'groundingMetadata': {'webSearchQueries': ['Example bridge'],
            'groundingChunks': [{'web': {'uri': 'https://example.gov.in/bridge', 'title': 'Bridge'}},
                                {'web': {'uri': 'https://untrusted.example/claim', 'title': 'Do not use'}}]}}]})
    monkeypatch.setattr('backend.grounded_search.requests.post', lambda *a, **k: response)
    monkeypatch.setattr('backend.verification._article_excerpt', lambda *a: ('The Example bridge opened in 1985.', None))
    result = search_grounded('Example bridge opening test', [{'domain': 'example.gov.in'}], 'fixture-key')
    assert len(result['evidence']) == 1
    assert result['evidence'][0]['url'] == 'https://example.gov.in/bridge'
    assert result['evidence'][0]['snippet'] == 'The Example bridge opened in 1985.'


def test_grounding_redirect_cannot_fetch_arbitrary_hosts(monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError('An untrusted host must not be requested')
    monkeypatch.setattr('backend.grounded_search.requests.get', forbidden)
    sources = [{'domain': 'example.gov.in'}]
    assert _original_url('http://127.0.0.1/private', sources) is None
    assert _original_url('https://untrusted.example/private', sources) is None
    assert _original_url('https://user:secret@example.gov.in/story', sources) is None


def test_disabled_specific_source_overrides_enabled_parent():
    sources = [{'domain': 'gov.in'}, {'domain': 'example.gov.in', 'enabled': False}]
    assert _original_url('https://example.gov.in/story', sources) is None


def test_quota_failure_is_reported_and_not_retried_for_each_input(monkeypatch):
    calls = []
    def post(*a, **k):
        calls.append(1)
        return SimpleNamespace(status_code=429)
    monkeypatch.setattr('backend.grounded_search.requests.post', post)
    sources = [{'domain': 'example.gov.in'}]
    first = search_grounded('First quota test', sources, 'quota-fixture-key')
    second = search_grounded('Different quota test', sources, 'quota-fixture-key')
    assert first['provider_status'] == second['provider_status'] == 'quota_or_rate_limit'
    assert len(calls) == 1
