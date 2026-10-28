from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from backend.result_cache import ResultCache


def test_repeated_results_are_copied_and_expire():
    clock = [0]
    cache = ResultCache(ttl=5, clock=lambda: clock[0])
    calls = []
    def compute():
        calls.append(1)
        return {'sources': [{'title': 'Original'}]}
    first = cache.get_or_compute('claim', compute)
    first['sources'][0]['title'] = 'Changed by report'
    assert cache.get_or_compute('claim', compute)['sources'][0]['title'] == 'Original'
    assert len(calls) == 1
    clock[0] = 5
    cache.get_or_compute('claim', compute)
    assert len(calls) == 2


def test_parallel_identical_requests_share_one_computation():
    cache = ResultCache()
    started, release = threading.Event(), threading.Event()
    calls = []
    def compute():
        calls.append(1)
        started.set()
        assert release.wait(5)
        return {'verdict': 'true'}
    with ThreadPoolExecutor(max_workers=4) as executor:
        first = executor.submit(cache.get_or_compute, 'same', compute)
        assert started.wait(5)
        others = [executor.submit(cache.get_or_compute, 'same', compute) for _ in range(3)]
        release.set()
        assert [job.result() for job in [first, *others]] == [{'verdict': 'true'}] * 4
    assert len(calls) == 1


def test_errors_and_rejected_results_allow_retry():
    cache = ResultCache()
    def fail():
        raise ValueError('Temporary failure')
    with pytest.raises(ValueError):
        cache.get_or_compute('same', fail)
    assert cache.get_or_compute('same', lambda: 'outage', lambda value: False) == 'outage'
    assert cache.get_or_compute('same', lambda: 'recovered') == 'recovered'


def test_capacity_evicts_old_entries():
    cache = ResultCache(capacity=1)
    cache.get_or_compute('first', lambda: 1)
    cache.get_or_compute('second', lambda: 2)
    assert cache.get_or_compute('first', lambda: 3) == 3


def test_analyzer_reuses_exact_input_but_invalidates_changed_context(app, tmp_path, monkeypatch):
    analyzer = app.extensions['analyzer']
    calls = []
    def analyze(*args):
        calls.append(args)
        return {'report_summary': {'verdict': 'true'}, 'ai_percentage': 0.6}
    monkeypatch.setattr(analyzer, '_analyze', analyze)
    claim = "India's Chandrayaan-3 mission successfully landed near the Moon's south polar region on August 23, 2023."
    sources = [{'domain': 'isro.gov.in', 'enabled': True}]
    first = analyzer.analyze('text', claim, None, 'en', sources)
    assert analyzer.analyze('text', claim, None, 'en', sources) == first
    assert len(calls) == 1
    analyzer.analyze('text', claim + ' Not true.', None, 'en', sources)
    analyzer.analyze('text', claim, None, 'hi', sources)
    analyzer.analyze('text', claim, None, 'en', [dict(sources[0], enabled=False)])
    assert len(calls) == 4
    # An artifact revision must not keep a previous model's result alive.
    model = tmp_path / 'ai-text.joblib'
    model.write_bytes(b'new model revision')
    analyzer.analyze('text', claim, None, 'en', sources)
    assert len(calls) == 5
    analyzer.config['GEMINI_MODEL'] = 'different-model'
    analyzer.analyze('text', claim, None, 'en', sources)
    assert len(calls) == 6


def test_identical_media_bytes_share_snapshot_and_changed_bytes_do_not(app, tmp_path, monkeypatch):
    analyzer = app.extensions['analyzer']
    calls = []
    monkeypatch.setattr(analyzer, '_analyze', lambda *args: (calls.append(args) or {}))
    first, second = tmp_path / 'a.mp4', tmp_path / 'b.mp4'
    first.write_bytes(b'same video bytes')
    second.write_bytes(first.read_bytes())
    assert analyzer.analyze('video', '', first, 'en', []) == analyzer.analyze('video', '', second, 'en', [])
    assert len(calls) == 1
    second.write_bytes(b'different video bytes')
    analyzer.analyze('video', '', second, 'en', [])
    assert len(calls) == 2


@pytest.mark.parametrize('result', [
    {'verification': {'provider_status': 'unavailable'}},
    {'report_summary': {'gemini_analysis': {'status': 'response_rejected'}}},
    {'verification': {'providers': [{'status': 'quota_or_auth_unavailable'}]}},
])
def test_outage_reports_are_not_reused(app, monkeypatch, result):
    analyzer = app.extensions['analyzer']
    calls = []
    monkeypatch.setattr(analyzer, '_analyze', lambda *args: (calls.append(args) or result))
    analyzer.analyze('text', 'A claim with a provider outage.', None, 'en', [])
    analyzer.analyze('text', 'A claim with a provider outage.', None, 'en', [])
    assert len(calls) == 2
