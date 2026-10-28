from copy import deepcopy
from types import SimpleNamespace

from backend.evidence_policy import repair_unqualified_evidence


def old_result():
    claim = 'The Riverford bridge opened in 1985.'
    return claim, {'ai_percentage': 12., 'confidence': 88.,
        'verification': {'evidence': [{'title': 'Riverford bridge record',
            'url': 'https://example.gov.in/bridge', 'kind': 'official_source_snapshot',
            'snippet': 'The Riverford bridge opened to the public in 1985.'}]},
        'report_summary': {'schema_version': 5, 'verdict': 'true', 'provider': 'Local evidence comparison',
            'reason': 'The local model predicted support.', 'gemini_analysis': {'status': 'not_configured'}}}


def test_saved_unqualified_model_settlement_is_repaired_without_external_calls(monkeypatch):
    import requests
    def no_network(*args, **kwargs):
        raise AssertionError('Saved evidence policy upgrades cannot access the network')
    monkeypatch.setattr(requests, 'post', no_network)
    monkeypatch.setattr(requests, 'get', no_network)
    claim, result = old_result()
    verifier = SimpleNamespace(settlement_qualification=lambda: {'qualified': False})
    assert repair_unqualified_evidence(result, claim, 'text', verifier)
    assert result['report_summary']['verdict'] == 'unverified'
    assert result['retired_evidence_comparison']['verdict'] == 'true'
    assert result['confidence'] == 88. and result['ai_percentage'] == 12.
    once = deepcopy(result)
    assert not repair_unqualified_evidence(result, claim, 'text', verifier)
    assert result == once


def test_qualified_or_separately_grounded_verdicts_are_preserved():
    claim, result = old_result()
    original = deepcopy(result)
    verifier = SimpleNamespace(settlement_qualification=lambda: {'qualified': True})
    assert not repair_unqualified_evidence(result, claim, 'text', verifier)
    assert result == original
    result['report_summary']['gemini_analysis']['status'] = 'completed'
    verifier = SimpleNamespace(settlement_qualification=lambda: {'qualified': False})
    original = deepcopy(result)
    assert not repair_unqualified_evidence(result, claim, 'text', verifier)
    assert result == original
