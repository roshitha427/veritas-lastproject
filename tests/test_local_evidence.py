import json

import pytest

from backend.local_evidence import verify_local
from backend.verification import verify
from ml.build_evidence import build


def corpus(tmp_path):
    # Test fixtures, not a shipped evidence corpus or a factual-truth dataset.
    rows = [
        {'title': 'Council budget announcement', 'claim': 'City council approved the new public transport budget on Monday',
         'url': 'https://trusted.example/council', 'publisher': 'Trusted test publisher', 'rating': 'Test rating only', 'reviewed_at': '2025-01-02'},
        {'title': 'Unrelated test record', 'claim': 'Volcanic mineral zircon crystals formed before the ice age',
         'url': 'https://other.example/geology', 'publisher': 'Other test publisher', 'rating': '', 'reviewed_at': '2024-12-01'},
        {'title': 'चंद्रमा की जाँच', 'claim': 'चंद्रमा की तस्वीर का संदर्भ गलत है',
         'url': 'https://trusted.example/hindi', 'publisher': 'Trusted test publisher', 'rating': 'Test only', 'reviewed_at': '2025-01-01'},
    ]
    source = tmp_path / 'corpus.json'
    source.write_text(json.dumps(rows, ensure_ascii=False), encoding='utf8')
    output = tmp_path / 'index.joblib'
    build(source, output, 'test-v1')
    return output


def test_local_retrieval_is_offline_and_source_filtered(tmp_path, monkeypatch):
    index = corpus(tmp_path)
    def network_forbidden(*args, **kwargs):
        pytest.fail('Local inference must not call external evidence services')
    monkeypatch.setattr('requests.get', network_forbidden)
    sources = [{'domain': 'trusted.example', 'enabled': True}]
    result = verify('City council approved the public transport budget on Monday', 'en', sources,
                    provider='local', evidence_path=index)
    assert result['provider'] == 'Local evidence model'
    assert result['status'] == 'needs_review'
    assert result['evidence'][0]['url'] == 'https://trusted.example/council'
    assert result['evidence'][0]['rating'] == 'Test rating only'
    assert result['corpus']['version'] == 'test-v1'
    assert result['corpus']['documents'] == 3
    assert verify_local('zircon crystals volcanic mineral', sources, index)['evidence'] == []
    assert verify_local('City council budget', [{'domain': 'trusted.example', 'enabled': False}], index)['evidence'] == []


def test_local_retrieval_handles_scripts_missing_and_corrupt_index(tmp_path):
    index = corpus(tmp_path)
    result = verify_local('चंद्रमा की तस्वीर का संदर्भ गलत है', [{'domain': 'trusted.example'}], index)
    assert result['evidence'][0]['url'].endswith('/hindi')
    assert verify_local('claim', [], tmp_path / 'missing')['provider_status'] == 'not_configured'
    assert verify_local('', [], index)['provider_status'] == 'no_claim'
    bad = tmp_path / 'broken.joblib'
    bad.write_text('broken', encoding='utf8')
    assert verify_local('claim', [], bad)['provider_status'] == 'unavailable'


def test_evidence_builder_rejects_undated_and_credential_urls(tmp_path):
    source = tmp_path / 'corpus.json'
    row = {'title': 'Test', 'claim': 'Test claim', 'url': 'https://user:password@trusted.example/story',
           'publisher': 'Fixture', 'reviewed_at': '2025-01-01'}
    source.write_text(json.dumps([row, dict(row, claim='Other claim')]), encoding='utf8')
    with pytest.raises(ValueError, match='without credentials'):
        build(source, tmp_path / 'out.joblib', 'test')
    row['url'] = 'https://trusted.example/story'
    row.pop('reviewed_at')
    source.write_text(json.dumps([row, dict(row, claim='Other claim')]), encoding='utf8')
    with pytest.raises(ValueError, match='reviewed_at'):
        build(source, tmp_path / 'out.joblib', 'test')
