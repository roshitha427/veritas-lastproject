from copy import deepcopy

from PIL import Image

from backend.content_credentials import AI_SOURCE, inspect_content_credentials, summarize_credentials
from backend.generation import media_origin_assessment


def fixture():
    return {'active_manifest': 'active', 'validation_state': 'Invalid',
            'validation_status': [{'code': 'signingCredential.untrusted'}],
            'validation_results': {'activeManifest': {'success': [
                {'code': 'claimSignature.validated'}, {'code': 'assertion.dataHash.match'}]}},
            'manifests': {'active': {'assertions': [{'label': 'c2pa.actions.v2',
                'data': {'actions': [{'action': 'c2pa.created', 'digitalSourceType': AI_SOURCE}]}}]}}}


def test_untrusted_ai_declaration_is_distinct_from_verified_origin():
    result = summarize_credentials(fixture())
    assert result['ai_declared']
    assert result['status'] == 'unverified'
    origin = media_origin_assessment(8., {'prediction_enabled': True}, 'cartoon_or_illustration', result)
    assert origin['verdict'] == 'ai_content_declared'
    assert origin['basis'] == 'content_credentials'
    assert not origin['credential_verified']
    assert origin['score'] is None
    assert 'not independently verified' in origin['reason']


def test_invalid_binding_or_bad_signature_cannot_override_model():
    for code in ('assertion.dataHash.mismatch', 'claimSignature.mismatch'):
        data = fixture()
        data['validation_status'].append({'code': code})
        assert not summarize_credentials(data)['ai_declared']
    data = fixture()
    data['validation_results'] = {}
    assert not summarize_credentials(data)['ai_declared']


def test_only_active_or_linked_components_are_examined():
    data = fixture()
    data['manifests']['unrelated'] = deepcopy(data['manifests']['active'])
    data['manifests']['active']['assertions'] = []
    assert not summarize_credentials(data)['ai_declared']
    data['manifests']['active']['ingredients'] = [{'active_manifest': 'unrelated'}]
    result = summarize_credentials(data)
    assert result['ai_declared'] and result['scope'] == 'source_component'


def test_credential_absence_does_not_mean_real(tmp_path):
    path = tmp_path/'plain.png'
    Image.new('RGB', (30,30)).save(path)
    result = inspect_content_credentials(path)
    assert not result['ai_declared']
    assert 'real' not in result


def test_trusted_clean_validation_is_reported_separately():
    data = fixture()
    data['validation_state'] = 'Trusted'
    data['validation_status'] = []
    assert summarize_credentials(data)['status'] == 'verified'
