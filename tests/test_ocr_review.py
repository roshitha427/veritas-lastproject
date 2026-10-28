"""Automatic OCR selection and optional legacy correction preserve raw readings."""
import base64
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from backend.media_inspection import _image_part, inspect_media, select_automatic_ocr_claim
from tests.conftest import register, token
from tests.test_api import wait_report
from tests.test_media_inspection import response


ORIGINAL = 'Elon Musk will apay free Tesla vehicles.\nApply now!'
CORRECTED = 'Elon Musk will give away free Tesla vehicles.\nApply now!'


def image_bytes(color='white'):
    output = BytesIO()
    Image.new('RGB', (120, 80), color).save(output, 'PNG')
    return output.getvalue()


def inspection():
    return {'status': 'completed', 'description': 'A poster with a vehicle.',
        'visible_text': ORIGINAL, 'text_role': 'claim', 'ocr_performed': True,
        'visual_type': 'screenshot_or_document', 'visual_artifacts': '',
        'main_claim': ORIGINAL.splitlines()[0], 'main_headline': '',
        'ocr_uncertain': True, 'uncertain_spans': [{'text': 'apay', 'reason': 'Small blurred lettering.'}],
        'ocr_warnings': ['Check the uncertain words.']}


def preview(client, auth, payload=None):
    return client.post('/api/media/ocr-preview', data={'type': 'image',
        'file': (BytesIO(payload or image_bytes()), 'poster.png')},
        headers={'X-CSRF-Token': auth['csrf_token']})


def submit(client, auth, preview_token='', payload=None, corrected=CORRECTED):
    data = {'type': 'image', 'ocr_reviewed': 'true', 'reviewed_ocr_text': corrected,
        'file': (BytesIO(payload or image_bytes()), 'poster.png')}
    if preview_token:
        data['ocr_preview_token'] = preview_token
    return client.post('/api/analyze', data=data, headers={'X-CSRF-Token': auth['csrf_token']})


def test_ocr_preserves_wording_and_marks_uncertainty_without_rewriting(monkeypatch):
    payload = inspection()
    payload.update(visible_text='  Free [unclear] vehicles!\nOriginal spellling.  ',
        main_headline='Free [unclear] vehicles!',
        main_claim='Elon Musk is giving away free Tesla vehicles.',
        uncertain_spans=[{'text': '[unclear]', 'reason': 'Cropped word.'},
                         {'text': 'invented', 'reason': 'Not in transcription.'}])
    monkeypatch.setattr('backend.media_inspection.requests.post', lambda *args, **kwargs: response(payload))
    result = inspect_media([Image.new('RGB', (50, 50))], 'test')
    assert result['visible_text'] == payload['visible_text']
    assert result['main_headline'] == 'Free [unclear] vehicles!'
    assert result['main_claim'] == ''  # It must be an exact excerpt, not a paraphrase.
    assert result['ocr_uncertain'] is True
    assert result['uncertain_spans'] == [{'text': '[unclear]', 'reason': 'Cropped word.'}]
    assert result['ocr_warnings']
    assert 'before verifying' not in ' '.join(result['ocr_warnings'])
    assert 'confirm' not in ' '.join(result['ocr_warnings']).lower()


def test_ocr_encoding_preserves_more_small_text_detail_with_a_fixed_bound():
    part = _image_part(Image.new('RGB', (3200, 1600)))
    with Image.open(BytesIO(base64.b64decode(part['inline_data']['data']))) as encoded:
        assert encoded.size == (1600, 800)


def test_preview_requires_authentication_and_csrf(client, app):
    assert client.post('/api/media/ocr-preview').status_code == 403
    assert client.post('/api/media/ocr-preview', headers={'X-CSRF-Token': token(client)}).status_code == 401
    assert app.extensions['store'].count('reports') == 0


def test_preview_reads_only_ocr_and_removes_the_temporary_upload(client, app, monkeypatch):
    auth = register(client)
    files = []
    import backend.app as app_module
    original_temp = app_module.tempfile.NamedTemporaryFile
    def temporary(*args, **kwargs):
        handle = original_temp(*args, **kwargs)
        files.append(handle.name)
        return handle
    monkeypatch.setattr(app_module.tempfile, 'NamedTemporaryFile', temporary)
    monkeypatch.setattr('backend.media_inspection.inspect_media', lambda *args: inspection())
    monkeypatch.setattr(app.extensions['analyzer'], 'analyze', lambda *args, **kwargs: pytest.fail('Preview must not verify a claim'))
    result = preview(client, auth)
    assert result.status_code == 200
    assert result.json['inspection']['visible_text'] == ORIGINAL
    assert result.json['preview_token']
    assert app.extensions['store'].count('reports') == 0
    assert files and all(not Path(path).exists() for path in files)


@pytest.mark.parametrize(('filename', 'content', 'kind'), [
    ('poster.svg', b'<svg/>', 'image'),
    ('poster.png', b'not an image', 'image'),
    ('poster.png', b'', 'image'),
    ('clip.mp4', b'video', 'video'),
])
def test_preview_rejects_unsupported_or_invalid_uploads(client, filename, content, kind):
    auth = register(client)
    result = client.post('/api/media/ocr-preview', data={'type': kind,
        'file': (BytesIO(content), filename)}, headers={'X-CSRF-Token': auth['csrf_token']})
    assert result.status_code == 400


def test_preview_uses_analysis_pixel_limit_and_request_size_limit(client, app, monkeypatch):
    auth = register(client)
    # A declared oversized image is rejected before any decompressed pixels load.
    class OversizedImage:
        format, width, height = 'PNG', 6000, 6000
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def verify(self): pytest.fail('Oversized pixels must be rejected first')
    monkeypatch.setattr('PIL.Image.open', lambda *args: OversizedImage())
    assert preview(client, auth).status_code == 400
    app.config['MAX_CONTENT_LENGTH'] = 100
    assert preview(client, auth).status_code == 413


@pytest.mark.parametrize('field_update', [
    {'ocr_reviewed': 'false'}, {'type': 'video'}, {'type': 'text'},
    {'reviewed_ocr_text': 'x' * 4001}, {'ocr_preview_token': 'invalid-token'},
])
def test_review_fields_require_explicit_valid_image_review(client, field_update):
    auth = register(client)
    data = {'type': 'image', 'ocr_reviewed': 'true', 'reviewed_ocr_text': CORRECTED,
        'file': (BytesIO(image_bytes()), 'poster.png')}
    data.update(field_update)
    result = client.post('/api/analyze', data=data, headers={'X-CSRF-Token': auth['csrf_token']})
    assert result.status_code == 400


def test_signed_preview_cannot_be_reused_for_another_image_or_user(client, app, monkeypatch):
    auth = register(client)
    monkeypatch.setattr('backend.media_inspection.inspect_media', lambda *args: inspection())
    result = preview(client, auth)
    preview_token = result.json['preview_token']
    assert submit(client, auth, preview_token, payload=image_bytes('black')).status_code == 400
    other_client = app.test_client()
    other_auth = register(other_client, 'other@example.com')
    assert submit(other_client, other_auth, preview_token).status_code == 400
    assert app.extensions['store'].count('reports') == 0


def test_preview_expires_before_it_can_be_reused(client, app, monkeypatch):
    from itsdangerous import TimestampSigner, URLSafeTimedSerializer
    auth = register(client)
    monkeypatch.setattr('backend.media_inspection.inspect_media', lambda *args: inspection())
    result = preview(client, auth)
    signer = URLSafeTimedSerializer(app.config['SECRET_KEY'], salt='image-ocr-preview-v1')
    payload = signer.loads(result.json['preview_token'])
    class OldTimestampSigner(TimestampSigner):
        def get_timestamp(self): return super().get_timestamp() - 3600
    old_signer = URLSafeTimedSerializer(app.config['SECRET_KEY'], salt='image-ocr-preview-v1', signer=OldTimestampSigner)
    assert submit(client, auth, old_signer.dumps(payload)).status_code == 400


@pytest.mark.parametrize('use_preview', [False, True])
def test_reviewed_ocr_is_authoritative_for_verification_and_raw_is_preserved(client, app, monkeypatch, use_preview):
    auth = register(client)
    monkeypatch.setattr('backend.media_inspection.inspect_media', lambda *args: inspection())
    if use_preview:
        preview_token = preview(client, auth).json['preview_token']
        monkeypatch.setattr('backend.analysis.inspect_media', lambda *args: pytest.fail('Do not rerun reviewed OCR'))
    else:
        preview_token = ''
        monkeypatch.setattr('backend.analysis.inspect_media', lambda *args: inspection())
    checked = []
    def verify(claim, *args, **kwargs):
        checked.append(claim)
        return {'status': 'unverified', 'provider_status': 'not_configured', 'evidence': []}
    monkeypatch.setattr('backend.analysis.verify_submission', verify)
    monkeypatch.setattr(app.extensions['analyzer'], 'text', lambda text: (.987, {'name': 'Test', 'status': 'ready'}))
    queued = submit(client, auth, preview_token)
    assert queued.status_code == 202
    report = wait_report(client, queued.json['id'])
    assert report['status'] == 'complete'
    result = report['result']
    assert checked == [CORRECTED]
    assert result['claim_text'] == CORRECTED
    assert result['ocr_text'] == CORRECTED
    assert result['media_inspection']['visible_text'] == ORIGINAL
    assert result['media_inspection']['corrected_text'] == CORRECTED
    assert result['ocr_review'] == {'user_reviewed': True, 'original_text': ORIGINAL,
        'corrected_text': CORRECTED, 'changed': True}
    assert result['media_text_analysis']['text_model_assessment']['fake_percentage'] == 98.7
    assert result['report_summary']['verdict'] == 'unverified'


def test_blank_review_suppresses_wrong_ocr_claim(client, app, monkeypatch):
    auth = register(client)
    monkeypatch.setattr('backend.analysis.inspect_media', lambda *args: inspection())
    monkeypatch.setattr('backend.analysis.verify_submission', lambda *args, **kwargs: pytest.fail('No corrected claim remains'))
    result = submit(client, auth, corrected='')
    report = wait_report(client, result.json['id'])
    assert report['status'] == 'complete'
    assert report['result']['claim_text'] == ''
    assert report['result']['ocr_review']['original_text'] == ORIGINAL
    assert report['result']['report_summary']['mode'] == 'media_only'


def test_unreviewed_search_selects_main_claim_without_publisher_banners(client, app, monkeypatch):
    auth = register(client)
    scene = inspection()
    scene['visible_text'] = 'Global News\nBREAKING NEWS\n' + ORIGINAL
    monkeypatch.setattr('backend.analysis.inspect_media', lambda *args: scene)
    checked = []
    def verify(claim, *args, **kwargs):
        checked.append(claim)
        return {'status': 'unverified', 'provider_status': 'not_configured', 'evidence': []}
    monkeypatch.setattr('backend.analysis.verify_submission', verify)
    result = client.post('/api/analyze', data={'type': 'image',
        'file': (BytesIO(image_bytes()), 'poster.png')}, headers={'X-CSRF-Token': auth['csrf_token']})
    report = wait_report(client, result.json['id'])
    assert report['status'] == 'complete'
    assert checked == [scene['main_claim']]
    assert report['result']['media_inspection']['visible_text'] == scene['visible_text']


def test_corrections_invalidate_the_analysis_snapshot(app, monkeypatch):
    analyzer = app.extensions['analyzer']
    calls = []
    monkeypatch.setattr(analyzer, '_analyze', lambda *args, **kwargs: calls.append(kwargs) or {})
    analyzer.analyze('image', '', None, 'en', [], reviewed_ocr={'text': ORIGINAL})
    analyzer.analyze('image', '', None, 'en', [], reviewed_ocr={'text': CORRECTED})
    assert len(calls) == 2


HEADLINE = 'Elon Musk Announces Free Tesla Cars for Everyone in 2025'


@pytest.mark.parametrize('body,uncertain_spans,expected', [
    ('Tech billionaire has announced plans to give away free Tesla vehicles next year.', [], HEADLINE),
    ('Elon Musk will give away free Tesla vehicles next year.', [], HEADLINE),
    ('Tech billionaire announced plans to apay free Tesla vehicles.', [{'text': 'apay'}], HEADLINE),
    ('Elon Musk announced free Tesla cars for everyone in 2025 during a company event.', [],
     'Elon Musk announced free Tesla cars for everyone in 2025 during a company event.'),
])
def test_automatic_claim_keeps_self_contained_headline_context_without_rewriting(body, uncertain_spans, expected):
    scene = {'visible_text': HEADLINE + '\n' + body, 'main_headline': HEADLINE,
             'main_claim': body, 'uncertain_spans': uncertain_spans}
    raw = scene['visible_text']
    assert select_automatic_ocr_claim(scene) == expected
    assert scene['visible_text'] == raw
    assert '2026' not in select_automatic_ocr_claim(scene)


def test_automatic_selection_does_not_use_an_invented_excerpt_or_replace_unclear_words():
    raw = 'Free [unclear] vehicles next year.'
    scene = {'visible_text': raw, 'main_claim': 'Elon Musk offers everyone free Tesla cars in 2025.',
             'main_headline': '', 'uncertain_spans': [{'text': '[unclear]'}]}
    assert select_automatic_ocr_claim(scene) == raw
    clear_body = 'Elon Musk announced a Tesla promotion for eligible customers.'
    scene = {'visible_text': 'Elon [unclear] Announces Free Tesla Cars\n' + clear_body,
             'main_headline': 'Elon [unclear] Announces Free Tesla Cars', 'main_claim': clear_body,
             'uncertain_spans': [{'text': '[unclear]'}]}
    assert select_automatic_ocr_claim(scene) == clear_body


@pytest.mark.parametrize('role', ['user', 'admin'])
def test_automatic_image_api_needs_no_review_fields_and_checks_one_contextual_claim(client, app, monkeypatch, role):
    auth = register(client)
    if role == 'admin':
        store = app.extensions['store']
        account = store.one('users', {'id': auth['user']['id']})
        account['role'] = 'admin'
        store.save('users', account)
    body = 'Tech billionaire has announced plans to apay free Tesla vehicles next year.'
    scene = inspection()
    raw = 'Global News\nBREAKING NEWS\n' + HEADLINE + '\n' + body + '\nFAKE NEWS'
    scene.update(visible_text=raw, main_headline=HEADLINE, main_claim=body,
                 ocr_warnings=['Some automatically extracted text is uncertain.'])
    vision_calls = []
    def automatic_inspection(*args):
        vision_calls.append(1)
        return scene
    monkeypatch.setattr('backend.analysis.inspect_media', automatic_inspection)
    checked = []
    def verify(claim, *args, **kwargs):
        checked.append(claim)
        return {'status': 'unverified', 'provider_status': 'not_configured', 'evidence': []}
    monkeypatch.setattr('backend.analysis.verify_submission', verify)
    queued = client.post('/api/analyze', data={'type': 'image',
        'file': (BytesIO(image_bytes()), 'news.png')}, headers={'X-CSRF-Token': auth['csrf_token']})
    assert queued.status_code == 202
    report = wait_report(client, queued.json['id'])
    assert report['status'] == 'complete'
    result = report['result']
    assert vision_calls == [1]
    assert checked == [HEADLINE]
    assert result['claim_text'] == HEADLINE
    assert result['ocr_text'] == raw
    assert result['media_inspection']['visible_text'] == raw
    assert 'ocr_review' not in result
    assert result['report_summary']['verdict'] == 'unverified'
    review = result['report_summary']['verification_review']
    assert any('uncertain' in reason for reason in review['insufficiency_reasons'])
    guidance = ' '.join(review['insufficiency_reasons'] + review['additional_evidence_needed'])
    assert 'confirmed by the user' not in guidance
    assert 'user-confirmed' not in guidance
