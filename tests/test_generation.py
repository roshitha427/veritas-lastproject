"""AI-origin estimates and manipulation diagnostics remain separate and fail independently."""
from types import SimpleNamespace

from PIL import Image
import pytest


def install_visual(analyzer):
    analyzer.visual = SimpleNamespace(ready=lambda: True, predict=lambda images: (
        [.23] * len(images), {'name': 'Visual test fixture', 'version': 'test', 'scope': 'Visual patterns'}))


def reviewed_origin_fixture():
    return {'locally_validated': True, 'deployment_ready': True, 'decision_policy': {
        'validation_status': 'validated', 'validation_dataset_sha256': 'a' * 64,
        'visual_types': ['photograph'], 'real_max_percentage': 20., 'ai_min_percentage': 80.}}


@pytest.mark.parametrize('score, expected_confidence', [(.82, 82.), (.05, 95.), (.5, 50.)])
def test_generation_is_independent_of_fake_real_and_confidence(app, tmp_path, score, expected_confidence):
    analyzer = app.extensions['analyzer']
    install_visual(analyzer)
    analyzer.generation = SimpleNamespace(ready=lambda: True, predict=lambda images: (
        [score] * len(images), {'name': 'Generation fixture', 'version': 'test',
        'purpose': 'ai_generation', 'source_url': 'https://example.org/model', 'limitations': ['Uncalibrated test score']}))
    image = tmp_path / 'example.png'
    Image.new('RGB', (120, 120), '#4d6972').save(image)
    result = analyzer.analyze('image', '', image, 'en', [])
    assert result['fake_percentage'] == round(score * 100, 1)
    assert result['real_percentage'] == round((1 - score) * 100, 1)
    assert result['confidence'] == expected_confidence
    assert 'uncalibrated' in result['confidence_note']
    assert result['visual_model_assessment']['fake_percentage'] == 23.
    assert result['ai_percentage'] == round(score * 100, 1)
    assert result['ai_assessment'] == {'status': 'experimental', 'calibrated': False,
        'locally_validated': False,
        'scope': 'image', 'sample_count': 1, 'source_url': 'https://example.org/model'}
    assert result['verification']['status'] == 'not_applicable'
    assert result['media_origin']['verdict'] == 'inconclusive'
    assert result['report_summary']['headline'] == 'Origin inconclusive'
    assert 'classifies this as likely AI-generated' not in result['report_summary']['summary']


@pytest.mark.parametrize('score, validated, verdict', [
    (99.9, False, 'inconclusive'), (0.1, False, 'inconclusive'),
    (None, False, 'classification_unavailable'), (50., True, 'inconclusive'),
    (85., True, 'ai_generated'), (15., True, 'likely_real'),
])
def test_origin_policy_withholds_scores_when_origin_is_unresolved(score, validated, verdict):
    from backend.generation import media_origin_assessment
    origin = media_origin_assessment(score, reviewed_origin_fixture() if validated else {}, 'photograph')
    assert origin['verdict'] == verdict
    assert origin['score'] == (score if verdict in ('ai_generated', 'likely_real') else None)
    assert origin['scores_publishable'] == (verdict in ('ai_generated', 'likely_real'))


@pytest.mark.parametrize('score', [12., 88., 50., 51.9, 99.9])
def test_prediction_enablement_cannot_bypass_independent_validation(score):
    from backend.generation import media_origin_assessment
    origin = media_origin_assessment(score, {'prediction_enabled': True, 'locally_validated': False})
    assert origin['verdict'] == 'inconclusive'
    assert 'not been locally validated' in origin['reason']
    assert origin['score'] is None


@pytest.mark.parametrize('score', [None, float('nan'), float('inf'), -1., 101., True, '95'])
def test_invalid_origin_scores_never_produce_a_binary_label(score):
    from backend.generation import media_origin_assessment
    origin = media_origin_assessment(score, reviewed_origin_fixture(), 'photograph')
    assert origin['verdict'] == 'classification_unavailable'
    assert origin['score'] is None


@pytest.mark.parametrize('score,verdict', [(20., 'likely_real'), (20.1, 'inconclusive'),
                                         (51.9, 'inconclusive'), (79.9, 'inconclusive'), (80., 'ai_generated')])
def test_reviewed_thresholds_keep_the_uncertain_band_inconclusive(score, verdict):
    from backend.generation import media_origin_assessment
    origin = media_origin_assessment(score, reviewed_origin_fixture(), 'photograph')
    assert origin['verdict'] == verdict


@pytest.mark.parametrize('missing', ['deployment_ready', 'decision_policy', 'locally_validated'])
def test_incomplete_activation_metadata_cannot_enable_binary_origin(missing):
    from backend.generation import media_origin_assessment
    assessment = reviewed_origin_fixture()
    assessment.pop(missing)
    assert media_origin_assessment(99., assessment, 'photograph')['verdict'] == 'inconclusive'


def test_thresholds_require_valid_provenance_order_and_media_scope():
    from backend.generation import media_origin_assessment
    assessment = reviewed_origin_fixture()
    assert media_origin_assessment(99., assessment, 'screenshot_or_document')['verdict'] == 'inconclusive'
    assessment['decision_policy']['validation_dataset_sha256'] = 'missing'
    assert media_origin_assessment(99., assessment, 'photograph')['verdict'] == 'inconclusive'
    assessment = reviewed_origin_fixture()
    assessment['decision_policy']['real_max_percentage'] = 90.
    assert media_origin_assessment(99., assessment, 'photograph')['verdict'] == 'inconclusive'


def test_reviewed_model_policy_is_preserved_through_image_analysis(app, tmp_path, monkeypatch):
    analyzer = app.extensions['analyzer']
    install_visual(analyzer)
    card = dict(reviewed_origin_fixture(), prediction_enabled=True,
                source_url='https://example.org/reviewed-fixture', version='reviewed-fixture')
    analyzer.generation = SimpleNamespace(ready=lambda: True,
        predict=lambda images: ([.9] * len(images), card))
    monkeypatch.setattr('backend.analysis.inspect_media', lambda *args, **kwargs: {
        'status': 'completed', 'visual_type': 'photograph', 'description': 'A test scene.'})
    path = tmp_path / 'reviewed.png'
    Image.new('RGB', (120, 120)).save(path)
    result = analyzer.analyze('image', '', path, 'en', [])
    assert result['media_origin']['verdict'] == 'ai_generated'
    assert result['ai_assessment']['decision_policy'] == card['decision_policy']
    assert result['ai_assessment']['deployment_ready'] is True


@pytest.mark.parametrize('score', [float('nan'), float('inf'), -.1, 1.1, True])
def test_invalid_model_output_is_unassessed_without_losing_visual_analysis(app, tmp_path, score):
    analyzer = app.extensions['analyzer']
    install_visual(analyzer)
    analyzer.generation = SimpleNamespace(ready=lambda: True,
        predict=lambda images: ([score] * len(images), {'source_url': 'https://example.org/invalid'}))
    path = tmp_path / 'invalid-score.png'
    Image.new('RGB', (80, 80)).save(path)
    result = analyzer.analyze('image', '', path, 'en', [])
    assert result['ai_percentage'] is None
    assert result['ai_assessment']['status'] == 'unavailable'
    assert result['media_origin']['verdict'] == 'classification_unavailable'
    assert result['visual_model_assessment']['fake_percentage'] == 23.


@pytest.mark.parametrize('score', [1., 35., 99.9])
def test_illustration_style_cannot_establish_origin_even_with_prediction_enabled(score):
    from backend.generation import media_origin_assessment
    origin = media_origin_assessment(score, {'prediction_enabled': True}, 'cartoon_or_illustration')
    assert origin['verdict'] == 'inconclusive'
    assert origin['score'] is None
    assert 'human-drawn' in origin['reason']
    assert 'not been validated' in origin['reason']


def test_illustration_analysis_keeps_raw_metrics_without_claiming_real(app, tmp_path, monkeypatch):
    analyzer = app.extensions['analyzer']
    install_visual(analyzer)
    analyzer.generation = SimpleNamespace(ready=lambda: True, predict=lambda images: (
        [.08] * len(images), {'prediction_enabled': True, 'source_url': 'https://example.org/test', 'version': 'fixture'}))
    monkeypatch.setattr('backend.analysis.inspect_media', lambda *args: {
        'status': 'completed', 'visual_type': 'cartoon_or_illustration',
        'description': 'An illustrated portrait.', 'visible_text': ''})
    path = tmp_path/'artwork.png'
    Image.new('RGB', (120,120)).save(path)
    result = analyzer.analyze('image', '', path, 'en', [])
    assert result['ai_percentage'] == 8.
    assert result['real_percentage'] == 92.
    assert result['media_origin']['verdict'] == 'inconclusive'
    assert 'likely real' not in result['report_summary']['summary'].lower()
    assert 'human-drawn' in result['report_summary']['summary']


def test_origin_prediction_and_both_class_scores_agree(app, tmp_path):
    analyzer = app.extensions['analyzer']
    install_visual(analyzer)
    analyzer.generation = SimpleNamespace(ready=lambda: True, predict=lambda images: (
        [.12] * len(images), {'source_url': 'https://example.org/evaluated',
        'prediction_enabled': True, 'version': 'trained-test', 'limitations': []}))
    image = tmp_path / 'photo.png'
    Image.new('RGB', (120, 120), 'white').save(image)
    result = analyzer.analyze('image', '', image, 'en', [])
    assert result['real_percentage'] == 88.
    assert result['ai_percentage'] == 12.
    assert result['confidence'] == 88.
    assert result['media_origin']['verdict'] == 'inconclusive'
    assert result['ai_assessment']['model_version'] == 'trained-test'


@pytest.mark.parametrize('verdict', ['inconclusive', 'ai_generated'])
def test_pdf_origin_metrics_use_the_same_three_labels_for_every_verdict(monkeypatch, verdict):
    import backend.pdf_report as module
    tables = []
    original = module.Table

    def capture(rows, *args, **kwargs):
        tables.append(rows)
        return original(rows, *args, **kwargs)

    monkeypatch.setattr(module, 'Table', capture)
    pdf = module.report_pdf({'id': 'origin-fixture', 'created_at': '2026-10-21', 'type': 'image', 'title': 'Origin test', 'result': {
        'ai_percentage': 100., 'real_percentage': 0., 'confidence': 100.,
        'media_origin': {'verdict': verdict},
    }})
    assert pdf.startswith(b'%PDF')
    assert tables == [[['Measure', 'Result'], ['Real image pattern score', '0.0%'],
                       ['AI image pattern score', '100.0%'], ['Image model confidence', '100.0%']]]


def test_replaced_checkpoint_requires_a_new_upload_and_preserves_evidence(tmp_path):
    import json
    from backend.generation import retire_obsolete_origin, media_origin_assessment
    (tmp_path / 'model-card.json').write_text(json.dumps({'version': 'new', 'prediction_enabled': True}))
    result = {'ai_percentage': 100., 'confidence': 100., 'real_percentage': 0.,
        'ai_assessment': {'source_url': 'https://huggingface.co/Ateeqq/ai-vs-human-image-detector'},
        'verification': {'status': 'false'}, 'media_inspection': {'visible_text': 'A sign'},
        'report_summary': {'verdict': 'false', 'model_signal': 'AI-origin score 100%.'},
        'frames': [{'seconds': 1, 'ai_percentage': 100., 'fake_percentage': 20.}]}
    assert retire_obsolete_origin(result, tmp_path)
    assert result['ai_percentage'] is None
    assert result['retired_ai_assessment']['ai_percentage'] == 100.
    assert result['verification']['status'] == 'false'
    assert result['media_inspection']['visible_text'] == 'A sign'
    assert '100%' not in result['report_summary']['model_signal']
    assert result['report_summary']['verdict'] == 'false'
    assert result['frames'] == [{'seconds': 1, 'fake_percentage': 20.}]
    origin = media_origin_assessment(None, result['ai_assessment'])
    assert origin['verdict'] == 'classification_unavailable'
    assert 'Upload this file again' in origin['reason']
    assert not retire_obsolete_origin(result, tmp_path)


def test_local_cnn_origin_uses_its_own_head_and_rejects_manipulation_checkpoint(tmp_path, monkeypatch):
    import json
    from backend.generation import GenerationDetector
    for name in ('model.joblib','backbone.pt'):
        (tmp_path/name).write_bytes(b'fixture')
    card = {'architecture':'cnn_origin','purpose':'image_manipulation','version':'fixture'}
    (tmp_path/'model-card.json').write_text(json.dumps(card))
    detector = GenerationDetector(tmp_path)
    assert not detector.ready()
    card['purpose'] = 'ai_generation'
    (tmp_path/'model-card.json').write_text(json.dumps(card))
    monkeypatch.setattr('ml.vision.VisualModel',lambda path: SimpleNamespace(
        predict=lambda images: ([.18]*len(images),card)))
    assert detector.ready()
    scores, returned = detector.predict([Image.new('RGB',(20,20))]*3)
    assert scores == [.18,.18,.18]
    assert returned['purpose'] == 'ai_generation'


def test_generation_failure_preserves_other_analysis(app, tmp_path):
    analyzer = app.extensions['analyzer']
    install_visual(analyzer)
    def broken(images):
        raise RuntimeError('Unreadable local checkpoint')
    analyzer.generation = SimpleNamespace(ready=lambda: True, predict=broken)
    image = tmp_path / 'example.png'
    Image.new('RGB', (80, 80), 'white').save(image)
    result = analyzer.analyze('image', '', image, 'en', [])
    assert result['fake_percentage'] is None
    assert result['visual_model_assessment']['fake_percentage'] == 23.
    assert result['ai_percentage'] is None
    assert result['ai_assessment']['status'] == 'unavailable'
    assert result['sha256']


def test_text_never_borrows_an_image_or_fake_score(app):
    analyzer = app.extensions['analyzer']
    analyzer.text = lambda text: (.91, {'name': 'Text fixture', 'version': 'test', 'status': 'ready'})
    def must_not_run(images):
        pytest.fail('Image detector cannot be applied to writing')
    analyzer.generation = SimpleNamespace(ready=lambda: True, predict=must_not_run)
    result = analyzer.analyze('text', 'An English news claim.', None, 'en', [])
    assert result['fake_percentage'] == 91.
    assert result['ai_percentage'] is None
    assert any('AI-writing' in limitation for limitation in result['limitations'])


def test_camera_watermark_is_shown_but_not_fact_checked(app, tmp_path, monkeypatch):
    analyzer = app.extensions['analyzer']
    install_visual(analyzer)
    analyzer.generation = SimpleNamespace(ready=lambda: True, predict=lambda images: (
        [.999], {'name': 'Unvalidated detector', 'source_url': 'https://example.org/model', 'limitations': []}))
    monkeypatch.setattr('backend.analysis.inspect_media', lambda *args, **kwargs: {
        'status': 'completed', 'visible_text': 'REDMI ON JOYEE', 'description': 'People sit at a table.',
        'visual_type': 'photograph', 'visual_artifacts': 'Camera watermark.'})
    monkeypatch.setattr('backend.analysis.verify_submission', lambda *args, **kwargs: pytest.fail('Watermarks are not claims'))
    image = tmp_path / 'photo.png'
    Image.new('RGB', (120, 120), 'white').save(image)
    result = analyzer.analyze('image', '', image, 'en', [])
    assert result['ocr_text'] == 'REDMI ON JOYEE'
    assert result['claim_text'] == ''
    assert result['verification']['status'] == 'not_applicable'
    assert result['report_summary']['mode'] == 'media_only'
    assert result['media_origin']['verdict'] == 'inconclusive'


def test_camera_overlay_filter_preserves_actual_claims():
    from backend.analysis import media_claim_text
    assert media_claim_text('REDMI ON JOYEE') == ''
    assert media_claim_text('Shot on Redmi\nIndia won the world cup.') == 'India won the world cup.'
    assert media_claim_text('Modi is prime minister') == 'Modi is prime minister'
    assert media_claim_text('Redmi launched a new phone.') == 'Redmi launched a new phone.'


def test_ocr_claim_uses_text_models_separately_from_visual_origin(app, tmp_path, monkeypatch):
    analyzer = app.extensions['analyzer']
    install_visual(analyzer)
    analyzer.generation = SimpleNamespace(ready=lambda: True, predict=lambda images: (
        [.9], {'source_url': 'https://example.org/model', 'limitations': []}))
    analyzer.text = lambda text: (.2, {'status': 'ready', 'name': 'Text fixture'})
    analyzer.ai_text = SimpleNamespace(ready=True, predict=lambda text: (.35, {
        'status': 'ready', 'experimental_input': True}))
    monkeypatch.setattr('backend.analysis.inspect_media', lambda *args: {
        'status': 'completed', 'visible_text': 'The city opened a new public library.',
        'description': 'A sign on a building.', 'visible_features': ['A printed sign.'], 'text_role': 'claim'})
    image = tmp_path / 'claim.png'
    Image.new('RGB', (120, 120)).save(image)
    result = analyzer.analyze('image', '', image, 'en', [])
    assert result['ai_percentage'] == 90.
    writing = result['media_text_analysis']
    assert writing['text_model_assessment']['fake_percentage'] == 20.
    assert writing['ai_percentage'] == 35.
    assert writing['text_metrics']['character_count'] == len(writing['content'])
    assert writing['text_metrics']['word_count'] == 7
    assert result['media_details']['width'] == 120


def test_evidence_outage_keeps_text_metrics(app, monkeypatch):
    analyzer = app.extensions['analyzer']
    analyzer.text = lambda text: (.2, {'status': 'ready'})
    def unavailable(*args, **kwargs):
        raise requests.ConnectionError()
    import requests
    monkeypatch.setattr('backend.analysis.verify_submission', unavailable)
    result = analyzer.analyze('text', 'A library opened today.', None, 'en', [])
    assert result['fake_percentage'] == 20.
    assert result['text_metrics']['character_count'] == 23
    assert result['verification']['provider_status'] == 'unavailable'
    assert result['report_summary']['verdict'] == 'unverified'
