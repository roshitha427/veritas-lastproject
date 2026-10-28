from copy import deepcopy

import pytest

from backend.media_labels import CONFIRMED_VIDEO_ORIGINS, media_labels


@pytest.mark.parametrize('digest', list(CONFIRMED_VIDEO_ORIGINS))
def test_confirmed_origin_does_not_replace_model_scores(digest):
    result = {'sha256': digest, 'ai_percentage': 46.7, 'confidence': 53.3,
              'media_origin': {'verdict': 'inconclusive'}}
    before = deepcopy(result)
    labels = media_labels('video', result)
    assert labels['confirmed_origin'] == 'ai_generated'
    assert labels['provenance'] == 'user_confirmed'
    assert result == before


def test_names_and_image_uploads_do_not_match_video_confirmations():
    assert not media_labels('video', {'filename': 'WhatsApp Video 2026-10-19 at 11.00.44 PM.mp4'})
    assert not media_labels('image', {'sha256': next(iter(CONFIRMED_VIDEO_ORIGINS))})


@pytest.mark.parametrize('style,label', [('animation', 'Animated'), ('cartoon', 'Cartoon')])
def test_visual_style_is_not_an_ai_origin_label(style, label):
    labels = media_labels('video', {'media_inspection': {'presentation_type': style}})
    assert labels == {'style': label}


def test_deepfake_signal_remains_separate_from_generation():
    labels = media_labels('video', {'temporal_assessment': {'status': 'ready', 'fake_percentage': 90}})
    assert labels == {'manipulation': 'Possible deepfake (face-swap model signal)'}
    assert not media_labels('video', {'temporal_assessment': {'status': 'out_of_scope', 'fake_percentage': 90}})


@pytest.mark.parametrize('digest', list(CONFIRMED_VIDEO_ORIGINS))
def test_saved_video_receives_annotation_without_score_changes(client, app, digest):
    from tests.conftest import register
    auth = register(client)
    store = app.extensions['store']
    report = store.save('reports', {
        'owner_id': auth['user']['id'], 'type': 'video', 'status': 'complete', 'content': '',
        'result': {'sha256': digest, 'ai_percentage': 46.7, 'real_percentage': 53.3,
                   'confidence': 53.3, 'media_inspection': {'presentation_type': 'animation'},
                   'report_summary': {'schema_version': 5, 'mode': 'media_only'}}})
    updated = client.get('/api/reports/' + report['id']).json['result']
    assert updated['media_labels']['confirmed_origin'] == 'ai_generated'
    assert updated['media_labels']['style'] in ('Animated', 'Cartoon')
    assert updated['ai_percentage'] == 46.7
    assert updated['confidence'] == 53.3
    assert store.one('reports', {'id': report['id']})['result']['media_labels'] == updated['media_labels']


def test_pdf_explains_confirmed_origin_and_keeps_model_scores(monkeypatch):
    from backend.pdf_report import report_pdf
    import backend.pdf_report as module
    messages, tables = [], []
    paragraph = module.Paragraph
    monkeypatch.setattr(module, 'Paragraph', lambda text, *a, **kw: (messages.append(text) or paragraph(text, *a, **kw)))
    import backend.pdf_report as pdf_module
    table = pdf_module.Table
    monkeypatch.setattr(pdf_module, 'Table', lambda rows, *a, **kw: (tables.append(rows) or table(rows, *a, **kw)))
    result = {'sha256': next(iter(CONFIRMED_VIDEO_ORIGINS)), 'ai_percentage': 46.7,
              'real_percentage': 53.3, 'confidence': 53.3, 'media_origin': {'verdict': 'inconclusive'}}
    result['media_labels'] = media_labels('video', result)
    report_pdf({'id': 'confirmed', 'type': 'video', 'created_at': '2026-10-22', 'result': result})
    assert any('reference label' in text for text in messages)
    assert not any('origin could not be confirmed' in text for text in messages)
    assert tables == [[['Measure', 'Result'], ['Real video frame pattern score', '53.3%'],
                       ['AI video frame pattern score', '46.7%'], ['Video frame model confidence', '53.3%']]]
