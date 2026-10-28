import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from backend.generation import media_origin_assessment
from backend.video_model import VideoModel


def test_video_detector_rejects_face_swap_checkpoint(tmp_path):
    (tmp_path / 'model.joblib').write_bytes(b'not-loaded-in-readiness')
    path = tmp_path / 'model-card.json'
    path.write_text(json.dumps({'purpose': 'temporal'}))
    assert not VideoModel(tmp_path / 'model.joblib').ready()
    path.write_text(json.dumps({'purpose': 'video_synthetic_or_manipulated'}))
    assert not VideoModel(tmp_path / 'model.joblib').ready()
    path.write_text(json.dumps({'purpose': 'video_synthetic_or_manipulated', 'prediction_enabled': True}))
    assert VideoModel(tmp_path / 'model.joblib').ready()


def test_video_training_fit_is_identified_and_never_verified_origin(tmp_path, monkeypatch):
    model = VideoModel(tmp_path / 'model.joblib')
    monkeypatch.setattr(model, 'ready', lambda: True)
    monkeypatch.setattr(model, 'predict', lambda images: (.94, {
        'version': 'test-video', 'scope': 'Synthetic or manipulated video only.',
        'personal_training_fit': [{'sha256': 'training-clip'}]}))
    result = model.assess([Image.new('RGB', (32, 32))] * 12, 'training-clip')
    assert result['training_example']
    assert result['synthetic_or_manipulated_percentage'] == 94
    assert 'training fit' in result['note']
    origin = media_origin_assessment(1, {'prediction_enabled': True}, kind='video', video_assessment=result)
    assert origin['basis'] == 'video_consistency_features'
    assert origin['verdict'] == 'inconclusive'
    assert not origin['scores_publishable']
    assert 'AI/manipulation-like' in origin['headline']
    assert 'training fit' in origin['reason']


@pytest.mark.parametrize('score', [float('nan'), float('inf'), -.1, 1.1])
def test_invalid_video_score_is_rejected(tmp_path, monkeypatch, score):
    model = VideoModel(tmp_path / 'model.joblib')
    monkeypatch.setattr(model, 'ready', lambda: True)
    monkeypatch.setattr(model, 'predict', lambda images: (score, {}))
    with pytest.raises(ValueError, match='Invalid video prediction'):
        model.assess([None] * 3)


def test_video_model_runs_without_faces_and_preserves_other_scores(app, tmp_path, monkeypatch):
    import cv2
    from backend.pdf_report import report_pdf
    path = tmp_path / 'no-faces.mp4'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (96, 96))
    assert writer.isOpened()
    for value in range(20):
        writer.write(np.full((96, 96, 3), value * 10, dtype=np.uint8))
    writer.release()
    analyzer = app.extensions['analyzer']
    analyzer.generation = SimpleNamespace(ready=lambda: True, predict=lambda images: (
        [.2] * len(images), {'source_url': 'https://example.invalid/model'}))
    calls = []

    def assess(images, digest):
        calls.append((len(images), digest))
        return {'status': 'experimental', 'synthetic_or_manipulated_percentage': 93.,
                'real_class_percentage': 7., 'sampled_frames': len(images),
                'scope': 'Test video-specific signal.', 'note': 'Training fit only.',
                'training_example': True, 'model_version': 'test-video'}

    analyzer.video = SimpleNamespace(ready=lambda: True, assess=assess)
    monkeypatch.setattr('backend.analysis.detected_face_counts', lambda images: [0] * len(images))
    result = analyzer.analyze('video', '', path, 'en', [])
    assert calls == [(12, result['sha256'])]
    assert result['video_assessment']['synthetic_or_manipulated_percentage'] == 93
    assert result['ai_percentage'] == 20  # Still-image scores remain separate.
    assert result['verification']['status'] == 'not_applicable'
    assert result['media_origin']['verdict'] == 'inconclusive'
    pdf = report_pdf({'id': 'test-video', 'type': 'video', 'created_at': '2026-10-22',
                      'content': '', 'result': result})
    assert pdf.startswith(b'%PDF')
