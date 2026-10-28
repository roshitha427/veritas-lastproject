import numpy as np
from PIL import Image
import pytest

from ml.temporal import temporal_features, TemporalModel
from ml.train_temporal import clip_frames


def frame(value):
    return Image.fromarray(np.full((96, 96, 3), value, dtype=np.uint8))


def test_temporal_features_are_order_sensitive_finite_and_repeatable():
    sequence = [frame(value) for value in [0, 40, 80, 120, 160, 200]]
    scores = [.1, .2, .3, .4, .5, .6]
    features = temporal_features(sequence, scores)
    assert features.shape == (525,)
    assert np.isfinite(features).all()
    np.testing.assert_array_equal(features, temporal_features(sequence, scores))
    order = [0, 5, 1, 4, 2, 3]
    shuffled = temporal_features([sequence[index] for index in order], [scores[index] for index in order])
    assert not np.allclose(features, shuffled)


def test_temporal_rejects_missing_order_or_mismatched_scores(tmp_path):
    with pytest.raises(ValueError):
        temporal_features([frame(10)] * 2, [.2, .3])
    with pytest.raises(ValueError):
        temporal_features([frame(10)] * 3, [.2, .3])
    assert not TemporalModel(tmp_path / 'missing').ready()


def test_training_real_frames_read_ordered_jpegs_without_extraction(tmp_path):
    from io import BytesIO
    import zipfile
    archive = tmp_path / 'real.zip'
    members = ['subject/video/clip/001', 'subject/video/clip/002', 'subject/video/clip/003']
    with zipfile.ZipFile(archive, 'w') as zipped:
        for member, value in zip(members, [30, 100, 200]):
            output = BytesIO()
            frame(value).save(output, format='JPEG')
            zipped.writestr(member, output.getvalue())
    frames = clip_frames({'kind': 'zip_jpeg_sequence', 'archive': str(archive), 'members': members})
    assert len(frames) == 3
    assert [round(np.asarray(image).mean()) for image in frames] == [30, 100, 200]


def test_external_claim_warning_keeps_diagnostics_separate(app):
    analyzer = app.extensions['analyzer']
    card = {'name': 'BERT test fixture', 'status': 'ready', 'version': 'test', 'evaluation': {'accuracy': .98},
        'external_evaluations': {'liar_strict_binary': {'accuracy': .5683, 'test_count': 549,
            'source_url': 'https://aclanthology.org/P17-2067/'}}}
    analyzer.text = lambda text: (.4, card)
    result = analyzer.analyze('text', 'A short claim.', None, 'en', [])
    assert result['fake_percentage'] == 40
    assert result['text_domain_warning']['accuracy'] == .5683
    assert result['text_domain_warning']['article_accuracy'] == .98
    assert '56.8%' in '\n'.join(result['limitations'])
    assert result['verification']['status'] == 'unverified'


def test_video_temporal_score_is_separate_from_frame_chart(app, tmp_path, monkeypatch):
    import cv2
    from types import SimpleNamespace
    path = tmp_path / 'ordered.mp4'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (96, 96))
    for value in range(20):
        writer.write(np.full((96, 96, 3), value*10, dtype=np.uint8))
    writer.release()
    # Solid-color frames stand in for an in-scope face clip in this score test.
    monkeypatch.setattr('backend.analysis.detected_face_counts', lambda images: [1] * len(images))
    analyzer = app.extensions['analyzer']
    analyzer.visual = SimpleNamespace(ready=lambda: True, predict=lambda images: (
        [.25]*len(images), {'name': 'Image fixture', 'version': 'image-v1', 'scope': 'Frame-only score'}))
    calls = []
    def predict(images, probabilities, version):
        calls.append((len(images), probabilities, version))
        return .81, {'name': 'Temporal fixture', 'version': 'video-v1', 'purpose': 'temporal',
            'scope': 'Temporal fixture scope', 'evaluation': {'accuracy': .93, 'test_count': 240}}
    analyzer.temporal = SimpleNamespace(ready=lambda: True, predict=predict)
    analyzer.generation = SimpleNamespace(ready=lambda: True, predict=lambda images: (
        [.6] * len(images), {'name': 'AI-origin fixture', 'version': 'ai-v1',
                             'source_url': 'https://example.test/model'}))
    result = analyzer.analyze('video', '', path, 'en', [])
    assert calls == [(12, [.25]*12, 'image-v1')]
    assert result['fake_percentage'] == 60
    assert result['real_percentage'] == 40
    assert result['confidence'] == 60
    assert 'average sampled-frame' in result['confidence_note']
    assert result['temporal_assessment']['fake_percentage'] == 81
    assert result['temporal_assessment']['primary_score'] is False
    assert all(frame['visual_fake_percentage'] == 25 for frame in result['frames'])
    assert all(frame['ai_percentage'] == 60 for frame in result['frames'])
    assert result['scope'] == 'Temporal fixture scope'
    assert not any('no trained temporal' in text for text in result['limitations'])
