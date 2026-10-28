import json
from types import SimpleNamespace

from backend import video_model


def test_video_status_explains_rejected_candidates_without_activating_them(tmp_path, monkeypatch):
    monkeypatch.setattr(video_model, 'ROOT', tmp_path)
    directory = tmp_path / 'models' / 'video-test'
    directory.mkdir(parents=True)
    card = {'name': 'Research video candidate', 'version': 'test', 'purpose': 'video_synthetic_or_manipulated',
            'prediction_enabled': False, 'deployment_ready': False, 'locally_validated': False,
            'activation_decision': 'not_activated_no_heldout_improvement',
            'evaluation_by_dataset': {'New videos': {'accuracy': .5, 'test_count': 16}},
            'personal_training_fit': [{'filename': 'private-example.mp4'}]}
    path = directory / 'model-card.json'
    path.write_text(json.dumps(card))
    original = path.read_bytes()
    status = video_model.video_detection_status({'VIDEO_MODEL_PATH': None}, SimpleNamespace(ready=lambda: False))
    assert status['status'] == 'validation_required'
    assert not status['configured'] and not status['inference_enabled']
    assert status['candidates'][0]['dataset_diagnostics']['New videos']['accuracy'] == .5
    assert 'private-example' not in str(status)
    assert path.read_bytes() == original
