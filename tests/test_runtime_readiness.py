import json
from types import SimpleNamespace

from backend import analysis


def analyzer_with_artifacts(tmp_path):
    for name in ('config.json', 'model-card.json', 'model.safetensors'):
        (tmp_path / name).write_text('{}', encoding='utf8')
    analyzer = analysis.Analyzer.__new__(analysis.Analyzer)
    analyzer.config = {'TEXT_MODEL_PATH': str(tmp_path), 'EVIDENCE_PROVIDER': 'public'}
    for name in ('visual', 'temporal', 'video', 'speech', 'generation', 'evidence_verifier'):
        setattr(analyzer, name, SimpleNamespace(ready=lambda: True))
    analyzer.ai_text = SimpleNamespace(ready=True)
    return analyzer


def test_artifacts_do_not_claim_neural_readiness_when_runtime_missing(tmp_path, monkeypatch):
    analyzer = analyzer_with_artifacts(tmp_path)
    monkeypatch.setattr(analysis, 'runtime_available', lambda *names: False)
    ready = analyzer.readiness()
    for name in ('image', 'video', 'text', 'speech', 'temporal', 'video_synthetic_detection',
                 'ai_generation', 'local_evidence_comparison'):
        assert ready[name] is False
    assert ready['ai_writing'] is True
    assert ready['evidence_search'] is True


def test_cnn_origin_only_requires_vision_runtime(tmp_path, monkeypatch):
    analyzer = analyzer_with_artifacts(tmp_path)
    (tmp_path / 'model-card.json').write_text(json.dumps({'architecture': 'cnn_origin'}), encoding='utf8')
    analyzer.generation.directory = tmp_path
    monkeypatch.setattr(analysis, 'runtime_available', lambda *names: 'transformers' not in names)
    ready = analyzer.readiness()
    assert ready['image'] is True
    assert ready['ai_generation'] is True
    assert ready['text'] is False


def test_dependencies_alone_do_not_claim_missing_text_model_ready(tmp_path, monkeypatch):
    analyzer = analyzer_with_artifacts(tmp_path)
    (tmp_path / 'model.safetensors').unlink()
    monkeypatch.setattr(analysis, 'runtime_available', lambda *names: True)
    assert analyzer.readiness()['text'] is False
