import json
from types import SimpleNamespace

from PIL import Image
import requests

from backend.media_inspection import inspect_media


def response(payload):
    return SimpleNamespace(status_code=200, raise_for_status=lambda: None,
        json=lambda: {'candidates': [{'content': {'parts': [{'text': json.dumps(payload)}]}}]})


def test_inspection_returns_scene_features_and_retries_timeout(monkeypatch):
    payload = {'description': 'Four people sit at a table.', 'visible_text': 'SHOT ON PHONE',
               'visual_type': 'photograph', 'visual_artifacts': '', 'text_role': 'watermark_or_label',
               'visible_features': ['A wooden table.', 'Four seated people.', 'A wooden table.', None],
               'observed_actions': ['People face each other.']}
    calls = []
    def post(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise requests.Timeout()
        return response(payload)
    monkeypatch.setattr('backend.media_inspection.requests.post', post)
    result = inspect_media([Image.new('RGB', (32, 32))], 'test-key')
    assert len(calls) == 2
    assert result['status'] == 'completed'
    assert result['visible_features'] == ['A wooden table.', 'Four seated people.']
    assert result['observed_actions'] == ['People face each other.']
    assert result['text_role'] == 'watermark_or_label'


def test_unavailable_vision_does_not_invent_scene_details(monkeypatch):
    calls = []
    def post(*args, **kwargs):
        calls.append(1)
        raise requests.Timeout()
    monkeypatch.setattr('backend.media_inspection.requests.post', post)
    result = inspect_media([Image.new('RGB', (32, 32))], 'test-key')
    assert len(calls) == 2
    assert result['status'] == 'unavailable'
    assert result['description'] == ''
