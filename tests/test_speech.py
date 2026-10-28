"""Audio edge cases and video transcript routing; no network is used."""
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from backend.analysis import Analyzer
from backend.speech import SpeechTranscriber, ffmpeg_executable


def test_missing_checkpoint_abstains_without_loading():
    speech = SpeechTranscriber(None)
    assert speech.ready() is False
    assert speech.transcribe('unused.mp4')['status'] == 'not_configured'
    assert speech.model is None


@pytest.mark.skipif(not ffmpeg_executable(), reason='FFmpeg is not installed')
def test_no_audio_and_silence_do_not_invoke_whisper(tmp_path):
    executable = ffmpeg_executable()
    silent = tmp_path / 'silent.wav'
    subprocess.run([executable, '-nostdin', '-y', '-loglevel', 'error', '-f', 'lavfi',
                    '-i', 'anullsrc=r=16000:cl=mono', '-t', '1', str(silent)], check=True)
    audio, status = SpeechTranscriber._audio(silent)
    assert audio is None and status == 'silence'
    import cv2
    path = tmp_path / 'no-audio.mp4'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 5, (64, 64))
    for _ in range(5):
        writer.write(np.full((64, 64, 3), 120, dtype=np.uint8))
    writer.release()
    audio, status = SpeechTranscriber._audio(path)
    assert audio is None and status == 'no_audio'


def test_audio_timeout_is_reported(monkeypatch):
    monkeypatch.setattr('backend.speech.ffmpeg_executable', lambda: 'ffmpeg')
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 45)
    monkeypatch.setattr('backend.speech.subprocess.run', timeout)
    audio, status = SpeechTranscriber._audio('clip.mp4')
    assert audio is None and status == 'audio_timeout'


def test_video_transcript_is_assessed_and_used_for_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr('backend.analysis.runtime_available', lambda *names: True)
    import cv2
    path = tmp_path / 'clip.mp4'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 5, (64, 64))
    for _ in range(5):
        writer.write(np.full((64, 64, 3), 120, dtype=np.uint8))
    writer.release()
    analyzer = Analyzer({'IMAGE_MODEL_PATH': str(tmp_path / 'missing.joblib'),
                         'TEXT_MODEL_PATH': str(tmp_path / 'text'), 'WHISPER_MODEL': None,
                         'EVIDENCE_PROVIDER': 'none'})
    transcript = 'The city council will meet on Monday to discuss the public transport budget.'
    analyzer.speech = SimpleNamespace(ready=lambda: True, transcribe=lambda _: {
        'name': 'Whisper tiny speech transcription', 'version': 'test', 'status': 'complete', 'text': transcript})
    assessed = []
    def assess(text):
        assessed.append(text)
        return .25, {'name': 'BERT test', 'status': 'ready'}
    analyzer.text = assess
    verified = []
    def verify(text, *args, **kwargs):
        verified.append(text)
        return {'status': 'unverified', 'evidence': []}
    monkeypatch.setattr('backend.analysis.verify_submission', verify)
    result = analyzer.analyze('video', 'Submitted caption.', path, 'en', [])
    assert result['transcript'] == transcript
    assert assessed[0] == transcript
    assert assessed[-1] == result['claim_text']
    assert transcript in verified[0] and 'Submitted caption.' in verified[0]
    assert result['speech_text_assessment']['fake_percentage'] == 25
    # Speech is a separate assessment; it must not fabricate a visual score.
    assert result['fake_percentage'] is None
    assert result['speech']['status'] == 'complete'
