"""Bounded, offline Whisper transcription for uploaded videos.

Transcripts are observations, never evidence that a recording or claim is genuine.
Model loading and inference share a lock so concurrent jobs reuse one CPU model.
"""
from pathlib import Path
import shutil
import subprocess
import threading

import numpy as np


SAMPLE_RATE = 16_000
MAX_SECONDS = 120


def ffmpeg_executable():
    executable = shutil.which('ffmpeg')
    if executable:
        return executable
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        return None


class SpeechTranscriber:
    def __init__(self, model_path):
        self.directory = Path(model_path) if model_path else None
        self.lock = threading.Lock()
        self.processor = None
        self.model = None

    def ready(self):
        directory = self.directory
        return bool(directory and directory.is_dir()
                    and all((directory / name).is_file() for name in (
                        'config.json', 'preprocessor_config.json', 'tokenizer_config.json'))
                    and any((directory / name).is_file() for name in (
                        'model.safetensors', 'pytorch_model.bin'))
                    and ffmpeg_executable())

    @staticmethod
    def _audio(filepath):
        executable = ffmpeg_executable()
        if not executable:
            return None, 'unavailable'
        # Local input only; extraction is bounded in time, channels and sample rate.
        command = [executable, '-nostdin', '-hide_banner', '-loglevel', 'error',
                   '-protocol_whitelist', 'file,pipe', '-i', str(Path(filepath).resolve()),
                   '-t', str(MAX_SECONDS), '-map', '0:a:0?', '-vn', '-sn', '-dn',
                   '-ac', '1', '-ar', str(SAMPLE_RATE), '-f', 's16le', 'pipe:1']
        try:
            completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       timeout=45, check=False, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except subprocess.TimeoutExpired:
            return None, 'audio_timeout'
        if completed.returncode:
            error = completed.stderr.decode('utf8', errors='replace').lower()
            if 'does not contain any stream' in error or 'matches no streams' in error:
                return None, 'no_audio'
            return None, 'audio_unreadable'
        # Never hand unexpectedly oversized decoder output to a model.
        if len(completed.stdout) > (MAX_SECONDS * SAMPLE_RATE + SAMPLE_RATE) * 2:
            return None, 'audio_unreadable'
        audio = np.frombuffer(completed.stdout[:MAX_SECONDS * SAMPLE_RATE * 2], dtype='<i2').astype(np.float32) / 32768
        if audio.size < SAMPLE_RATE // 4:
            return None, 'no_audio'
        if not SpeechTranscriber._has_signal(audio):
            return None, 'silence'
        return audio, 'ready'

    @staticmethod
    def _has_signal(audio):
        # A conservative silence gate, not a claim to distinguish speech from music.
        return bool(audio.size and np.sqrt(np.mean(np.square(audio))) >= .001
                    and np.max(np.abs(audio)) >= .005)

    def _load(self):
        if self.model is None:
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
            self.processor = WhisperProcessor.from_pretrained(str(self.directory), local_files_only=True)
            self.model = WhisperForConditionalGeneration.from_pretrained(
                str(self.directory), local_files_only=True, use_safetensors=True).eval()

    def transcribe(self, filepath):
        info = {'name': 'Whisper tiny speech transcription', 'version': 'openai/whisper-tiny',
                'status': 'not_configured', 'text': ''}
        if not self.ready():
            return info
        audio, status = self._audio(filepath)
        if audio is None:
            return dict(info, status=status)
        import torch
        chunks = []
        skipped = 0
        with self.lock, torch.inference_mode():
            self._load()
            # Fixed <=30-second windows keep CPU generation and feature memory bounded.
            for start in range(0, audio.size, SAMPLE_RATE * 30):
                segment = audio[start:start + SAMPLE_RATE * 30]
                if not self._has_signal(segment):
                    skipped += 1
                    continue
                inputs = self.processor(segment, sampling_rate=SAMPLE_RATE,
                                        return_tensors='pt', return_attention_mask=True)
                features = inputs.input_features
                decoder_start = torch.tensor([[self.model.config.decoder_start_token_id]])
                first_logits = self.model(input_features=features, decoder_input_ids=decoder_start).logits[0, -1]
                no_speech_id = self.processor.tokenizer.convert_tokens_to_ids('<|nospeech|>')
                no_speech = float(torch.softmax(first_logits.float(), dim=-1)[no_speech_id])
                if no_speech >= .6:
                    skipped += 1
                    continue
                tokens = self.model.generate(
                    input_features=features, attention_mask=inputs.attention_mask,
                    task='transcribe', max_new_tokens=224, do_sample=False,
                    num_beams=1, return_timestamps=False)
                text = self.processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()
                if text:
                    chunks.append({'seconds': round(start / SAMPLE_RATE, 2), 'text': text})
        transcript = ' '.join(chunk['text'] for chunk in chunks).strip()
        return dict(info, status='complete' if transcript else 'no_speech_detected', text=transcript,
                    chunks=chunks, skipped_chunks=skipped, duration_seconds=round(audio.size / SAMPLE_RATE, 2))
