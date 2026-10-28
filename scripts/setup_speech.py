"""Download the official compact Whisper checkpoint; no API key is required."""
import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='models/whisper-tiny')
    parser.add_argument('--configure', action='store_true', help='Set WHISPER_MODEL in backend/.env after validation.')
    args = parser.parse_args()
    repository = 'openai/whisper-tiny'
    revision = HfApi().model_info(repository).sha
    output = Path(args.output).resolve()
    snapshot_download(repo_id=repository, revision=revision, local_dir=str(output),
        allow_patterns=['*.json', 'merges.txt', 'model.safetensors', 'README.md', 'LICENSE'])
    from backend.speech import SpeechTranscriber
    speech = SpeechTranscriber(output)
    if not speech.ready():
        raise RuntimeError('Whisper files or FFmpeg are incomplete; install imageio-ffmpeg and retry.')
    with speech.lock:
        speech._load()
    (output / 'installation.json').write_text(json.dumps({
        'repository': repository, 'revision': revision,
        'purpose': 'Local speech transcription only; not a factuality, deepfake or AI-generation detector.'}, indent=2), encoding='utf8')
    if args.configure:
        from dotenv import set_key
        root = Path(__file__).resolve().parents[1]
        env_path = root / 'backend' / '.env'
        configured_path = output.relative_to(root) if output.is_relative_to(root) else output
        set_key(str(env_path), 'WHISPER_MODEL', configured_path.as_posix(), quote_mode='auto')
    print(json.dumps({'model': repository, 'revision': revision, 'ready': True, 'configured': args.configure}))


if __name__ == '__main__':
    main()
