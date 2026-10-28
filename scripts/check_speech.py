"""Transcribe a local audio/video file through the same offline video speech module."""
import argparse
import json
from pathlib import Path

from backend.speech import SpeechTranscriber
from scripts.check_setup import read_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file', type=Path)
    args = parser.parse_args()
    if not args.file.is_file():
        parser.error('Supply an existing local audio/video file.')
    config = read_environment()
    import torch
    torch.set_num_threads(min(4, torch.get_num_threads()))
    transcriber = SpeechTranscriber(config.get('WHISPER_MODEL'))
    result = transcriber.transcribe(args.file)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['status'] in ('complete', 'no_audio', 'silence', 'no_speech_detected') else 1


if __name__ == '__main__':
    raise SystemExit(main())
