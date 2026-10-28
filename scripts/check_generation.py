"""Smoke-test the installed origin model on explicit local image inputs."""
import argparse
import json
from pathlib import Path
import time

from PIL import Image

from backend.generation import GenerationDetector
from scripts.check_setup import read_environment


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('images', nargs='+', type=Path, help='Local images to use for inference; no accuracy is measured.')
    parser.add_argument('--model', type=Path, help='Override AI_IMAGE_MODEL_PATH from the environment.')
    args = parser.parse_args()
    config = read_environment()
    model_path = args.model or Path(config.get('AI_IMAGE_MODEL_PATH') or 'models/ai-image')
    detector = GenerationDetector(root / model_path)
    if not detector.ready():
        parser.error('Install a trusted origin model bundle and configure AI_IMAGE_MODEL_PATH first.')
    images = []
    for path in args.images:
        if not path.is_file():
            parser.error('Every image argument must identify an existing local file.')
        with Image.open(path) as image:
            images.append(image.convert('RGB'))
    started = time.monotonic()
    import torch
    torch.set_num_threads(3)
    scores, card = detector.predict(images)
    assert len(scores) == len(images) and all(0 <= score <= 1 for score in scores)
    # Warm inference exercises the cached model without downloading or rebuilding it.
    repeat, _ = detector.predict(images[:1])
    assert abs(scores[0] - repeat[0]) < 1e-5
    print(json.dumps({'ready': True, 'images': len(images), 'ai_model_scores': scores,
                      'seconds': round(time.monotonic() - started, 2), 'revision': card['version'],
                      'accuracy_measured': False, 'status': 'experimental'}))


if __name__ == '__main__':
    main()
