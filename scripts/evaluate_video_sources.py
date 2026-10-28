"""Evaluate a frozen video detector; never fit or select on external benchmarks."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve

from ml.temporal import TemporalModel
from ml.train_image import metrics
from ml.train_temporal import clip_frames


def score_report(rows):
    labels = np.array([int(row['label'] == 'fake') for row in rows])
    scores = np.array([row['score'] for row in rows])
    if len(set(labels)) < 2:
        return {'count': len(rows), 'status': 'both_classes_required'}
    result = metrics(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    curve = interp1d(fpr, tpr)
    result['eer'] = float(brentq(lambda value: 1 - value - curve(value), 0, 1))
    result['threshold_policy'] = 'Existing fixed 0.5 threshold; no threshold fitted on this dataset.'
    return result


def main(args):
    import torch
    torch.set_num_threads(3)
    destination = Path(args.output)
    if destination.exists():
        raise ValueError('Choose a new evaluation output to preserve earlier results.')
    manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    entries = manifest.get('evaluation_entries', [])
    if not entries:
        raise ValueError('No external evaluation media were imported.')
    if any(row.get('split') != 'external_evaluation' for row in entries):
        raise ValueError('Expected explicitly reserved external evaluation entries.')
    model_path = Path(args.model)
    model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    model = TemporalModel(model_path)
    results = []
    for index, entry in enumerate(entries):
        frames = clip_frames(entry)
        score, card = model.predict(frames)
        results.append({key: entry.get(key) for key in ('id', 'dataset', 'label', 'pair_id', 'generator', 'sha256')})
        results[-1]['score'] = score
        if (index + 1) % 20 == 0:
            print(f'Evaluated {index+1}/{len(entries)} clips', flush=True)
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != model_sha:
        raise ValueError('Model changed during evaluation; refuse mixed-version results.')
    by_dataset = defaultdict(list)
    for row in results:
        by_dataset[row['dataset']].append(row)
    per_generator = {}
    df26 = by_dataset.get('DF26', [])
    for generator in sorted({r['generator'] for r in df26 if r['label'] == 'fake'}):
        fake = [r for r in df26 if r['generator'] == generator and r['label'] == 'fake']
        pairs = {r['pair_id'] for r in fake}
        real = [r for r in df26 if r['label'] == 'real' and r['pair_id'] in pairs]
        per_generator[generator] = score_report(real + fake)
    report = {'model_sha256': model_sha, 'model_version': card['version'],
              'manifest_sha256': hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
              'model_updated': False, 'calibration_fitted': False, 'threshold_fitted': False,
              'usage': 'Held-out reporting only; do not use DF26 results to tune or select a model.',
              'per_dataset': {name: score_report(rows) for name, rows in by_dataset.items()},
              'df26_per_generator': per_generator, 'predictions': results}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'predictions'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--model', default='models/video-temporal-retrained-20261020/model.joblib')
    parser.add_argument('--output', required=True)
    main(parser.parse_args())
