"""Adapt an origin CNN using images explicitly labelled with CLI arguments.

Personal images are training data; their fitted scores are not held-out accuracy.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time

import joblib
import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

from ml.train_diverse_origin import (evaluate_datasets, infer, load_manifest,
    positive_int, prepare_output, train_epoch, validate_records, validation_partition)


def personal_records(real_images=(), ai_images=()):
    """Assign labels only from the explicit flags, never from names or pixels."""
    rows = []
    for label, names in ((0, real_images), (1, ai_images)):
        for name in names:
            path = Path(name).resolve()
            if not path.is_file():
                raise ValueError('An explicitly labelled image does not exist.')
            try:
                with Image.open(path) as image:
                    image.load()
            except (UnidentifiedImageError, OSError):
                raise ValueError('An explicitly labelled file is not a readable image.') from None
            with path.open('rb') as handle:
                digest = hashlib.file_digest(handle, 'sha256').hexdigest()
            rows.append({'path': str(path), 'sha256': digest, 'label': label, 'split': 'train',
                'dataset': 'user_labelled_real' if label == 0 else 'user_labelled_ai',
                'source_group': digest, 'label_source': '--real-image' if label == 0 else '--ai-image'})
    if not rows:
        raise ValueError('Provide at least one --real-image or --ai-image explicitly.')
    return validate_records(rows, Path.cwd(), require_splits=False)


def merge_personal_records(rows, personal):
    """Retain held-out splits and reject conflicting or reused evaluation images."""
    return validate_records(list(rows) + list(personal), Path.cwd())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', required=True, help='Origin CNN directory with backbone, model card, and training manifest.')
    parser.add_argument('--output', required=True, help='New model output directory.')
    parser.add_argument('--manifest', help='Optional relocated parent training manifest.')
    parser.add_argument('--real-image', action='append', default=[], help='Explicitly confirmed real image; repeat for more images.')
    parser.add_argument('--ai-image', action='append', default=[], help='Explicitly confirmed AI-generated image; repeat for more images.')
    parser.add_argument('--personal-weight', type=positive_int, default=64, help='Training exposures per personal image per epoch.')
    parser.add_argument('--epochs', type=positive_int, default=3)
    parser.add_argument('--threads', type=positive_int, default=3)
    parser.add_argument('--batch-size', type=positive_int, default=64)
    args = parser.parse_args()
    base = Path(args.parent).resolve()
    if Path(args.output).resolve() == base:
        raise ValueError('Choose a different output directory from the parent checkpoint.')
    if not (base/'backbone.pt').is_file() or not (base/'model-card.json').is_file():
        raise ValueError('The explicit parent must contain backbone.pt and model-card.json.')
    parent_card = json.loads((base/'model-card.json').read_text(encoding='utf-8'))
    rows, sources = load_manifest(args.manifest or base/'training-manifest.json')
    personal = personal_records(args.real_image, args.ai_image)
    rows = merge_personal_records(rows, personal)
    train = [row for row in rows if row['split'] == 'train'] + personal*(args.personal_weight-1)
    validation = [row for row in rows if row['split'] == 'validation']
    labels, calibration = validation_partition(validation)
    output = prepare_output(args.output)
    import torch
    from torchvision.models import mobilenet_v3_small
    started = time.monotonic()
    torch.set_num_threads(args.threads)
    torch.manual_seed(57)
    np.random.seed(57)
    random.seed(57)
    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 2)
    model.load_state_dict(torch.load(base/'backbone.pt', map_location='cpu', weights_only=True))
    for parameter in model.features[:6].parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW([{'params': model.features[6:].parameters(), 'lr': 5e-5},
        {'params': model.classifier.parameters(), 'lr': 1e-4}], weight_decay=.01)
    history, best = [], -1
    for epoch in range(1, args.epochs+1):
        loss = train_epoch(model, optimizer, train, epoch, started, args.batch_size)
        logits = infer(model, validation, flip=True, batch_size=args.batch_size)
        score = float(balanced_accuracy_score(labels[~calibration], logits[~calibration] >= 0))
        history.append({'epoch': epoch, 'loss': loss, 'validation_balanced_accuracy': score})
        print(json.dumps(history[-1]), flush=True)
        if score > best:
            best = score
            torch.save(model.state_dict(), output/'backbone.pt')
    model.load_state_dict(torch.load(output/'backbone.pt', map_location='cpu', weights_only=True))
    logits = infer(model, validation, flip=True, batch_size=args.batch_size)
    with threadpool_limits(limits=args.threads):
        calibrator = LogisticRegression(C=1e6, class_weight='balanced', max_iter=1000).fit(logits[calibration, None], labels[calibration])
    slope, intercept = float(calibrator.coef_[0, 0]), float(calibrator.intercept_[0])
    sigmoid = lambda values: 1/(1+np.exp(-np.clip(values, -30, 30)))
    test = [row for row in rows if row['split'] == 'test']
    probabilities = sigmoid(slope*infer(model, test, flip=True, batch_size=args.batch_size)+intercept)
    test_labels = np.array([row['label'] for row in test])
    fitted = sigmoid(slope*infer(model, personal, flip=True, batch_size=args.batch_size)+intercept)
    card = {'name': 'Image-origin CNN with explicitly labelled adaptation',
        'version': output.name+'-v1', 'purpose': 'ai_generation', 'architecture': 'cnn_origin',
        'parent_model_version': parent_card['version'],
        'parent_backbone_sha256': hashlib.sha256((base/'backbone.pt').read_bytes()).hexdigest(),
        'training_sources': sources,
        'prediction_enabled': False, 'locally_validated': False, 'deployment_ready': False, 'status': 'experimental',
        'training_history': history, 'selected_validation_balanced_accuracy': best,
        'split_counts': dict(Counter(row['split']+'/'+row['dataset'] for row in rows)),
        'personal_training_images': len(personal), 'personal_exposures_per_epoch': len(personal)*args.personal_weight,
        'training_examples_per_epoch': len(train), 'evaluation_by_dataset': evaluate_datasets(test, probabilities),
        'validation_calibration': {'method': 'balanced Platt', 'examples': int(calibration.sum()), 'slope': slope, 'intercept': intercept},
        'training_photo_fit': {'correct': int(np.sum((fitted >= .5).astype(int) == np.array([row['label'] for row in personal]))),
            'count': len(personal), 'ai_scores': fitted.tolist(), 'labels': [row['label'] for row in personal],
            'scope': 'Training fit only; not held-out accuracy.'},
        'limitations': ['Explicitly labelled personal images are training data, not independent acceptance evidence.',
            'Weights require independent review before enabling prediction.',
            'Dataset calibration is not a general real-world probability.'],
        'runtime_seconds': round(time.monotonic()-started)}
    joblib.dump({'kind': 'cnn', 'tta': 'horizontal_flip_average',
        'calibration': {'type': 'platt', 'slope': slope, 'intercept': intercept}, 'card': card}, output/'model.joblib')
    (output/'model-card.json').write_text(json.dumps(card, indent=2), encoding='utf-8')
    (output/'training-manifest.json').write_text(json.dumps({'sources': sources, 'entries': rows}, separators=(',', ':')), encoding='utf-8')
    np.savez_compressed(output/'test-predictions.npz', probabilities=probabilities, labels=test_labels,
                        hashes=[row['sha256'] for row in test])
    print(json.dumps(card), flush=True)


if __name__ == '__main__':
    main()
