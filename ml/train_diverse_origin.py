"""Train and calibrate a MobileNet origin classifier from an explicit manifest.

The manifest has an ``entries`` array. Each entry provides label (0 real, 1 AI),
split, sha256, dataset, source_group, and either path or archive plus member.
Paths are relative to the manifest. Training runs offline, never at app startup.
"""
import argparse
from collections import Counter
import hashlib
from io import BytesIO
import json
from pathlib import Path
import random
import re
import time
import zipfile

import joblib
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

from ml.train_image import metrics


def printed_view(image, seed):
    """Apply the same camera/print augmentation to either origin class."""
    rng = random.Random(seed)
    image = ImageOps.exif_transpose(image).convert('RGB')
    image.thumbnail((320, 320))
    photo = ImageOps.expand(image, border=8, fill='white')
    canvas = Image.new('RGB', (round(photo.width * 1.35), round(photo.height * 1.4)),
                       tuple(rng.randint(130, 235) for _ in range(3)))
    canvas.paste(photo, ((canvas.width-photo.width)//2, (canvas.height-photo.height)//2))
    return canvas.rotate(rng.uniform(-3, 3), resample=Image.Resampling.BICUBIC).filter(ImageFilter.GaussianBlur(.45))


def validate_records(entries, base_dir, *, require_splits=True):
    """Verify bytes and labels; reject content or source groups across splits."""
    if not isinstance(entries, list) or not entries:
        raise ValueError('The manifest must contain a nonempty entries array.')
    base_dir = Path(base_dir).resolve()
    unique, groups = {}, {}
    archives = {}
    try:
        for number, original in enumerate(entries, 1):
            if not isinstance(original, dict):
                raise ValueError(f'Entry {number} must be an object.')
            row = dict(original)
            if type(row.get('label')) is not int or row['label'] not in (0, 1):
                raise ValueError(f'Entry {number}: label must be integer 0 (real) or 1 (AI-generated).')
            if row.get('split') not in ('train', 'validation', 'test'):
                raise ValueError(f'Entry {number}: split must be train, validation, or test.')
            for field in ('dataset', 'source_group'):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise ValueError(f'Entry {number}: {field} is required.')
            if not isinstance(row.get('sha256'), str) or not re.fullmatch(r'[0-9a-fA-F]{64}', row['sha256']):
                raise ValueError(f'Entry {number}: sha256 must contain 64 hexadecimal characters.')
            row['sha256'] = row['sha256'].lower()
            has_path = isinstance(row.get('path'), str) and bool(row['path'])
            has_archive = isinstance(row.get('archive'), str) and bool(row['archive'])
            if has_path == has_archive or (has_path and row.get('member')):
                raise ValueError(f'Entry {number}: provide either path or archive and member.')
            field = 'path' if has_path else 'archive'
            path = (base_dir / row[field]).resolve()
            if not path.is_file():
                raise ValueError(f'Entry {number}: its image or archive file does not exist.')
            row[field] = str(path)
            if has_archive:
                if not isinstance(row.get('member'), str) or not row['member'] or row['member'].endswith('/'):
                    raise ValueError(f'Entry {number}: an archive member image is required.')
                if path not in archives:
                    archives[path] = zipfile.ZipFile(path)
                try:
                    with archives[path].open(row['member']) as handle:
                        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
                except KeyError:
                    raise ValueError(f'Entry {number}: archive member does not exist.') from None
            else:
                with path.open('rb') as handle:
                    digest = hashlib.file_digest(handle, 'sha256').hexdigest()
            if digest != row['sha256']:
                raise ValueError(f'Entry {number}: image bytes differ from sha256 in the manifest.')
            previous = unique.get(digest)
            if previous and previous['label'] != row['label']:
                raise ValueError('The same image has conflicting origin labels.')
            if previous and previous['split'] != row['split']:
                raise ValueError('An image overlaps training, validation, or test splits.')
            group = (row['dataset'], row['source_group'])
            if group in groups and groups[group] != row['split']:
                raise ValueError('A source group overlaps training, validation, or test splits.')
            groups[group] = row['split']
            unique.setdefault(digest, row)
    finally:
        for archive in archives.values():
            archive.close()
    rows = list(unique.values())
    if require_splits:
        for split in ('train', 'validation', 'test'):
            if {row['label'] for row in rows if row['split'] == split} != {0, 1}:
                raise ValueError(f'The {split} split needs independent examples of both origin classes.')
    return rows


def load_manifest(path, *, require_splits=True):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError('The manifest must be an object containing entries.')
    rows = validate_records(data.get('entries'), path.parent, require_splits=require_splits)
    return rows, {'manifest_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                  'datasets': sorted({row['dataset'] for row in rows}),
                  'input_entries': len(data['entries']), 'deduplicated_entries': len(rows)}


def validation_partition(rows):
    """Keep each validation source group in either tuning or calibration."""
    labels = np.array([row['label'] for row in rows])
    calibration = np.array([int(hashlib.sha256((row['dataset']+'/'+row['source_group']).encode()).hexdigest()[:8], 16) % 2 == 0
                            for row in rows], dtype=bool)
    if len(set(labels[calibration])) != 2 or len(set(labels[~calibration])) != 2:
        raise ValueError('Validation tuning and calibration groups each need both classes; add independent validation groups.')
    return labels, calibration


class OriginImages:
    def __init__(self, rows, augment=False):
        from torchvision import transforms
        self.rows, self.augment, self.archives = rows, augment, {}
        self.transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        if row.get('archive'):
            path = row['archive']
            if path not in self.archives:
                self.archives[path] = zipfile.ZipFile(path)
            handle = BytesIO(self.archives[path].read(row['member']))
        else:
            handle = row['path']
        with Image.open(handle) as original:
            image = ImageOps.exif_transpose(original).convert('RGB')
        if self.augment:
            if random.random() < .5:
                image = ImageOps.mirror(image)
            if random.random() < .2:
                image = printed_view(image, random.randrange(2**31))
            if random.random() < .2:
                buffer = BytesIO()
                image.save(buffer, format='JPEG', quality=random.randint(55, 95))
                buffer.seek(0)
                with Image.open(buffer) as decoded:
                    image = decoded.convert('RGB')
        return self.transform(image), row['label']

    def close(self):
        for archive in self.archives.values():
            archive.close()


def infer(model, rows, flip=False, batch_size=64):
    import torch
    from torch.utils.data import DataLoader
    dataset, output = OriginImages(rows), []
    model.eval()
    try:
        with torch.inference_mode():
            for images, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False):
                logits = model(images)
                score = logits[:, 1] - logits[:, 0]
                if flip:
                    other = model(torch.flip(images, dims=[3]))
                    score = (score + other[:, 1] - other[:, 0]) / 2
                output.extend(score.tolist())
    finally:
        dataset.close()
    return np.array(output)


def train_epoch(model, optimizer, selected, epoch, started, batch_size=64):
    import torch
    from torch.utils.data import DataLoader
    counts = Counter(row['label'] for row in selected)
    if set(counts) != {0, 1}:
        raise ValueError('Training requires examples from both origin classes.')
    dataset = OriginImages(selected, augment=True)
    weights = torch.tensor([len(selected)/(2*counts[label]) for label in (0, 1)], dtype=torch.float32)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    model.train()
    model.features[:6].eval()
    total_loss = total = 0
    try:
        for step, (images, labels) in enumerate(DataLoader(dataset, batch_size=batch_size, shuffle=True)):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()*len(labels)
            total += len(labels)
            if step % 100 == 0:
                print(json.dumps({'epoch': epoch, 'seen': total, 'total': len(selected),
                    'loss': round(total_loss/total, 4), 'seconds': round(time.monotonic()-started)}), flush=True)
    finally:
        dataset.close()
    return total_loss / total


def evaluate_datasets(rows, probabilities):
    labels = np.array([row['label'] for row in rows])
    evaluations = {}
    for name in sorted({row['dataset'] for row in rows}):
        mask = np.array([row['dataset'] == name for row in rows])
        if len(set(labels[mask])) == 2:
            evaluations[name] = metrics(labels[mask], probabilities[mask])
            evaluations[name]['labels'] = ['real', 'AI-generated']
        else:
            label = int(labels[mask][0])
            evaluations[name] = {'count': int(mask.sum()),
                'real_recall' if label == 0 else 'ai_recall': float(np.mean((probabilities[mask] >= .5) == label))}
    return evaluations


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return value


def prepare_output(path):
    output = Path(path).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Choose a new or empty output directory; existing checkpoints are never overwritten.')
    output.mkdir(parents=True, exist_ok=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, help='Normalized labelled image manifest JSON.')
    parser.add_argument('--backbone', required=True, help='Trusted local MobileNetV3 Small state dictionary.')
    parser.add_argument('--output', required=True, help='New model output directory.')
    parser.add_argument('--epochs', type=positive_int, default=3)
    parser.add_argument('--threads', type=positive_int, default=3)
    parser.add_argument('--batch-size', type=positive_int, default=64)
    args = parser.parse_args()
    rows, sources = load_manifest(args.manifest)
    train = [row for row in rows if row['split'] == 'train']
    validation = [row for row in rows if row['split'] == 'validation']
    labels, calibration = validation_partition(validation)
    initial = Path(args.backbone).resolve()
    if not initial.is_file():
        raise ValueError('The explicit backbone file does not exist.')
    output = prepare_output(args.output)
    import torch
    from torchvision.models import mobilenet_v3_small
    started = time.monotonic()
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    model = mobilenet_v3_small(weights=None)
    state = torch.load(initial, map_location='cpu', weights_only=True)
    transferred = model.load_state_dict({key: value for key, value in state.items() if not key.startswith('classifier.')}, strict=False)
    if transferred.unexpected_keys or any(not key.startswith('classifier.') for key in transferred.missing_keys):
        raise ValueError('The backbone must contain MobileNetV3 Small feature weights.')
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 2)
    for parameter in model.features[:6].parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW([{'params': model.features[6:].parameters(), 'lr': 2e-4},
        {'params': model.classifier.parameters(), 'lr': 7e-4}], weight_decay=.01)
    history, best = [], -1
    for epoch in range(1, args.epochs+1):
        loss = train_epoch(model, optimizer, train, epoch, started, args.batch_size)
        logits = infer(model, validation, batch_size=args.batch_size)
        score = float(balanced_accuracy_score(labels[~calibration], logits[~calibration] >= 0))
        history.append({'epoch': epoch, 'validation_balanced_accuracy': score, 'loss': loss})
        print(json.dumps(history[-1]), flush=True)
        if score > best:
            best = score
            torch.save(model.state_dict(), output/'backbone.pt')
    model.load_state_dict(torch.load(output/'backbone.pt', map_location='cpu', weights_only=True))
    logits = infer(model, validation, flip=True, batch_size=args.batch_size)
    with threadpool_limits(limits=args.threads):
        calibrator = LogisticRegression(C=1e6, class_weight='balanced', max_iter=1000).fit(logits[calibration, None], labels[calibration])
    slope, intercept = float(calibrator.coef_[0, 0]), float(calibrator.intercept_[0])
    test = [row for row in rows if row['split'] == 'test']
    test_y = np.array([row['label'] for row in test])
    probabilities = 1/(1+np.exp(-np.clip(slope*infer(model, test, flip=True, batch_size=args.batch_size)+intercept, -30, 30)))
    card = {'name': 'Fine-tuned image-origin CNN', 'version': output.name+'-v1',
        'purpose': 'ai_generation', 'architecture': 'cnn_origin',
        'scope': 'Dataset-specific real/AI origin classification, separate from manipulation and factual truth.',
        'training_sources': sources, 'initial_backbone_sha256': hashlib.sha256(initial.read_bytes()).hexdigest(),
        'split_counts': dict(Counter(row['split']+'/'+row['dataset'] for row in rows)),
        'training_history': history, 'selected_validation_balanced_accuracy': best,
        'evaluation_by_dataset': evaluate_datasets(test, probabilities),
        'prediction_enabled': False, 'locally_validated': False, 'deployment_ready': False, 'status': 'experimental',
        'validation_calibration': {'method': 'balanced Platt', 'examples': int(calibration.sum()),
            'slope': slope, 'intercept': intercept, 'scope': 'Dataset-specific; not general real-world probability.'},
        'limitations': ['Training labels and source groups are supplied explicitly by the dataset operator.',
            'Weights require independent review before enabling prediction.',
            'Dataset calibration is not a general real-world probability.'],
        'runtime_seconds': round(time.monotonic()-started)}
    joblib.dump({'kind': 'cnn', 'tta': 'horizontal_flip_average',
        'calibration': {'type': 'platt', 'slope': slope, 'intercept': intercept}, 'card': card}, output/'model.joblib')
    (output/'model-card.json').write_text(json.dumps(card, indent=2), encoding='utf-8')
    (output/'training-manifest.json').write_text(json.dumps({'sources': sources, 'entries': rows}, separators=(',', ':')), encoding='utf-8')
    np.savez_compressed(output/'test-predictions.npz', probabilities=probabilities, labels=test_y,
                        hashes=[row['sha256'] for row in test])
    print(json.dumps(card), flush=True)


if __name__ == '__main__':
    main()
