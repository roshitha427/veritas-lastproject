"""Train a source/identity-group-disjoint video model on audited paired media."""
import argparse
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile

import cv2
import joblib
import numpy as np
from PIL import Image
from scipy.optimize import minimize_scalar
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

from ml.temporal import ImageNetFrameEncoder, FEATURE_VERSION
from ml.train_image import metrics


def source_balanced_weights(entries):
    """Give each corpus/class cell equal influence, regardless of its size."""
    from collections import Counter
    cells = Counter((row.get('dataset', 'base'), row['label']) for row in entries)
    return np.array([len(entries) / (len(cells) * cells[(row.get('dataset', 'base'), row['label'])])
                     for row in entries])


def selection_score(labels, predictions, entries, macro_datasets=False):
    if not macro_datasets:
        return float(balanced_accuracy_score(labels, predictions))
    datasets = np.array([row.get('dataset', 'base') for row in entries])
    return float(np.mean([balanced_accuracy_score(labels[datasets == name], predictions[datasets == name])
                          for name in sorted(set(datasets))]))


def clip_frames(entry, count=12):
    if entry['kind'] == 'zip_video':
        import shutil
        import tempfile
        from ml.prepare_video_sources import safe_name, MEDIA
        member = entry['member']
        if not safe_name(member) or Path(member).suffix.lower() not in MEDIA:
            raise ValueError('Unsafe archive video reference.')
        scratch_root = Path(__file__).resolve().parents[1] / 'artifacts'
        scratch_root.mkdir(exist_ok=True)
        with zipfile.ZipFile(entry['archive']) as archive:
            size = archive.getinfo(member).file_size
            if not 0 < size <= 512 * 1024 * 1024:
                raise ValueError('Archive video exceeds the supported size.')
            with tempfile.TemporaryDirectory(prefix='video-decode-', dir=scratch_root) as scratch:
                target = Path(scratch) / ('clip' + Path(member).suffix.lower())
                with archive.open(member) as source, target.open('wb') as destination:
                    shutil.copyfileobj(source, destination, length=1024 * 1024)
                return clip_frames({'kind': 'video', 'filepath': str(target), 'id': entry['id']}, count=count)
    if entry['kind'] == 'zip_jpeg_sequence':
        members = entry['members']
        indices = np.rint(np.linspace(0, len(members)-1, min(count, len(members)))).astype(int)
        with zipfile.ZipFile(entry['archive']) as archive:
            frames = []
            for index in indices:
                with Image.open(BytesIO(archive.read(members[int(index)]))) as image:
                    frames.append(image.convert('RGB'))
        return frames
    cap = cv2.VideoCapture(entry['filepath'])
    frames = []
    try:
        length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not cap.isOpened() or not np.isfinite(length) or length < 3:
            raise ValueError('Cannot decode training clip: ' + entry['id'])
        for index in np.rint(np.linspace(0, length-1, min(count, int(length)))).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, bgr = cap.read()
            if ok:
                frames.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    finally:
        cap.release()
    if len(frames) < 3:
        raise ValueError('Too few readable frames: ' + entry['id'])
    return frames


def main(args):
    started = time.monotonic()
    manifest_path, output = Path(args.manifest), Path(args.output)
    if (output / 'model.joblib').exists() or (output / 'model-card.json').exists():
        raise ValueError('Choose a new candidate output directory; existing model artifacts are preserved.')
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    if manifest.get('download_status') not in (None, 'complete') and not args.cache_available:
        raise ValueError('Finish and verify all dataset downloads before fitting or evaluation.')
    entries = manifest['entries']
    if any(entry.get('training_prohibited') or entry.get('dataset') == 'DF26'
           or entry.get('role') == 'evaluation' for entry in entries):
        raise ValueError('Evaluation-only records cannot enter model selection, calibration, or training.')
    if set(entry['label'] for entry in entries) != {'real', 'fake'}:
        raise ValueError('Temporal labels must explicitly be real/fake.')
    groups = {split: {entry['pair_id'] for entry in entries if entry['split'] == split}
              for split in ('train', 'validation', 'test')}
    if any(groups[a] & groups[b] for a, b in [('train','validation'), ('train','test'), ('validation','test')]):
        raise ValueError('Identity/source swap groups overlap between splits.')
    if min(map(len, groups.values())) < 2:
        raise ValueError('Need at least two independent source/identity groups in every split.')
    import torch
    torch.set_num_threads(args.threads)
    output.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache_dir) if args.cache_dir else output / 'cache'
    cache.mkdir(exist_ok=True)
    if not Path(args.backbone).is_file():
        raise ValueError('Install the ImageNet-only frame backbone first.')
    encoder = ImageNetFrameEncoder(args.backbone)
    base_sha = hashlib.sha256(Path(args.backbone).read_bytes()).hexdigest()
    rows = {}
    # Reserve test feature extraction until selection and calibration are finished.
    def features(split):
        selected = [entry for entry in entries if entry['split'] == split]
        x, y = [], []
        for i, entry in enumerate(selected):
            identity = hashlib.sha256((FEATURE_VERSION+base_sha+json.dumps(entry, sort_keys=True)).encode()).hexdigest()
            if entry.get('feature_backbone_sha256') == base_sha:
                import re
                cached_key = entry.get('feature_cache_key', '')
                if re.fullmatch(r'[0-9a-f]{64}', cached_key):
                    identity = cached_key
            path = cache / (identity + '.npy')
            if path.is_file():
                vector = np.load(path)
            else:
                frames = clip_frames(entry)
                vector = encoder.features(frames)
                temporary = path.with_suffix('.tmp')
                with temporary.open('wb') as handle:
                    np.save(handle, vector)
                temporary.replace(path)
            x.append(vector)
            y.append(int(entry['label'] == 'fake'))
            if (i+1) % 40 == 0:
                print(f'{split}: {i+1}/{len(selected)} clips, {time.monotonic()-started:.0f}s', flush=True)
        rows[split] = selected
        return np.stack(x), np.asarray(y)
    if args.cache_available:
        # Frozen features only; no fitting and no test-media access while the
        # remaining source downloads are in progress.
        features('train')
        features('validation')
        print('Available training/validation features cached. No model fitted or test scored.', flush=True)
        return
    x, y = features('train')
    vx, vy = features('validation')
    validation_groups = sorted(groups['validation'])
    if all(entry.get('validation_role') in ('selection', 'calibration') for entry in rows['validation']):
        tune = np.array([entry['validation_role'] == 'selection' for entry in rows['validation']])
    else:
        tune = np.array([entry['pair_id'] in validation_groups[:len(validation_groups)//2] for entry in rows['validation']])
    cal = ~tune
    balance_sources = manifest.get('balance_training_datasets') is True
    macro_validation = manifest.get('validation_metric') == 'macro_dataset_balanced_accuracy'
    weights = source_balanced_weights(rows['train']) if balance_sources else compute_sample_weight('balanced', y)
    candidates = [(f'logistic-C{c}', make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=1200,
        class_weight=None if balance_sources else 'balanced', random_state=42))) for c in [.001, .01, .1, .3, 1, 3, 10]]
    candidates += [('histgradient', HistGradientBoostingClassifier(max_iter=140, max_leaf_nodes=7,
        learning_rate=.06, l2_regularization=10, random_state=42))]
    history, best, best_score, best_name = [], None, -1, None
    with threadpool_limits(limits=args.threads):
        for name, model in candidates:
            if name == 'histgradient':
                model.fit(x, y, sample_weight=weights)
            else:
                if balance_sources:
                    model.fit(x, y, logisticregression__sample_weight=weights)
                else:
                    model.fit(x, y)
            score = selection_score(vy[tune], model.predict(vx[tune]),
                                    [entry for entry, selected in zip(rows['validation'], tune) if selected],
                                    macro_validation)
            history.append({'candidate': name, 'validation_balanced_accuracy': score})
            print(json.dumps(history[-1]), flush=True)
            if score > best_score:
                best, best_score, best_name = model, score, name
        p = np.clip(best.predict_proba(vx[cal])[:,1], 1e-7, 1-1e-7)
        logits = np.log(p/(1-p))
        sigmoid = lambda z: 1/(1+np.exp(-np.clip(z, -30, 30)))
        cal_weights = source_balanced_weights([entry for entry, selected in zip(rows['validation'], cal) if selected]) if balance_sources else None
        temperature = float(minimize_scalar(lambda t: log_loss(vy[cal], sigmoid(logits/t), labels=[0,1], sample_weight=cal_weights),
            bounds=(.1,10), method='bounded').x)
        joblib.dump({'selected': best, 'temperature': temperature, 'history': history}, output/'selection-before-test.joblib')
        tx, ty = features('test')
        raw = np.clip(best.predict_proba(tx)[:,1], 1e-7, 1-1e-7)
        p = sigmoid(np.log(raw/(1-raw))/temperature)
        evaluation = metrics(ty, p)
        test_datasets = np.array([entry.get('dataset', 'base') for entry in rows['test']])
        evaluation_by_dataset = {name: metrics(ty[test_datasets == name], p[test_datasets == name])
                                 for name in sorted(set(test_datasets))}
        test_pairs = np.array([entry['pair_id'] for entry in rows['test']])
        rng = np.random.default_rng(42)
        group_scores = []
        for _ in range(2000):
            chosen = rng.choice(sorted(groups['test']), len(groups['test']), replace=True)
            indices = np.concatenate([np.flatnonzero(test_pairs == group) for group in chosen])
            group_scores.append(float(np.mean((p[indices]>=.5) == ty[indices])))
        evaluation['source_group_bootstrap_accuracy_interval'] = np.quantile(group_scores, [.025,.975]).tolist()
        matrix = evaluation['confusion_matrix']
        evaluation['real_recall'] = matrix[0][0]/sum(matrix[0])
        evaluation['fake_recall'] = matrix[1][1]/sum(matrix[1])
        video_task = manifest.get('purpose') == 'video_synthetic_or_manipulated'
        card = {'name': 'Experimental synthetic/manipulated video classifier' if video_task else 'Learned face-video consistency classifier',
            'version': '1.0-'+manifest['version'],
            'purpose': 'video_synthetic_or_manipulated' if video_task else 'temporal',
            'prediction_enabled': False, 'locally_validated': False, 'deployment_ready': False,
            'feature_version': FEATURE_VERSION, 'status': 'experimental',
            'scope': ('Source-grouped video manipulation features. Does not establish AI authorship, factual truth or audio authenticity.'
                      if video_task else 'DeepfakeTIMIT-style face-swap consistency over sampled frames. Not general video truth, AI authorship, or audio authenticity.'),
            'evaluation': evaluation, 'target_accuracy': .90,
            'evaluation_by_dataset': evaluation_by_dataset,
            'validation_metric': manifest.get('validation_metric', 'balanced_accuracy'),
            'balanced_training_datasets': balance_sources,
            'target_met': evaluation['accuracy'] >= .90 and evaluation['balanced_accuracy'] >= .90,
            'split_counts': {split: len(rows[split]) for split in rows},
            'class_counts': {split: {'real': sum(entry['label']=='real' for entry in rows[split]),
                'fake': sum(entry['label']=='fake' for entry in rows[split])} for split in rows},
            'majority_class_test_accuracy': float(max(np.mean(ty==0), np.mean(ty==1))),
            'source_group_counts': {split: len(value) for split,value in groups.items()},
            'validation_selection': best_score, 'selected_head': best_name, 'temperature': temperature,
            'backbone': {'name': 'ImageNet MobileNetV3 Small (frozen)', 'sha256': base_sha,
                'fine_tuned_on_user_archive': False},
            'training_history': history, 'dataset': manifest.get('provenance', {}),
            'dataset_licenses': manifest.get('license', {}),
            'limitations': (manifest.get('limitations', []) + [
                'External evaluation datasets were not used for training, selection or calibration.',
                'Training on face manipulation does not establish accuracy on arbitrary fully generated scenes.']
                if video_task else ['Only a small set of source/identity groups from one older face-swap dataset was evaluated.',
                'Real clips originate as JPEG sequences and fake clips as AVI. Normalization cannot eliminate every source/compression artifact.',
                'Accuracy on this dataset does not establish accuracy on newer generators, social-media videos or audio deepfakes.',
                'Source-group bootstrap intervals are more appropriate than treating correlated quality variants as independent videos.'])}
        joblib.dump({'kind': 'temporal-statistics', 'feature_version': FEATURE_VERSION,
            'classifier': best, 'temperature': temperature, 'card': card,
            'base_backbone_relative_path': os.path.relpath(Path(args.backbone).resolve(), output.resolve()),
            'base_backbone_sha256': base_sha}, output/'model.joblib')
        (output/'model-card.json').write_text(json.dumps(card, indent=2), encoding='utf8')
        print(json.dumps(card, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='data/deepfaketimit/manifest.json')
    parser.add_argument('--output', default='models/video-temporal')
    parser.add_argument('--backbone', default='models/image/backbone.pt')
    parser.add_argument('--cache-dir', help='Reuse audited frozen-frame features from a prior run.')
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--cache-available', action='store_true', help='Cache train/validation features only; never fit or access test media.')
    main(parser.parse_args())
