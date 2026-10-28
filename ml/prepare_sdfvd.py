"""Audit paired SDFVD videos and combine source-disjoint training manifests."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import random
import re
import shutil
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from ml.prepare_video_sources import DatasetFiles, Groups
from ml.train_temporal import clip_frames
from ml.temporal import FEATURE_VERSION


SOURCE = 'https://data.mendeley.com/datasets/bcmkfgct2s/1'


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def csv_audit(paths):
    seen, receipts = {}, []
    for path in paths:
        with ZipFile(path) as archive:
            names = [name for name in archive.namelist() if PurePosixPath(name).name == 'Deepfake_Misinformation_Dataset.csv']
            if len(names) != 1:
                raise ValueError('Expected the supplied misinformation CSV.')
            if archive.getinfo(names[0]).file_size > 16 * 1024 * 1024:
                raise ValueError('CSV exceeds the metadata read bound.')
            data = archive.read(names[0])
        sha = hashlib.sha256(data).hexdigest()
        rows = list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))))
        receipts.append({'path': str(Path(path).resolve()), 'archive_sha256': digest(path),
                         'csv_sha256': sha, 'rows': len(rows), 'duplicate_of': seen.get(sha),
                         'content_types': dict(Counter(row.get('CONTENT TYPE') for row in rows)),
                         'decision': 'Not used for visual training: no source media or media URLs; '
                                     'precomputed detector scores and post-outcome fields risk target leakage.'})
        seen.setdefault(sha, str(Path(path).resolve()))
    return receipts


def phash(image):
    gray = np.asarray(image.convert('L').resize((32, 32)), dtype=np.float32)
    values = cv2.dct(gray)[:8, :8].ravel()[1:]
    return values > np.median(values)


def main(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError('Choose a new preparation directory to preserve previous data.')
    output.mkdir(parents=True)
    archive_path = Path(args.archive).resolve()
    with ZipFile(archive_path) as outer:
        nested = [item for item in outer.infolist() if PurePosixPath(item.filename).name == 'SDFVD.zip']
        if nested:
            if len(nested) != 1 or nested[0].file_size > 1024 ** 3:
                raise ValueError('Unexpected nested SDFVD archive.')
            source = output / 'SDFVD.zip'
            with outer.open(nested[0]) as incoming, source.open('wb') as destination:
                shutil.copyfileobj(incoming, destination, length=1024 * 1024)
        else:
            source = archive_path
    inventory = DatasetFiles(source)
    entries, thumbnails, fingerprints = [], {}, {}
    groups = Groups()
    try:
        for name in sorted(inventory.names):
            match = re.fullmatch(r'(?:.*/)?videos_(real|fake)/(v|vs)(\d+)\.mp4', name)
            if not match:
                raise ValueError('Unexpected SDFVD media member: ' + name)
            label, prefix, number = match.groups()
            if (label == 'real' and prefix != 'v') or (label == 'fake' and prefix != 'vs'):
                raise ValueError('SDFVD folder and filename labels disagree.')
            group = 'sdfvd/pair-' + number
            groups.find(group)
            entry = dict(inventory.media(name), dataset='SDFVD', label=label, pair_id=group,
                         role='training_pool', annotation='authentic' if label == 'real' else 'face_swapped',
                         source_url=SOURCE)
            frames = clip_frames(entry)
            if len(frames) < 3:
                raise ValueError('Too few decoded SDFVD frames.')
            selected = [frames[i] for i in (0, len(frames) // 2, len(frames) - 1)]
            fingerprints[entry['id']] = [phash(frame) for frame in selected]
            entry['sampled_frame_count'] = len(frames)
            entry['decoded_size'] = list(frames[0].size)
            entries.append(entry)
            if label == 'real':
                thumbnails[group] = frames[len(frames) // 2]
        if Counter(row['label'] for row in entries) != {'real': 53, 'fake': 53}:
            raise ValueError('Expected the published 53 real and 53 fake SDFVD videos.')
        pair_labels = {}
        for row in entries:
            pair_labels.setdefault(row['pair_id'], []).append(row['label'])
        if any(sorted(labels) != ['fake', 'real'] for labels in pair_labels.values()):
            raise ValueError('Every SDFVD pair must contain one real and one fake clip.')
        # Conservative near-duplicate check: all three ordered frame hashes close.
        merges = []
        for index, a in enumerate(entries):
            for b in entries[index + 1:]:
                if a['pair_id'] == b['pair_id']:
                    continue
                distances = [int(np.count_nonzero(x != y)) for x, y in
                             zip(fingerprints[a['id']], fingerprints[b['id']])]
                if max(distances) <= 6 or a['sha256'] == b['sha256']:
                    if a['sha256'] == b['sha256'] and a['label'] != b['label']:
                        raise ValueError('Exact SDFVD duplicate has contradictory labels.')
                    groups.union(a['pair_id'], b['pair_id'])
                    merges.append({'a': a['pair_id'], 'b': b['pair_id'], 'frame_hash_distances': distances})
        for entry in entries:
            entry['pair_id'] = groups.find(entry['pair_id'])
        group_ids = sorted({row['pair_id'] for row in entries})
        random.Random(42).shuffle(group_ids)
        ntrain, nvalidation = int(len(group_ids) * .7), round(len(group_ids) * .15)
        assignments = {g: ('train' if i < ntrain else 'validation' if i < ntrain + nvalidation else 'test')
                       for i, g in enumerate(group_ids)}
        val_groups = sorted(g for g in group_ids if assignments[g] == 'validation')
        for entry in entries:
            entry['split'] = assignments[entry['pair_id']]
            if entry['split'] == 'validation':
                entry['validation_role'] = ('selection' if entry['pair_id'] in val_groups[:len(val_groups)//2]
                                            else 'calibration')
        base = json.loads(Path(args.base).read_text(encoding='utf-8'))
        backbone_sha = digest(args.backbone)
        base_entries = [dict(entry, dataset=entry.get('dataset', 'DeepfakeTIMIT'),
                             annotation='authentic' if entry['label'] == 'real' else 'face_swapped',
                             feature_backbone_sha256=backbone_sha,
                             feature_cache_key=hashlib.sha256((FEATURE_VERSION + backbone_sha +
                                 json.dumps(entry, sort_keys=True)).encode()).hexdigest())
                        for entry in base['entries']]
        if {r['sha256'] for r in entries} & {r.get('sha256') for r in base_entries}:
            raise ValueError('SDFVD overlaps base data; grouping requires review.')
        metadata_receipts = csv_audit(args.metadata_archive)
        manifest = {'schema_version': 1, 'name': 'DeepfakeTIMIT plus paired SDFVD',
                    'version': 'sdfvd-20261022-group42', 'download_status': 'complete',
                    'purpose': 'video_synthetic_or_manipulated',
                    'validation_metric': 'macro_dataset_balanced_accuracy', 'balance_training_datasets': True,
                    'entries': base_entries + entries,
                    'provenance': {'base': base.get('provenance'), 'base_manifest_sha256': digest(args.base),
                                   'sdfvd_source': SOURCE, 'sdfvd_doi': '10.17632/bcmkfgct2s.1',
                                   'outer_archive_sha256': digest(archive_path), 'inner_archive_sha256': digest(source),
                                   'csv_archives': metadata_receipts},
                    'license': {'base': base.get('license'), 'sdfvd': 'CC BY 4.0 per original Mendeley release'},
                    'limitations': ['SDFVD contains face swaps, not a benchmark for arbitrary fully generated video.',
                                    'Matching vN/vsN and near-duplicate frame groups are kept together; full identity independence is unestablished.',
                                    'SDFVD test is small; the older base test was previously inspected.',
                                    'Supplied metadata CSVs contain no source videos and are not visual training input.']}
        audit = {'sdfvd_counts': dict(Counter(e['split'] for e in entries)),
                 'combined_counts': dict(Counter(e['split'] for e in manifest['entries'])),
                 'sdfvd_group_counts': dict(Counter(assignments.values())), 'near_duplicate_merges': merges,
                 'decoded_media': len(entries), 'csv_archives': metadata_receipts}
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        (output / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
        sheet = Image.new('RGB', (1000, 150 * ((len(thumbnails) + 5) // 6)), '#e6ebef')
        draw = ImageDraw.Draw(sheet)
        for index, (group, image) in enumerate(sorted(thumbnails.items())):
            tile = ImageOps.contain(image, (158, 120))
            x, y = index % 6 * 166, index // 6 * 150
            sheet.paste(tile, (x, y + 20))
            draw.text((x + 3, y + 3), group, fill='black')
        sheet.save(output / 'real-source-contact.jpg')
        print(json.dumps(audit, indent=2), flush=True)
    finally:
        inventory.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--metadata-archive', action='append', default=[])
    parser.add_argument('--base', default='data/deepfaketimit/manifest.json')
    parser.add_argument('--backbone', default='models/image/backbone.pt')
    parser.add_argument('--output', default='data/sdfvd-20261022')
    main(parser.parse_args())
