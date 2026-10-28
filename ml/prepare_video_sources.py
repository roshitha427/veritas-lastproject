"""Prepare supplied video datasets as data; never execute downloaded code.

FVC labels concern factual verification and never enter origin training.
DF26 is held-out evaluation only. Celeb-DF defaults to external evaluation.
DFDC originals and their manipulated descendants remain in the same split.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import random
from zipfile import ZipFile


MEDIA = {'.mp4', '.avi', '.mov', '.webm'}


def safe_name(name):
    name = name.replace('\\', '/')
    return bool(name and not PurePosixPath(name).is_absolute() and not PureWindowsPath(name).drive
                and '..' not in PurePosixPath(name).parts and ':' not in name and '\x00' not in name)


class DatasetFiles:
    def __init__(self, source):
        self.source = Path(source).resolve()
        self.zip = ZipFile(self.source) if self.source.is_file() else None
        if self.zip:
            self.names = [item.filename for item in self.zip.infolist() if not item.is_dir()]
            if len(set(self.names)) != len(self.names):
                self.close()
                raise ValueError('Duplicate archive member names are ambiguous.')
            if any(not safe_name(name) for name in self.names):
                self.close()
                raise ValueError('Unsafe archive member path.')
        else:
            if not self.source.is_dir():
                raise ValueError('Supply an existing dataset folder or ZIP.')
            self.names = [p.relative_to(self.source).as_posix() for p in self.source.rglob('*')
                          if p.is_file() and p.resolve().is_relative_to(self.source)]
        self.available = set(self.names)

    def close(self):
        if self.zip:
            self.zip.close()

    def read(self, name, limit=16 * 1024 * 1024):
        if name not in self.available or not safe_name(name):
            raise ValueError('Dataset member is missing or unsafe.')
        if self.zip:
            if self.zip.getinfo(name).file_size > limit:
                raise ValueError('Metadata exceeds the bounded read size.')
            return self.zip.read(name)
        path = self.source / name
        if path.stat().st_size > limit:
            raise ValueError('Metadata exceeds the bounded read size.')
        return path.read_bytes()

    def media(self, name):
        if name not in self.available or not safe_name(name):
            raise ValueError('Referenced media is missing or unsafe.')
        if PurePosixPath(name).suffix.lower() not in MEDIA:
            raise ValueError('Unsupported video file extension.')
        size = self.zip.getinfo(name).file_size if self.zip else (self.source / name).stat().st_size
        if size <= 0 or size > 512 * 1024 * 1024:
            raise ValueError('Expected a nonempty video of at most 512 MB.')
        handle = self.zip.open(name) if self.zip else (self.source / name).open('rb')
        with handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        reference = ({'kind': 'zip_video', 'archive': str(self.source), 'member': name} if self.zip else
                     {'kind': 'video', 'filepath': str(self.source / name)})
        return dict(reference, sha256=digest, id=hashlib.sha256((str(self.source) + '/' + name).encode()).hexdigest())


class Groups:
    def __init__(self):
        self.parents = {}

    def find(self, name):
        self.parents.setdefault(name, name)
        while self.parents[name] != name:
            self.parents[name] = self.parents[self.parents[name]]
            name = self.parents[name]
        return name

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parents[max(a, b)] = min(a, b)


def audit_fvc(source, output):
    inventory = DatasetFiles(source)
    try:
        wanted = ['FVC.csv', 'FVC_dup.csv', 'FVC_text_queries.csv']
        result = {'name': 'Fake Video Corpus 2018', 'task': 'video_factual_verification',
                  'usable_for_visual_origin_training': False, 'embedded_video_count': 0,
                  'label_policy': 'Preserve factual/debunking labels; never map fake to AI-generated.',
                  'license': 'No dataset license supplied in this archive.', 'files': {}}
        tables = {}
        for wanted_name in wanted:
            matches = [name for name in inventory.names if PurePosixPath(name).name == wanted_name]
            if len(matches) != 1:
                raise ValueError('Expected exactly one ' + wanted_name)
            data = inventory.read(matches[0])
            rows = list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))))
            if not rows or not {'cascade_id', 'video_url', 'label'}.issubset(rows[0]):
                raise ValueError('Unexpected FVC schema.')
            result['files'][wanted_name] = {'rows': len(rows), 'labels': dict(Counter(row['label'] for row in rows)),
                                          'sha256': hashlib.sha256(data).hexdigest()}
            tables[wanted_name] = data
        result['embedded_video_count'] = sum(PurePosixPath(n).suffix.lower() in MEDIA for n in inventory.names)
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        for name, data in tables.items():
            (output / name).write_bytes(data)
        if Path(source).is_file():
            with Path(source).open('rb') as handle:
                result['archive_sha256'] = hashlib.file_digest(handle, 'sha256').hexdigest()
        (output / 'audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        return result
    finally:
        inventory.close()


def dfdc(inventory):
    rows, groups = [], Groups()
    metadata = [name for name in inventory.names if PurePosixPath(name).name == 'metadata.json']
    if not metadata:
        raise ValueError('DFDC needs labelled training metadata.json, not an unlabelled test ZIP.')
    for name in metadata:
        values = json.loads(inventory.read(name))
        if not isinstance(values, dict):
            raise ValueError('Unexpected DFDC metadata schema.')
        folder = PurePosixPath(name).parent
        for filename, record in values.items():
            if not safe_name(filename) or PurePosixPath(filename).name != filename:
                raise ValueError('Unsafe DFDC media reference.')
            label = str(record.get('label', '')).upper()
            if label not in ('REAL', 'FAKE'):
                raise ValueError('DFDC label must explicitly be REAL or FAKE.')
            original = record.get('original') or filename
            if not safe_name(original) or PurePosixPath(original).name != original:
                raise ValueError('Unsafe original-video reference.')
            member = str(folder / filename)
            if member not in inventory.available:
                raise ValueError('DFDC metadata references missing media: ' + member)
            group = 'dfdc/' + original
            groups.union('dfdc/' + filename, group)
            row = dict(inventory.media(member), dataset='DFDC', label=label.lower(),
                       annotation='authentic' if label == 'REAL' else 'face_manipulated', pair_id=group,
                       role='training_pool', original=original)
            if record.get('split', 'train') not in ('train', 'training'):
                row['role'] = 'evaluation'
            rows.append(row)
    for row in rows:
        row['pair_id'] = groups.find(row['pair_id'])
    return rows


def celeb(inventory):
    rows = []
    for name in inventory.names:
        parts = PurePosixPath(name).parts
        category = next((part for part in parts if part in ('Celeb-real', 'YouTube-real', 'Celeb-synthesis')), None)
        if not category or PurePosixPath(name).suffix.lower() not in MEDIA:
            continue
        fake = category == 'Celeb-synthesis'
        rows.append(dict(inventory.media(name), dataset='Celeb-DF-v2', label='fake' if fake else 'real',
                         annotation='face_manipulated' if fake else 'authentic', role='evaluation',
                         pair_id='celeb-evaluation/' + PurePosixPath(name).stem))
    if not rows:
        raise ValueError('No Celeb-DF videos found. A source-code/demo repository is not the dataset.')
    return rows


def df26(inventory):
    if not any(name.endswith('metadata/generators.csv') for name in inventory.names):
        raise ValueError('DF26 needs the official generator inventory and authorized video files.')
    rows = []
    for name in inventory.names:
        parts = PurePosixPath(name).parts
        category_index = next((i for i, part in enumerate(parts) if part in ('real', 'fake')), None)
        if category_index is None or PurePosixPath(name).suffix.lower() not in MEDIA:
            continue
        parts = parts[category_index:]
        fake = parts[0] == 'fake'
        if len(parts) != (4 if fake else 3):
            raise ValueError('Unexpected DF26 scenario/generator/video layout.')
        scenario, clip_id = parts[1], PurePosixPath(name).stem
        rows.append(dict(inventory.media(name), dataset='DF26', label='fake' if fake else 'real',
                         annotation='ai_generated' if fake else 'authentic', role='evaluation',
                         generator=parts[2] if fake else 'real', scenario=scenario,
                         pair_id='df26/' + scenario + '/' + clip_id,
                         training_prohibited=True,
                         usage='Held-out evaluation only; no calibration, thresholds, or model selection.'))
    if not rows:
        raise ValueError('DF26 metadata contains no authorized local video files.')
    return rows


def assign_splits(rows, seed=42):
    # Join exact duplicates and original/fake chains across all imported chunks.
    groups = Groups()
    by_hash = {}
    for row in rows:
        group = row['pair_id']
        digest = row['sha256']
        if digest in by_hash:
            previous = by_hash[digest]
            if previous['label'] != row['label']:
                raise ValueError('Identical videos have conflicting labels.')
            groups.union(group, previous['pair_id'])
        else:
            by_hash[digest] = row
    for row in rows:
        row['pair_id'] = groups.find(row['pair_id'])
    reserved = {row['pair_id'] for row in rows if row['role'] == 'evaluation'}
    eligible = sorted({row['pair_id'] for row in rows if row['role'] == 'training_pool'} - reserved)
    random.Random(seed).shuffle(eligible)
    total = len(eligible)
    assignments = {group: ('train' if i < int(total * .7) else 'validation' if i < int(total * .85) else 'test')
                   for i, group in enumerate(eligible)}
    seen = set()
    output = []
    for row in rows:
        if row['sha256'] in seen:
            continue
        seen.add(row['sha256'])
        if row['pair_id'] in reserved:
            row['split'] = 'external_evaluation'
        else:
            row['split'] = assignments[row['pair_id']]
        output.append(row)
    validation_groups = sorted({r['pair_id'] for r in output if r['split'] == 'validation'})
    for row in output:
        if row['split'] == 'validation':
            row['validation_role'] = ('selection' if row['pair_id'] in validation_groups[:len(validation_groups)//2]
                                      else 'calibration')
    return output


def prepare(sources, output):
    output = Path(output)
    if output.exists():
        raise ValueError('Choose a new output directory to preserve earlier manifests.')
    rows, provenance = [], []
    for kind, source in sources:
        inventory = DatasetFiles(source)
        try:
            rows.extend({'dfdc': dfdc, 'celeb': celeb, 'df26': df26}[kind](inventory))
            provenance.append({'dataset': kind, 'source': str(inventory.source),
                               'role': 'training_pool' if kind == 'dfdc' else 'external_evaluation',
                               'metadata_sha256': {name: hashlib.sha256(inventory.read(name)).hexdigest()
                                   for name in inventory.names if PurePosixPath(name).name in
                                   ('metadata.json', 'generators.csv', 'List_of_testing_videos.txt')}})
        finally:
            inventory.close()
    rows = assign_splits(rows)
    training = [row for row in rows if row['split'] != 'external_evaluation']
    evaluation = [row for row in rows if row['split'] == 'external_evaluation']
    result = {'schema_version': 1, 'name': 'Additional labelled video sources',
              'purpose': 'video_synthetic_or_manipulated', 'version': 'video-sources-grouped42',
              'download_status': 'complete',
              'provenance': {'sources': provenance},
              'split_policy': 'DFDC original/derivative and exact-hash groups; Celeb-DF and DF26 external evaluation only.',
              'limitations': ['Source groups are not a claim of identity-disjoint DFDC subjects.',
                              'DF26 may not be used to select models, tune thresholds, or calibrate in this setup.',
                              'Celeb-DF and DFDC primarily cover facial manipulation, not arbitrary synthetic scenes.'],
              'counts': dict(Counter(row['split'] for row in rows)),
              'entries': training, 'evaluation_entries': evaluation}
    output.mkdir(parents=True)
    (output / 'manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fvc', help='Supplied FVC ZIP; factual verification only.')
    parser.add_argument('--celeb', action='append', default=[], help='Celeb-DF folder or ZIP, external evaluation.')
    parser.add_argument('--dfdc', action='append', default=[], help='DFDC labelled training folder or chunk ZIP.')
    parser.add_argument('--df26', action='append', default=[], help='Authorized DF26 folder or ZIP, evaluation only.')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.fvc:
        result = audit_fvc(args.fvc, Path(args.output) / 'fvc')
        print(json.dumps(result, indent=2))
    sources = [(kind, path) for kind in ('celeb', 'dfdc', 'df26') for path in getattr(args, kind)]
    if sources:
        result = prepare(sources, Path(args.output) / 'videos')
        print(json.dumps({'counts': result['counts'], 'training_rows': len(result['entries']),
                          'external_evaluation_rows': len(result['evaluation_entries'])}, indent=2))
    elif not args.fvc:
        parser.error('Supply at least one dataset source.')


if __name__ == '__main__':
    main()
