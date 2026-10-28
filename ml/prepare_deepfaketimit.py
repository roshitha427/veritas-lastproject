"""Download the official open TIMIT corpora and prepare source-grouped manifests.

No submitted scripts or serialized checkpoints are loaded. Downloaded archives
are inspected as data; only selected bounded regular media files are copied.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import random
import re
import shutil
import tarfile
import time
import zipfile
from email.utils import parsedate_to_datetime

import requests


FAKE_RECORD = 'https://zenodo.org/api/records/4068245'
REAL_RECORD = 'https://zenodo.org/api/records/158963'
FAKE_ARCHIVE = 'DeepfakeTIMIT.tar.gz'
FAKE_MD5 = '79b5dcee896ab825db43f1dfbdc8fdb5'
READ_LIMIT = 2 * 1024 * 1024
FAKE_PATTERN = re.compile(r'DeepfakeTIMIT/(higher_quality|lower_quality)/([fm][a-z]{3}[0-9])/([a-z]+[0-9]+)-video-([fm][a-z]{3}[0-9])\.avi')


def hashes(path):
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            md5.update(block)
            sha256.update(block)
    return {'md5': md5.hexdigest(), 'sha256': sha256.hexdigest()}


def safe_member(name):
    normal = name.replace('\\', '/')
    return not (
        PurePosixPath(normal).is_absolute() or PureWindowsPath(name).drive
        or '..' in PurePosixPath(normal).parts or ':' in normal or '\x00' in normal
    )


def fetch_record(url, path):
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    response = requests.get(url, timeout=(20, 60))
    response.raise_for_status()
    record = response.json()
    if record['metadata']['access_right'] != 'open':
        raise ValueError('Dataset access is not open')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding='utf-8')
    return record


def download(item, path):
    expected_md5 = item['checksum'].removeprefix('md5:')
    expected_size = item['size']
    if path.exists():
        digest = hashes(path)
        if path.stat().st_size != expected_size or digest['md5'] != expected_md5:
            raise ValueError('Cached archive checksum mismatch: ' + path.name)
        return {'path': str(path.resolve()), 'bytes': expected_size, **digest}
    partial = path.with_name(path.name + '.part')
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > expected_size:
            raise ValueError('Partial download exceeds expected size')
        if offset == expected_size:
            break
        try:
            headers = {'Range': 'bytes=' + str(offset) + '-'} if offset else {}
            with requests.get(item['links']['self'], headers=headers, stream=True, timeout=(20, 90)) as response:
                if response.status_code in (429, 503) and attempt < 4:
                    retry_after = response.headers.get('Retry-After', '60')
                    try:
                        delay = float(retry_after)
                    except ValueError:
                        try:
                            delay = (parsedate_to_datetime(retry_after) - datetime.now(timezone.utc)).total_seconds()
                        except (TypeError, ValueError):
                            delay = 60
                    print(json.dumps({'download_rate_limited': path.name, 'retry_after_seconds': max(1, delay)}), flush=True)
                    time.sleep(max(1, delay))
                    continue
                response.raise_for_status()
                if response.status_code == 206:
                    content_range = response.headers.get('Content-Range', '')
                    if not content_range.startswith('bytes ' + str(offset) + '-'):
                        raise ValueError('Server returned unexpected download range')
                    mode = 'ab'
                else:
                    mode = 'wb'
                    offset = 0
                with partial.open(mode) as handle:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if block:
                            offset += len(block)
                            if offset > expected_size:
                                raise ValueError('Download exceeds published size')
                            handle.write(block)
            if partial.stat().st_size != expected_size:
                raise requests.ConnectionError('Incomplete stream')
            break
        except (requests.RequestException, OSError) as exc:
            print(json.dumps({'download_retry': path.name, 'attempt': attempt + 1, 'error': type(exc).__name__}), flush=True)
            if attempt == 4:
                raise
            time.sleep(min(2 ** attempt, 8))
    digest = hashes(partial)
    if partial.stat().st_size != expected_size or digest['md5'] != expected_md5:
        raise ValueError('Downloaded archive checksum mismatch: ' + path.name)
    partial.replace(path)
    print(json.dumps({'download_complete': path.name, 'bytes': expected_size}), flush=True)
    return {'path': str(path.resolve()), 'bytes': expected_size, **digest}


def inspect_fake(path):
    members = []
    documents = {}
    with tarfile.open(path, 'r:*') as archive:
        for member in archive:
            if not safe_member(member.name) or member.issym() or member.islnk():
                raise ValueError('Unsafe archive member: ' + member.name)
            if not member.isfile():
                continue
            members.append({'path': member.name, 'bytes': member.size})
            if Path(member.name).suffix.lower() in {'.txt', '.md', '.json', '.csv'} and member.size <= READ_LIMIT:
                handle = archive.extractfile(member)
                documents[member.name] = handle.read().decode('utf-8', errors='replace') if handle else ''
    return {'members': members, 'documents': documents}


def split_plan(audit):
    clips = []
    graph = defaultdict(set)
    for member in audit['members']:
        if not member['path'].lower().endswith('.avi'):
            continue
        match = FAKE_PATTERN.fullmatch(member['path'])
        if not match:
            raise ValueError('Unexpected fake video naming scheme: ' + member['path'])
        quality, subject, sentence, swapped_identity = match.groups()
        graph[subject].add(swapped_identity)
        graph[swapped_identity].add(subject)
        clips.append({**member, 'quality': quality, 'subject': subject, 'sentence': sentence, 'swapped_identity': swapped_identity})
    components = []
    visited = set()
    for subject in sorted(graph):
        if subject in visited:
            continue
        stack = [subject]
        component = set()
        while stack:
            value = stack.pop()
            if value in component:
                continue
            component.add(value)
            stack.extend(graph[value] - component)
        visited.update(component)
        components.append(sorted(component))
    if len(clips) != 640 or len(components) != 16 or any(len(pair) != 2 for pair in components):
        raise ValueError('Dataset does not match the audited 640 videos / 16 identity pairs')
    random.Random(42).shuffle(components)
    plan = []
    for index, pair in enumerate(components):
        split = 'train' if index < 8 else 'validation' if index < 12 else 'test'
        plan.append({
            'pair_id': '__'.join(pair), 'subjects': pair, 'split': split,
            'validation_role': 'selection' if 8 <= index < 10 else 'calibration' if 10 <= index < 12 else None,
        })
    return clips, plan


def write_manifest(root, result):
    result['counts'] = dict(Counter(entry['split'] + '/' + entry['label'] for entry in result['entries']))
    partial = root / 'manifest.json.tmp'
    partial.write_text(json.dumps(result, indent=2), encoding='utf-8')
    partial.replace(root / 'manifest.json')


def fake_entries(root, fake_path, clips, groups):
    entries = []
    wanted = {clip['path']: clip for clip in clips}
    with tarfile.open(fake_path, 'r:*') as archive:
        for member in archive:
            if member.name not in wanted:
                continue
            clip = wanted[member.name]
            if not member.isfile() or member.size > 25 * 1024 * 1024 or not safe_member(member.name):
                raise ValueError('Invalid video member')
            group = groups[clip['subject']]
            # Components of this output path come only from the strict regex.
            target = root / 'media' / 'fake' / clip['quality'] / clip['subject'] / (clip['sentence'] + '-video-' + clip['swapped_identity'] + '.avi')
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open('wb') as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
            if target.stat().st_size != member.size:
                raise ValueError('Incomplete copied video')
            clip_id = 'DeepfakeTIMIT/' + clip['quality'] + '/' + clip['subject'] + '/' + clip['sentence']
            entries.append({
                'id': clip_id, 'clip_id': clip_id, 'label': 'fake', 'kind': 'video',
                'filepath': str(target.resolve()), 'archive': str(fake_path.resolve()), 'member': member.name,
                'sha256': hashes(target)['sha256'],
                'subject': clip['subject'], 'swapped_identity': clip['swapped_identity'],
                'source_clip_id': clip['subject'] + '/' + clip['sentence'], 'quality': clip['quality'],
                'pair_id': group['pair_id'], 'split': group['split'], 'validation_role': group['validation_role'],
                'fps': 25.0, 'fps_source': 'VidTIMIT documentation; actual video codec metadata must be checked by reader.',
            })
    return sorted(entries, key=lambda item: item['id'])


def real_entries(path, subject, expected_sentences, group):
    sequences = defaultdict(list)
    pattern = re.compile(re.escape(subject) + r'/video/([a-z]+[0-9]+)/([0-9]+)')
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            if not safe_member(member.filename):
                raise ValueError('Unsafe original archive member')
            match = pattern.fullmatch(member.filename)
            if not match or match[1] not in expected_sentences or member.is_dir():
                continue
            if member.file_size > 5 * 1024 * 1024:
                raise ValueError('Oversized source frame')
            sequences[match[1]].append(member.filename)
        if set(sequences) != expected_sentences:
            raise ValueError('Original/fake source clip mismatch for ' + subject)
        entries = []
        for sentence, members in sorted(sequences.items()):
            members.sort(key=lambda name: int(name.rsplit('/', 1)[1]))
            if len(members) < 12:
                raise ValueError('Too few frames in original clip')
            for name in (members[0], members[-1]):
                if not archive.read(name).startswith(b'\xff\xd8'):
                    raise ValueError('Original frame is not a JPEG image')
            clip_id = 'VidTIMIT/' + subject + '/' + sentence
            entries.append({
                'id': clip_id, 'clip_id': clip_id, 'label': 'real', 'kind': 'zip_jpeg_sequence',
                'archive': str(path.resolve()), 'members': members, 'frame_count': len(members),
                'subject': subject, 'source_clip_id': subject + '/' + sentence, 'quality': 'original_jpeg',
                'pair_id': group['pair_id'], 'split': group['split'], 'validation_role': group['validation_role'],
                'fps': 25.0,
                'fps_source': 'https://conradsanderson.id.au/vidtimit/zips/vidtimit_documentation.pdf',
            })
    return entries


def prepare_originals(root, real_record, clips, plan, result, workers):
    groups = {subject: group for group in plan for subject in group['subjects']}
    subjects = sorted(groups)
    files = {item['key']: item for item in real_record['files']}
    items = [files[subject + '.zip'] for subject in subjects]
    print(json.dumps({'matched_original_subjects': len(items), 'original_bytes': sum(item['size'] for item in items), 'workers': workers}), flush=True)
    expected = defaultdict(set)
    for clip in clips:
        expected[clip['subject']].add(clip['sentence'])

    def one(item):
        subject = item['key'].removesuffix('.zip')
        path = root / 'cache' / item['key']
        provenance = download(item, path)
        entries = real_entries(path, subject, expected[subject], groups[subject])
        return subject, provenance, entries

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(one, item) for item in items]
        for future in as_completed(futures):
            subject, provenance, entries = future.result()
            result['entries'].extend(entries)
            result['downloads'][subject] = provenance
            result['pending_subjects'].remove(subject)
            result['entries'].sort(key=lambda entry: entry['id'])
            write_manifest(root, result)
            print(json.dumps({'subject_ready': subject, 'real_clips': len(entries), 'pending_subjects': len(result['pending_subjects'])}), flush=True)


def prepare(root, fake_record, real_record, fake_path, provenance, audit, workers):
    clips, plan = split_plan(audit)
    groups = {subject: group for group in plan for subject in group['subjects']}
    # Persist the predetermined plan before opening any original media.
    (root / 'split-plan.json').write_text(json.dumps({'seed': 42, 'groups': plan}, indent=2), encoding='utf-8')
    (root / 'DeepfakeTIMIT-README.txt').write_text(audit['documents']['DeepfakeTIMIT/README.txt'], encoding='utf-8')
    documentation = next(item for item in real_record['files'] if item['key'] == 'vidtimit_documentation.pdf')
    download(documentation, root / 'vidtimit_documentation.pdf')
    result = {
        'schema_version': 1, 'name': 'DeepfakeTIMIT with matched VidTIMIT originals',
        'version': '1.0-9291bb413e09-group42', 'prepared_at': datetime.now(timezone.utc).isoformat(),
        'modality': 'video', 'download_status': 'incomplete', 'seed': 42,
        'label_meaning': 'real = original VidTIMIT JPEG video sequence; fake = DeepfakeTIMIT face-swapped video. Labels do not indicate news truth, audio manipulation, or universal AI generation.',
        'license': {'deepfaketimit': fake_record['metadata']['license'], 'vidtimit': real_record['metadata']['license'],
                    'note': 'CC-BY-NC-ND-4.0 according to official Zenodo metadata. Source archives remain intact. Private research preprocessing only; modified source media are not redistributed. Preserve original notices and citations.'},
        'provenance': {'fake_record': FAKE_RECORD, 'real_record': REAL_RECORD,
                       'fake_homepage': 'https://www.idiap.ch/en/dataset/deepfaketimit',
                       'real_homepage': 'https://conradsanderson.id.au/vidtimit/',
                       'citations': ['P. Korshunov and S. Marcel. DeepFakes: a New Threat to Face Recognition? Assessment and Detection. 2018. https://arxiv.org/abs/1812.08685',
                                     'C. Sanderson and B.C. Lovell. Multi-Region Probabilistic Histograms for Robust and Scalable Identity Inference. LNCS 5558, 199-208, 2009.']},
        'split_method': 'Connected original/swapped identity components, shuffled with Python Random(42), 8 train / 4 validation / 4 test pairs. All clips, frames, and quality variants remain in the same group. First two validation pairs select models; last two calibrate.',
        'groups': plan, 'downloads': {'DeepfakeTIMIT': provenance}, 'pending_subjects': sorted(groups),
        'limitations': ['Only 32 people and 16 swap pairs in a controlled historical corpus; four test pairs cannot establish broad real-world accuracy.',
                        'Original JPEG and generated AVI compression differ; account for codec/domain shortcuts and evaluate outside this corpus.',
                        'Audio speech content is reused, with possible resampling/encoding differences; these data cannot train cloned-voice or audio/video-sync labels.',
                        'Do not mix paired frames across splits or tune a threshold using test labels.'],
        'entries': fake_entries(root, fake_path, clips, groups),
    }
    write_manifest(root, result)
    prepare_originals(root, real_record, clips, plan, result, workers)
    if len(result['entries']) != 960 or result['pending_subjects']:
        raise ValueError('Incomplete corpus')
    result['download_status'] = 'complete'
    result['clip_counts'] = dict(Counter(entry['split'] for entry in result['entries']))
    result['quality_counts'] = dict(Counter(entry['quality'] for entry in result['entries']))
    write_manifest(root, result)
    print(json.dumps({'manifest': str(root / 'manifest.json'), 'counts': result['counts'], 'clips': len(result['entries'])}), flush=True)


def validate_manifest(path):
    """Check completed corpus membership without fitting or querying a detector."""
    manifest = json.loads(Path(path).read_text(encoding='utf-8'))
    if manifest['download_status'] != 'complete':
        raise ValueError('Preparation is incomplete')
    groups = {group['pair_id']: group for group in manifest['groups']}
    seen_ids = set()
    clips = defaultdict(list)
    counts = Counter()
    for entry in manifest['entries']:
        if entry['id'] in seen_ids:
            raise ValueError('Duplicate clip identifier')
        seen_ids.add(entry['id'])
        group = groups[entry['pair_id']]
        if entry['split'] != group['split'] or entry['subject'] not in group['subjects']:
            raise ValueError('Clip escaped its identity component')
        if entry.get('swapped_identity') and entry['swapped_identity'] not in group['subjects']:
            raise ValueError('Swapped identity escaped its component')
        clips[entry['source_clip_id']].append(entry)
        counts[entry['split'] + '/' + entry['label']] += 1
    if len(clips) != 320 or len(seen_ids) != 960:
        raise ValueError('Unexpected corpus size')
    for source, entries in clips.items():
        if Counter(entry['label'] for entry in entries) != {'real': 1, 'fake': 2}:
            raise ValueError('Missing source or quality variant: ' + source)
        if {entry['quality'] for entry in entries} != {'original_jpeg', 'higher_quality', 'lower_quality'}:
            raise ValueError('Incorrect source quality association: ' + source)
        if len({entry['pair_id'] for entry in entries}) != 1 or len({entry['split'] for entry in entries}) != 1:
            raise ValueError('Related clips cross splits')
    return {'clips': len(seen_ids), 'matched_sources': len(clips), 'identity_pairs': len(groups), 'counts': dict(counts), 'source_group_overlap': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/deepfaketimit'))
    parser.add_argument('--inspect-only', action='store_true', help='Download/inspect the fake archive only; do not fetch original subjects.')
    parser.add_argument('--workers', type=int, default=1, choices=[1, 2, 3, 4], help='Bounded transfers against the official Zenodo mirror; the original personal server recommended one.')
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    fake_record = fetch_record(FAKE_RECORD, args.root / 'deepfake-zenodo-record.json')
    real_record = fetch_record(REAL_RECORD, args.root / 'original-zenodo-record.json')
    fake_item = next(item for item in fake_record['files'] if item['key'] == FAKE_ARCHIVE)
    if fake_item['checksum'] != 'md5:' + FAKE_MD5:
        raise ValueError('Published fake dataset checksum changed; inspect the new release first')
    fake_path = args.root / 'cache' / FAKE_ARCHIVE
    provenance = download(fake_item, fake_path)
    audit = inspect_fake(fake_path)
    audit['archive'] = provenance
    audit['license'] = fake_record['metadata']['license']
    audit['source'] = FAKE_RECORD
    (args.root / 'fake-archive-inventory.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps({'fake_files': len(audit['members']), 'document_names': list(audit['documents']), 'sample': audit['members'][:3]}, ensure_ascii=True), flush=True)
    if args.inspect_only:
        return
    prepare(args.root, fake_record, real_record, fake_path, provenance, audit, args.workers)


if __name__ == '__main__':
    main()
