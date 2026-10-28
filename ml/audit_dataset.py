"""Inspect an untrusted dataset ZIP without extracting or executing its contents."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import zipfile


def audit(archive, group, output):
    prefixes = [f'Data Set {n}/Data Set {n}/' for n in (range(1, 5) if group == 'all' else [int(group)])]
    entries = []
    seen = {}
    removed = Counter()
    counts = Counter()
    archive_hash = hashlib.sha256()
    with open(archive, 'rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            archive_hash.update(block)
    with zipfile.ZipFile(archive) as z:
        total_entries = len(z.infolist())
        # Give test priority, followed by validation. Remove exact duplicates across all splits.
        for split in ('test', 'validation', 'train'):
            members = sorted(i for i in z.namelist() if any(i.startswith(prefix + split + '/') for prefix in prefixes) and i.lower().endswith('.jpg'))
            for name in members:
                label = name.split('/')[-2]
                if label not in ('fake', 'real'):
                    continue
                info = z.getinfo(name)
                if info.file_size > 25 * 1024 * 1024:
                    raise ValueError('Unexpected oversized image in archive')
                digest = hashlib.sha256(z.read(name)).hexdigest()
                counts[f'{split}/{label}'] += 1
                if digest in seen:
                    prior = seen[digest]
                    if prior['label'] != label:
                        raise ValueError('Identical image has conflicting labels. Resolve the dataset before training.')
                    removed[f"{split}_duplicates_of_{prior['split']}"] += 1
                    continue
                entry = dict(member=name, label=label, split=split, sha256=digest)
                seen[digest] = entry
                entries.append(entry)
    if not entries:
        raise ValueError('Dataset group not found')
    result = dict(name=f'User supplied image archive — group {group}', version=archive_hash.hexdigest()[:12],
        archive_sha256=archive_hash.hexdigest(), archive_entries=total_entries, modality='image',
        group=group, raw_counts=dict(counts), counts=dict(Counter(e['split'] + '/' + e['label'] for e in entries)),
        exact_duplicates_removed=dict(removed), label_meaning='Folder labels only; dataset provenance and generator labels not provided.',
        limitations=['Exact-byte duplicate removal does not remove near-duplicate images or identities across splits.',
                     'All groups were globally deduplicated by exact content hash.' if group == 'all' else 'Other dataset groups are excluded to avoid potentially overlapping data.',
                     'License, source provenance, and person/identity-level split independence require confirmation.'], entries=entries)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2), encoding='utf8')
    # Small local samples support repeatable integration tests without unpacking the archive.
    with zipfile.ZipFile(archive) as z:
        for label in ('fake', 'real'):
            sample = Path(output).parent / (label + '.jpg')
            if not sample.exists():
                sample.write_bytes(z.read(next(e['member'] for e in entries if e['label'] == label)))
    print(json.dumps({k: v for k, v in result.items() if k != 'entries'}, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', required=True)
    parser.add_argument('--group', default='1', choices=['1','2','3','4','all'])
    parser.add_argument('--output', default='data/image-manifest.json')
    args = parser.parse_args()
    audit(args.archive, args.group, args.output)
