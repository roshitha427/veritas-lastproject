"""Curated, local dataset metadata; never imports media into a model or evidence index."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from urllib.parse import urlsplit


REGISTRY_OWNER = 'audited-local-manifests-v1'


def _load(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        return None
    # Known audit files only. Do not discover or execute downloaded project code.
    if path.stat().st_size > 80 * 1024 * 1024:
        raise ValueError('Dataset manifest exceeds the metadata import limit: ' + relative)
    raw = path.read_bytes()
    value = json.loads(raw.decode('utf-8-sig'))
    if not isinstance(value, dict):
        raise ValueError('Dataset manifest must contain an object: ' + relative)
    return value, hashlib.sha256(raw).hexdigest()


def normalize_counts(value):
    """Keep split totals once; never sum a total together with its class breakdown."""
    output = {}
    def visit(item, prefix=''):
        if type(item) is int and item >= 0:
            output[prefix] = item
        elif isinstance(item, dict):
            total = item.get('total', item.get('answers'))
            if prefix and type(total) is int and total >= 0:
                output[prefix] = total
            else:
                for key, child in item.items():
                    visit(child, '/'.join(part for part in (prefix, str(key)) if part))
    visit(value)
    return output


def _urls(value):
    """Retain original public HTTPS sources, excluding archive paths and private filenames."""
    found = []
    def visit(item):
        if isinstance(item, dict):
            for key, child in item.items():
                if key not in {'path', 'archive', 'filepath', 'member', 'caption', 'entries', 'records'}:
                    visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)
        elif isinstance(item, str):
            for match in re.findall(r'https://[^\s<>"\']+', item):
                url = match.rstrip('.,;)')
                parsed = urlsplit(url)
                public_url = parsed._replace(query='', fragment='').geturl()
                if parsed.hostname and not parsed.username and not parsed.password and public_url not in found:
                    found.append(public_url)
    visit(value)
    return found[:12]


def _license(value):
    if isinstance(value, str):
        return value[:1000]
    if isinstance(value, dict):
        labels = []
        def visit(item):
            if isinstance(item, dict):
                for key, child in item.items():
                    if key in {'id', 'note'} and isinstance(child, str):
                        labels.append(child)
                    elif isinstance(child, dict):
                        visit(child)
                    elif isinstance(child, str) and ('CC ' in child or 'CC-' in child):
                        labels.append(child)
        visit(value)
        return ' '.join(dict.fromkeys(labels))[:1000] or 'Terms recorded in the source manifest.'
    return 'License not recorded in the local audit; upstream terms require review.'


def _entry_counts(rows):
    return dict(Counter(str(row.get('split') or 'unsplit') for row in rows))


def _missing_assets(root, rows):
    # Check referenced archives/files exist; do not re-decode media or expose its names.
    references = {str(row.get('archive') or row.get('path') or row.get('filepath') or '') for row in rows}
    references.discard('')
    missing = 0
    for reference in references:
        path = Path(reference.replace('\\', '/'))
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            missing += 1
    return missing


def _record(key, name, modality, manifest_path, loaded, *, task, role,
            availability='available', counts=None, note='', composition='standalone', overlaps=()):
    manifest, fingerprint = loaded
    version = str(manifest.get('version') or manifest.get('archive_sha256') or fingerprint)[:80]
    if len(version) > 20 and re.fullmatch(r'[a-f0-9]{64}', version):
        version = version[:12]
    record = dict(name=name, version=version, modality=modality,
                  counts=normalize_counts(manifest.get('counts', {}) if counts is None else counts),
                  registry_owner=REGISTRY_OWNER, registry_key=key, registry_schema_version=1,
                  availability=availability, task=task, role=role, status_note=note,
                  composition=composition, overlaps_with=list(overlaps),
                  source_manifest=manifest_path,
                  provenance={'manifest_sha256': fingerprint, 'source_urls': _urls({
                      key: manifest.get(key) for key in ('source', 'source_url', 'provenance', 'original_source')})},
                  license=_license(manifest.get('license')),
                  label_meaning=str(manifest.get('label_meaning') or manifest.get('scope') or '')[:1200],
                  limitations=[str(item)[:1200] for item in manifest.get('limitations', []) if isinstance(item, str)])
    if re.fullmatch(r'[a-f0-9]{64}', str(manifest.get('archive_sha256') or '')):
        record['archive_sha256'] = manifest['archive_sha256']
    return record


def discover_local_datasets(root):
    """Inventory audited imports and explicitly pending sources without training or networking."""
    root = Path(root).resolve()
    records, warnings = [], []
    def add(key, relative, name, modality, task, role, **kwargs):
        try:
            loaded = _load(root, relative)
            if loaded is None:
                return None
            record = _record(key, name, modality, relative, loaded, task=task, role=role, **kwargs)
            rows = loaded[0].get('entries') or loaded[0].get('records') or []
            missing = _missing_assets(root, rows)
            if missing:
                record['availability'] = 'missing_media'
                record['status_note'] = f'{missing} referenced media file(s) or archive(s) are unavailable locally. Counts describe the saved audit.'
            records.append(record)
            return loaded[0], record
        except (OSError, ValueError, TypeError) as error:
            warnings.append({'manifest': relative, 'reason': type(error).__name__})
            return None

    add('isot', 'data/text/manifest.json', 'ISOT fake/real news (Kaggle mirror)', 'text',
        'news_text_classification', 'training_and_evaluation',
        note='Prepared historical English news; dataset accuracy is separate from current claim verification.')
    add('supplied-image-group1', 'data/image-manifest.json', 'User supplied image archive â€” group 1', 'image',
        'visual_classification', 'training_and_evaluation', composition='subset',
        overlaps=('User supplied image archive â€” group all',),
        note='Original group-1 subset. Its entries overlap the all-groups manifest.')
    add('supplied-image-all', 'data/image-manifest-all.json', 'User supplied image archive â€” group all', 'image',
        'visual_classification', 'experimental_training', composition='overlapping_combination',
        overlaps=('User supplied image archive â€” group 1',),
        note='Combined view of the supplied archive; do not add its counts to the group-1 subset.')
    add('cifake', 'data/cifake-audited-20261021/manifest.json', 'CIFAKE â€” audited supplied archive', 'image',
        'ai_image_origin', 'experimental_training', note='Complete supplied archive audit after exact-image deduplication.')
    add('casia2', 'data/casia-audited-20261021/manifest.json', 'CASIA2 â€” authentic/tampered images', 'image',
        'image_manipulation', 'experimental_training', note='Tampering labels describe editing, not AI generation. Masks remain annotations.')
    add('genimage-sample', 'data/genimage-sample-20261021/manifest.json', 'GenImage â€” bounded sample', 'image',
        'ai_image_origin', 'experimental_training', availability='sampled',
        note='Bounded sample, not the complete collection. BigGAN is held out from training and validation.')
    result = add('defactify-sample', 'data/origin-training/manifest.json', 'Defactify â€” bounded image-origin sample', 'image',
                 'ai_image_origin', 'experimental_training', availability='sampled',
                 note='Original sampled images only; training augmentations are not independent examples.')
    if result:
        manifest, record = result
        record['counts'] = _entry_counts(manifest.get('records', []))
        record['label_meaning'] = 'Real versus generated images, preserving the supplied generator groups.'
    add('immfnd-english', 'data/immfnd-english-claims/manifest.json', 'IMMFND demo: English claim-only classification', 'text',
        'news_claim_classification', 'experimental_training',
        note='Prepared English claims from the multilingual demo; associated images are not given manipulation labels.')
    hc3 = add('hc3-selected', 'data/hc3/source-manifest.json', 'HC3 â€” selected ELI5/Wikipedia domains', 'text',
              'ai_written_text', 'training_and_evaluation', availability='sampled',
              note='Selected domains only. Human/AI writing labels are separate from news truth.')
    if hc3:
        manifest, record = hc3
        card = _load(root, 'models/ai-text/model-card.json')
        record['version'] = str(manifest.get('revision') or record['version'])[:12]
        record['provenance']['source_urls'] = _urls(manifest.get('source_url'))
        if card:
            record['counts'] = normalize_counts(card[0].get('training_counts', {}).get('prepared_counts', {}))
        if not all((root / 'data/hc3' / file).is_file() for file in ('train.csv', 'test.csv')):
            record['availability'] = 'missing_media'
    liar = add('liar-evaluation', 'data/text-liar/audit.json', 'LIAR â€” external six-label claim diagnostic', 'text',
               'news_claim_classification', 'evaluation_only', note='Six labels retained; strict binary subsets are separate diagnostic views.')
    if liar:
        manifest, record = liar
        audit = manifest.get('liar', {})
        record['version'] = str(audit.get('archive_sha256') or record['version'])[:12]
        record['counts'] = {split: info['all_six'] for split, info in audit.get('prepared_disjoint_counts', {}).items()}
        record['license'] = _license(audit.get('license'))
        record['provenance']['source_urls'] = _urls(audit.get('source'))
        record['label_meaning'] = 'Original six factual-claim labels; partly true categories are not silently mapped to binary.'
        record['limitations'] = [str(item) for item in audit.get('prohibited_features', [])]
    add('deepfaketimit', 'data/deepfaketimit/manifest.json', 'DeepfakeTIMIT with matched VidTIMIT originals', 'video',
        'face_swap_consistency', 'training_and_evaluation', overlaps=('DeepfakeTIMIT plus paired SDFVD',),
        note='Matched original and face-swapped clips. Also included in the combined SDFVD manifest.')
    combined = add('deepfaketimit-sdfvd-combined', 'data/sdfvd-20261022/manifest.json', 'DeepfakeTIMIT plus paired SDFVD', 'video',
                   'video_synthetic_or_manipulated', 'experimental_training', composition='overlapping_combination',
                   overlaps=('DeepfakeTIMIT with matched VidTIMIT originals', 'SDFVD â€” paired supplied clips'),
                   note='Combined view; contains the DeepfakeTIMIT and SDFVD records already listed separately.')
    if combined:
        manifest, record = combined
        record['counts'] = _entry_counts(manifest.get('entries', []))
        sdfvd = [row for row in manifest.get('entries', []) if str(row.get('dataset', '')).casefold() == 'sdfvd']
        if sdfvd:
            solo = _record('sdfvd', 'SDFVD â€” paired supplied clips', 'video', record['source_manifest'],
                           (manifest, record['provenance']['manifest_sha256']), task='face_swap_consistency',
                           role='experimental_training', counts=_entry_counts(sdfvd), composition='subset',
                           overlaps=('DeepfakeTIMIT plus paired SDFVD',),
                           note=f'SDFVD-only entries from the combined audit: {len(sdfvd)} clips. The combined view also includes the older base clips.')
            solo['version'] = str(manifest.get('provenance', {}).get('outer_archive_sha256') or solo['version'])[:12]
            solo['availability'] = 'missing_media' if _missing_assets(root, sdfvd) else 'available'
            solo['license'] = _license(manifest.get('license', {}).get('sdfvd'))
            solo['provenance']['source_urls'] = _urls(manifest.get('provenance', {}).get('sdfvd_source'))
            records.append(solo)
        archives = manifest.get('provenance', {}).get('csv_archives', [])
        if archives:
            metadata = _record('supplied-score-metadata', 'Supplied multimodal detector-score metadata', 'multimodal',
                               record['source_manifest'], (manifest, record['provenance']['manifest_sha256']),
                               task='metadata_audit', role='metadata_only', availability='metadata_only',
                               counts={'unique_rows': archives[0].get('rows', 0), 'media_files': 0},
                               note='Duplicate metadata archives counted once. No source media; precomputed detector scores are excluded from training.')
            metadata['version'] = str(archives[0].get('csv_sha256') or metadata['version'])[:12]
            metadata['label_meaning'] = 'Stored metadata and detector scores, not independent image/video training labels.'
            metadata['license'] = 'No license recorded for the supplied metadata CSV archives.'
            metadata['provenance']['source_urls'] = []
            metadata['limitations'] = ['Source media and media URLs are absent.',
                                       'Precomputed detector scores and post-outcome fields are excluded from model training.']
            records.append(metadata)
    private = add('user-confirmed-video', 'data/video-confirmed-20261022/manifest.json', 'User-confirmed AI video training references', 'video',
                  'video_synthetic_or_manipulated', 'training_references',
                  note='Private reference clips used by an inactive experiment; training fit is not independent evaluation.')
    if private:
        manifest, record = private
        record['counts'] = {'original_clips': len(manifest.get('entries', []))}
        record['label_meaning'] = 'User-confirmed generated clips; generator identities were not supplied.'
    fvc = add('fvc-metadata', 'data/video-sources-20261022/fvc/audit.json', 'Fake Video Corpus 2018 â€” verification metadata', 'video',
              'video_factual_verification', 'metadata_only', availability='metadata_only',
              note='Factual/debunking labels and links; the supplied archive contains no raw video files.')
    if fvc:
        manifest, record = fvc
        files = manifest.get('files', {})
        record['counts'] = {'main_records': files.get('FVC.csv', {}).get('rows', 0),
                            'related_link_records': files.get('FVC_dup.csv', {}).get('rows', 0),
                            'query_records': files.get('FVC_text_queries.csv', {}).get('rows', 0), 'video_files': 0}
        record['label_meaning'] = manifest.get('label_policy', '')

    for key, folder, name in [('government-snapshots', 'models/current-affairs', 'Government/NIC source snapshots'),
                              ('factcheck-articles', 'models/dataset-articles-v2', 'Supplied original fact-check articles')]:
        indexed = add(key, folder + '/import-audit.json', name, 'text', 'source_evidence_retrieval', 'retrieval_only',
                      note='Searchable source passages, not classifier training or a live guarantee of truth.')
        if indexed:
            _, record = indexed
            path = (root / folder / 'evidence.sqlite').resolve()
            if path.is_file():
                with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as connection:
                    record['counts'] = {'indexed_documents': connection.execute('SELECT count(*) FROM documents').fetchone()[0]}
            else:
                record['availability'] = 'missing_media'
                record['counts'] = {}
    gist = add('gist-source-pages', 'data/gist-source-import/import-audit.json', 'GIST official website source pages', 'text',
               'source_evidence_retrieval', 'retrieval_only', composition='subset',
               overlaps=('Supplied original fact-check articles',),
               note='Official institution pages included in the source index; not a training set or universal fact-check.')
    if gist:
        manifest, record = gist
        record['counts'] = {'source_pages': manifest.get('page_count', 0)}
        record['provenance']['source_urls'] = _urls(manifest.get('urls', []))

    # Additional imports have a fixed audited schema and retain evaluation-only boundaries.
    imported = set()
    for path in sorted((root / 'data').glob('*/videos/manifest.json')):
        relative = path.relative_to(root).as_posix()
        loaded = _load(root, relative)
        if not loaded or loaded[0].get('purpose') != 'video_synthetic_or_manipulated':
            continue
        manifest, fingerprint = loaded
        rows = list(manifest.get('entries', [])) + list(manifest.get('evaluation_entries', []))
        for family in ('DFDC', 'Celeb-DF-v2', 'DF26'):
            selected = [row for row in rows if row.get('dataset') == family]
            if not selected:
                continue
            imported.add(family)
            record = _record('additional-' + family.casefold(), family + ' â€” audited local import', 'video', relative,
                             loaded, task='video_manipulation', role='training_pool' if family == 'DFDC' else 'evaluation_only',
                             counts=_entry_counts(selected), availability='missing_media' if _missing_assets(root, selected) else 'available',
                             note='An imported subset; preparation does not train or activate a model.')
            record['version'] = fingerprint[:12]
            records.append(record)
    for family, title, source, role in [
            ('DFDC', 'DFDC â€” raw media awaiting import', 'https://www.kaggle.com/competitions/deepfake-detection-challenge/data', 'training_pool'),
            ('Celeb-DF-v2', 'Celeb-DF v2 â€” raw media awaiting import', 'https://github.com/yuezunli/celeb-deepfakeforensics', 'evaluation_only'),
            ('DF26', 'DF26 â€” authorized raw media awaiting import', 'https://huggingface.co/datasets/DF26/DF26', 'evaluation_only')]:
        if family in imported:
            continue
        record = _record('pending-' + family.casefold(), title, 'video', 'ml/prepare_video_sources.py',
                         ({'version': 'awaiting-local-media', 'source_url': source}, ''), task='video_manipulation',
                         role=role, availability='pending_media', counts={},
                         note='No audited local raw-media import was found. A source-code/demo repository is not the labelled video dataset.')
        record['limitations'] = (['Authorized access and upstream dataset terms are required.',
                                  'Evaluation-only data must not be used for fitting, calibration, or model selection.']
                                 if role == 'evaluation_only' else ['A labelled training chunk and matching metadata are required.'])
        records.append(record)

    for path in sorted((root / 'data' / 'benchmarks').glob('*/manifest.json')):
        relative = path.relative_to(root).as_posix()
        loaded = _load(root, relative)
        if not loaded or loaded[0].get('evaluation_only') is not True:
            continue
        manifest, _ = loaded
        record = _record('benchmark-' + path.parent.name, str(manifest.get('name') or path.parent.name), 'text', relative,
                         loaded, task='claim_verification_benchmark', role='evaluation_only',
                         counts=manifest.get('counts') or manifest.get('audit', {}).get('partition_counts', {}),
                         note='Independent evaluation/calibration partitions; excluded from model training and runtime evidence indexes.')
        record['provenance']['source_urls'] = _urls({key: manifest.get(key) for key in ('source_readme_url', 'paper_url')})
        records.append(record)
    return records, warnings


def reconcile_dataset_registry(store, records, *, apply=False):
    """Preview or install metadata; preserve custom rows, IDs, creation times and version history."""
    stats = dict(created=0, updated=0, unchanged=0, preserved=0)
    for record in records:
        previous = store.one('datasets', {'name': record['name'], 'version': record['version']})
        if previous and previous.get('registry_owner') != REGISTRY_OWNER:
            # Adopt the original manifest imports only when the recorded archive identity agrees.
            same_archive = bool(record.get('archive_sha256') and previous.get('archive_sha256') == record['archive_sha256'])
            if not same_archive:
                stats['preserved'] += 1
                continue
        merged = dict(previous or {})
        merged.update(record)
        if previous and ('limitations' in previous or 'limitations' in record):
            merged['limitations'] = list(dict.fromkeys(previous.get('limitations', []) + record.get('limitations', [])))
        if previous and merged == previous:
            stats['unchanged'] += 1
            continue
        stats['updated' if previous else 'created'] += 1
        if apply:
            store.save('datasets', merged)
    return stats
