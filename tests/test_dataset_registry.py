import json
from pathlib import Path
import sqlite3

from backend.dataset_registry import (REGISTRY_OWNER, discover_local_datasets,
                                      normalize_counts, reconcile_dataset_registry)
from backend.store import Store


def manifest(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')
    return path


def test_split_totals_do_not_double_count_class_breakdowns_or_questions():
    assert normalize_counts({'train': {'total': 10, 'real': 4, 'fake': 6},
                             'test': {'questions': 3, 'answers': 6, 'labels': {'human': 3, 'ai': 3}},
                             'validation/real': 2, 'negative': -1, 'boolean': True}) == {
        'train': 10, 'test': 6, 'validation/real': 2}


def test_sampled_and_metadata_only_imports_stay_distinct_and_paths_stay_private(tmp_path):
    image = tmp_path / 'private-original-image.jpg'
    image.write_bytes(b'exists')
    manifest(tmp_path, 'data/origin-training/manifest.json', {
        'dataset': 'Defactify', 'source_url': 'https://example.invalid/dataset?token=private-access-token',
        'provenance': {'url': 'https://private-user:private-secret@example.invalid/source'},
        'records': [{'path': str(image), 'split': 'train', 'caption': 'private caption'},
                    {'path': str(image), 'split': 'test'}]})
    manifest(tmp_path, 'data/video-sources-20261022/fvc/audit.json', {
        'name': 'FVC', 'archive_sha256': 'f' * 64, 'embedded_video_count': 0,
        'files': {'FVC.csv': {'rows': 380}, 'FVC_dup.csv': {'rows': 5199}, 'FVC_text_queries.csv': {'rows': 380}}})
    records, warnings = discover_local_datasets(tmp_path)
    by_key = {row['registry_key']: row for row in records}
    assert not warnings
    assert by_key['defactify-sample']['availability'] == 'sampled'
    assert by_key['defactify-sample']['counts'] == {'train': 1, 'test': 1}
    assert by_key['fvc-metadata']['availability'] == 'metadata_only'
    assert by_key['fvc-metadata']['counts']['video_files'] == 0
    assert by_key['fvc-metadata']['task'] == 'video_factual_verification'
    assert 'private-original-image' not in json.dumps(records)
    assert 'private caption' not in json.dumps(records)
    assert 'private-access-token' not in json.dumps(records)
    assert 'private-secret' not in json.dumps(records)
    assert by_key['defactify-sample']['provenance']['source_urls'] == ['https://example.invalid/dataset']
    assert str(tmp_path) not in json.dumps(records)
    assert {row['registry_key'] for row in records if row['availability'] == 'pending_media'} == {
        'pending-dfdc', 'pending-celeb-df-v2', 'pending-df26'}


def test_missing_referenced_archives_are_not_claimed_available(tmp_path):
    manifest(tmp_path, 'data/cifake-audited-20261021/manifest.json', {
        'counts': {'train/real': 5},
        'entries': [{'archive': str(tmp_path / 'missing-private-archive.zip'), 'split': 'train'}]})
    records, _ = discover_local_datasets(tmp_path)
    record = next(row for row in records if row['registry_key'] == 'cifake')
    assert record['availability'] == 'missing_media'
    assert record['counts'] == {'train/real': 5}  # Saved audit counts, not invented live counts.
    assert 'missing-private-archive' not in json.dumps(record)


def test_combined_video_manifest_exposes_subsets_without_counting_them_as_new_media(tmp_path):
    media = tmp_path / 'media.mp4'
    media.write_bytes(b'not decoded by metadata import')
    manifest(tmp_path, 'data/sdfvd-20261022/manifest.json', {
        'version': 'combined-1', 'entries': [
            {'dataset': 'DeepfakeTIMIT', 'split': 'train', 'filepath': str(media)},
            {'dataset': 'SDFVD', 'split': 'train', 'filepath': str(media)},
            {'dataset': 'SDFVD', 'split': 'test', 'filepath': str(media)}],
        'provenance': {'csv_archives': [{'rows': 1200, 'csv_sha256': 'c' * 64,
                                       'path': 'private-original.zip'},
                                      {'rows': 1200, 'csv_sha256': 'c' * 64,
                                       'duplicate_of': 'private-original.zip'}]}})
    records, _ = discover_local_datasets(tmp_path)
    by_key = {row['registry_key']: row for row in records}
    assert by_key['deepfaketimit-sdfvd-combined']['composition'] == 'overlapping_combination'
    assert by_key['deepfaketimit-sdfvd-combined']['counts'] == {'train': 2, 'test': 1}
    assert by_key['sdfvd']['counts'] == {'train': 1, 'test': 1}
    assert by_key['sdfvd']['composition'] == 'subset'
    assert by_key['supplied-score-metadata']['counts'] == {'unique_rows': 1200, 'media_files': 0}
    assert 'private-original.zip' not in json.dumps(records)


def test_retrieval_count_uses_live_index_and_benchmark_remains_evaluation_only(tmp_path):
    manifest(tmp_path, 'models/dataset-articles-v2/import-audit.json', {'documents': 0})
    index = tmp_path / 'models/dataset-articles-v2/evidence.sqlite'
    with sqlite3.connect(index) as connection:
        connection.execute('CREATE TABLE documents (id INTEGER)')
        connection.executemany('INSERT INTO documents VALUES (?)', [(1,), (2,)])
    manifest(tmp_path, 'data/benchmarks/averitec/manifest.json', {
        'name': 'Independent diagnostic', 'version': 'pinned-1', 'evaluation_only': True,
        'counts': {'calibration': 30, 'test': 120}, 'license': 'CC-BY-NC-4.0'})
    records, _ = discover_local_datasets(tmp_path)
    by_key = {row['registry_key']: row for row in records}
    assert by_key['factcheck-articles']['counts'] == {'indexed_documents': 2}
    assert by_key['factcheck-articles']['role'] == 'retrieval_only'
    assert by_key['benchmark-averitec']['role'] == 'evaluation_only'
    assert by_key['benchmark-averitec']['counts'] == {'calibration': 30, 'test': 120}
    assert 'excluded from model training' in by_key['benchmark-averitec']['status_note']


def test_preview_and_sync_preserve_user_records_and_are_idempotent(app):
    store = app.extensions['store']
    try:
        custom = store.save('datasets', {'name': 'Custom', 'version': '1', 'modality': 'text', 'counts': {'train': 4}, 'notes': 'Keep me'})
        legacy = store.save('datasets', {'name': 'Legacy', 'version': '1', 'modality': 'image',
                                        'archive_sha256': 'a' * 64, 'limitations': ['Operator note']})
        records = [dict(name='Imported', version='1', modality='text', counts={'test': 10}, registry_owner=REGISTRY_OWNER),
                   dict(name='Custom', version='1', modality='text', counts={'train': 99}, registry_owner=REGISTRY_OWNER),
                   dict(name='Legacy', version='1', modality='image', counts={'test': 5}, archive_sha256='a' * 64,
                        registry_owner=REGISTRY_OWNER, limitations=['Audit note'])]
        expected = {'created': 1, 'updated': 1, 'unchanged': 0, 'preserved': 1}
        assert reconcile_dataset_registry(store, records) == expected
        assert store.count('datasets') == 2
        assert reconcile_dataset_registry(store, records, apply=True) == expected
        assert store.one('datasets', {'name': 'Custom'}) == custom
        adopted = store.one('datasets', {'name': 'Legacy'})
        assert adopted['id'] == legacy['id'] and adopted['created_at'] == legacy['created_at']
        assert adopted['limitations'] == ['Operator note', 'Audit note']
        assert reconcile_dataset_registry(store, records, apply=True) == {
            'created': 0, 'updated': 0, 'unchanged': 2, 'preserved': 1}
    finally:
        store.close()

