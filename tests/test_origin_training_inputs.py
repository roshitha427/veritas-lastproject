"""Portable origin training inputs are checked without fitting a model."""
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import pytest

from ml.train_diverse_origin import load_manifest, validate_records, validation_partition
from ml.train_personal_origin import merge_personal_records, personal_records


def entry(folder, number=0, split='train', label=0, **extra):
    path = folder / f'image-{number}.png'
    Image.new('RGB', (8, 8), (number, number+1, number+2)).save(path)
    return {'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'split': split, 'label': label, 'dataset': 'operator-dataset',
            'source_group': f'group-{number}', **extra}


def test_manifest_resolves_relative_images_and_records_deduplication(tmp_path):
    rows = [entry(tmp_path, index, split, label)
            for index, (split, label) in enumerate((('train', 0), ('train', 1),
                ('validation', 0), ('validation', 1), ('test', 0), ('test', 1)))]
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps({'entries': rows + [rows[0]]}), encoding='utf-8')
    normalized, provenance = load_manifest(path)
    assert len(normalized) == 6
    assert all(Path(row['path']).is_absolute() for row in normalized)
    assert provenance['input_entries'] == 7
    assert provenance['deduplicated_entries'] == 6
    assert provenance['manifest_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize('label', [True, False, -1, 2, 'real', 'ai_generated'])
def test_manifest_requires_explicit_binary_integer_labels(tmp_path, label):
    row = entry(tmp_path, label=label)
    with pytest.raises(ValueError, match='label must be integer'):
        validate_records([row], tmp_path, require_splits=False)


def test_manifest_checks_exact_bytes_and_archive_members(tmp_path):
    row = entry(tmp_path)
    archive = tmp_path / 'images.zip'
    with zipfile.ZipFile(archive, 'w') as handle:
        handle.write(tmp_path / row['path'], 'images/photo.png')
    archived = {**row, 'archive': archive.name, 'member': 'images/photo.png'}
    del archived['path']
    normalized = validate_records([archived], tmp_path, require_splits=False)
    assert normalized[0]['archive'] == str(archive.resolve())
    with pytest.raises(ValueError, match='differ from sha256'):
        validate_records([{**row, 'sha256': '0'*64}], tmp_path, require_splits=False)
    with pytest.raises(ValueError, match='archive member does not exist'):
        validate_records([{**archived, 'member': 'missing.png'}], tmp_path, require_splits=False)


def test_manifest_rejects_conflicting_labels_and_content_split_leakage(tmp_path):
    row = entry(tmp_path)
    with pytest.raises(ValueError, match='conflicting origin labels'):
        validate_records([row, {**row, 'label': 1}], tmp_path, require_splits=False)
    with pytest.raises(ValueError, match='image overlaps'):
        validate_records([row, {**row, 'split': 'test'}], tmp_path, require_splits=False)
    with pytest.raises(ValueError, match='source group overlaps'):
        validate_records([row, entry(tmp_path, 1, 'test', source_group=row['source_group'])],
                         tmp_path, require_splits=False)


def test_manifest_requires_all_three_independent_two_class_splits(tmp_path):
    with pytest.raises(ValueError, match='train split needs'):
        validate_records([entry(tmp_path)], tmp_path)
    with pytest.raises(ValueError, match='source_group is required'):
        validate_records([entry(tmp_path, source_group='')], tmp_path, require_splits=False)
    with pytest.raises(ValueError, match='provide either path or archive'):
        validate_records([entry(tmp_path, archive='other.zip', member='image.png')],
                         tmp_path, require_splits=False)


def test_validation_calibration_keeps_groups_together_and_both_classes():
    rows = [{'dataset': 'example', 'source_group': f'pair-{index}', 'label': label}
            for index in range(12) for label in (0, 1)]
    labels, calibration = validation_partition(rows)
    assert np.array_equal(calibration[::2], calibration[1::2])
    assert set(labels[calibration]) == {0, 1}
    assert set(labels[~calibration]) == {0, 1}
    with pytest.raises(ValueError, match='each need both classes'):
        validation_partition([rows[0]])


def test_personal_labels_come_only_from_explicit_real_and_ai_arguments(tmp_path):
    named_ai = tmp_path / 'AI-generated-name.png'
    named_real = tmp_path / 'real-name.png'
    Image.new('RGB', (8, 8), 'red').save(named_ai)
    Image.new('RGB', (8, 8), 'blue').save(named_real)
    rows = personal_records([named_ai, named_ai], [named_real])
    assert len(rows) == 2
    assert {Path(row['path']).name: row['label'] for row in rows} == {
        named_ai.name: 0, named_real.name: 1}
    assert {row['label_source'] for row in rows} == {'--real-image', '--ai-image'}
    with pytest.raises(ValueError, match='conflicting origin labels'):
        personal_records([named_ai], [named_ai])
    with pytest.raises(ValueError, match='at least one'):
        personal_records()


def test_personal_adaptation_cannot_reuse_parent_evaluation_images(tmp_path):
    parent = [entry(tmp_path, index, split, label)
              for index, (split, label) in enumerate((('train', 0), ('train', 1),
                  ('validation', 0), ('validation', 1), ('test', 0), ('test', 1)))]
    parent = validate_records(parent, tmp_path)
    personal = [{**parent[-1], 'split': 'train', 'dataset': 'user_labelled_ai'}]
    with pytest.raises(ValueError, match='image overlaps'):
        merge_personal_records(parent, personal)
