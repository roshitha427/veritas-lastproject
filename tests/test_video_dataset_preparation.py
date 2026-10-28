"""Grouping and inert-archive handling for the optional research corpus."""

import io
import itertools
import zipfile

from PIL import Image
import pytest

from ml.prepare_deepfaketimit import real_entries, safe_member, split_plan


def synthetic_inventory():
    subjects = ['f' + ''.join(letters) + '0' for letters in itertools.islice(itertools.product('abcd', repeat=3), 32)]
    members = []
    for index in range(0, len(subjects), 2):
        a, b = subjects[index:index + 2]
        for original, swapped in ((a, b), (b, a)):
            for quality in ('higher_quality', 'lower_quality'):
                for sentence in range(10):
                    members.append({'path': f'DeepfakeTIMIT/{quality}/{original}/sa{sentence}-video-{swapped}.avi', 'bytes': 100})
    return {'members': members}


def test_identity_components_keep_sources_and_quality_variants_in_one_split():
    inventory = synthetic_inventory()
    clips, plan = split_plan(inventory)
    assert len(clips) == 640
    assert [len([group for group in plan if group['split'] == name]) for name in ('train', 'validation', 'test')] == [8, 4, 4]
    ownership = {}
    for group in plan:
        for subject in group['subjects']:
            assert subject not in ownership
            ownership[subject] = group
    for clip in clips:
        assert ownership[clip['subject']] == ownership[clip['swapped_identity']]
    validation_roles = [group['validation_role'] for group in plan if group['split'] == 'validation']
    assert validation_roles.count('selection') == validation_roles.count('calibration') == 2
    assert split_plan(inventory)[1] == plan


def test_unexpected_connected_identity_group_is_rejected():
    inventory = synthetic_inventory()
    inventory['members'][0]['path'] = 'DeepfakeTIMIT/higher_quality/faaa0/sa0-video-faac0.avi'
    with pytest.raises(ValueError, match='identity pairs'):
        split_plan(inventory)


def test_native_extensionless_jpegs_are_sorted_and_head_sequences_excluded(tmp_path):
    buffer = io.BytesIO()
    Image.new('RGB', (12, 12), 'white').save(buffer, format='JPEG')
    path = tmp_path / 'fadg0.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        for frame in range(15, -1, -1):
            archive.writestr(f'fadg0/video/sa1/{frame:03d}', buffer.getvalue())
        archive.writestr('fadg0/video/head/000', buffer.getvalue())
    group = {'pair_id': 'fadg0__fram1', 'split': 'test', 'validation_role': None}
    entries = real_entries(path, 'fadg0', {'sa1'}, group)
    assert len(entries) == 1
    assert len(entries[0]['members']) == 16
    assert entries[0]['members'][0].endswith('/000')
    assert entries[0]['members'][-1].endswith('/015')
    assert entries[0]['source_clip_id'] == 'fadg0/sa1'
    assert entries[0]['kind'] == 'zip_jpeg_sequence'
    assert entries[0]['split'] == 'test'


@pytest.mark.parametrize('name', ['../outside', '/absolute', 'C:/outside', 'folder\\..\\outside', '//server/share', 'safe:alternate'])
def test_archive_member_paths_reject_traversal_and_windows_drives(name):
    assert not safe_member(name)
