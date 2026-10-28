import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest

from ml.prepare_video_sources import DatasetFiles, assign_splits, audit_fvc, prepare
from ml.train_temporal import clip_frames, main as train_temporal
from scripts.evaluate_video_sources import score_report


def test_fvc_factual_labels_never_become_ai_training_labels(tmp_path):
    archive = tmp_path / 'corpus.zip'
    with ZipFile(archive, 'w') as z:
        for name in ('FVC.csv', 'FVC_dup.csv', 'FVC_text_queries.csv'):
            z.writestr(name, 'cascade_id,video_url,label\nf0,https://example.org/video,fake\n')
        z.writestr('download.py', 'raise RuntimeError("must never run")')
    result = audit_fvc(archive, tmp_path / 'prepared')
    assert result['usable_for_visual_origin_training'] is False
    assert result['embedded_video_count'] == 0
    assert result['files']['FVC.csv']['labels'] == {'fake': 1}
    assert not (tmp_path / 'prepared/download.py').exists()


def test_archive_parent_paths_are_rejected(tmp_path):
    archive = tmp_path / 'unsafe.zip'
    with ZipFile(archive, 'w') as z:
        z.writestr('../metadata.json', '{}')
    with pytest.raises(ValueError, match='Unsafe archive'):
        DatasetFiles(archive)


def test_dfdc_original_and_derivatives_remain_together(tmp_path):
    archive = tmp_path / 'chunk.zip'
    metadata = {}
    with ZipFile(archive, 'w') as z:
        for index in range(30):
            original, fake = f'real{index}.mp4', f'fake{index}.mp4'
            metadata[original] = {'label': 'REAL', 'split': 'train'}
            metadata[fake] = {'label': 'FAKE', 'original': original, 'split': 'train'}
            z.writestr(original, ('real media ' + str(index)).encode())
            z.writestr(fake, ('fake media ' + str(index)).encode())
        z.writestr('metadata.json', json.dumps(metadata))
    result = prepare([('dfdc', archive)], tmp_path / 'prepared')
    entries = result['entries']
    assert len(entries) == 60
    assert {e['split'] for e in entries} == {'train', 'validation', 'test'}
    for index in range(30):
        related = [e for e in entries if e['original'] == f'real{index}.mp4']
        assert len(related) == 2
        assert len({e['split'] for e in related}) == 1
    assert all(e['kind'] == 'zip_video' for e in entries)


def test_df26_stays_external_even_when_exact_duplicate_is_in_training_pool():
    base = {'sha256': 'same-video', 'label': 'real'}
    rows = [dict(base, dataset='DFDC', role='training_pool', pair_id='dfdc/original'),
            dict(base, dataset='DF26', role='evaluation', pair_id='df26/clip', training_prohibited=True)]
    result = assign_splits(rows)
    assert len(result) == 1
    assert result[0]['split'] == 'external_evaluation'


def test_df26_import_preserves_real_fake_pair_for_external_evaluation(tmp_path):
    archive = tmp_path / 'df26.zip'
    with ZipFile(archive, 'w') as z:
        z.writestr('metadata/generators.csv', 'generator,availability\nVeo_3.1,closed\n')
        z.writestr('real/Studio_Interview/clip.mp4', b'real clip')
        z.writestr('fake/Studio_Interview/Veo_3.1/clip.mp4', b'generated clip')
    result = prepare([('df26', archive)], tmp_path / 'prepared')
    assert not result['entries']
    assert len(result['evaluation_entries']) == 2
    assert all(e['training_prohibited'] for e in result['evaluation_entries'])
    assert len({e['pair_id'] for e in result['evaluation_entries']}) == 1


def test_training_rejects_df26_even_if_manifest_calls_it_training(tmp_path):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'entries': [{'dataset': 'DF26', 'split': 'train'}]}))
    args = SimpleNamespace(threads=1, manifest=str(manifest), output=str(tmp_path / 'model'), cache_available=False)
    with pytest.raises(ValueError, match='Evaluation-only'):
        train_temporal(args)


def test_celeb_source_repository_demo_cannot_be_used_as_labelled_dataset(tmp_path):
    archive = tmp_path / 'repo.zip'
    with ZipFile(archive, 'w') as z:
        z.writestr('src/demo.mp4', b'demo')
    with pytest.raises(ValueError, match='source-code/demo'):
        prepare([('celeb', archive)], tmp_path / 'prepared')


def test_zip_video_decoder_reads_only_media_and_matches_file_sampling(tmp_path):
    import cv2
    path = tmp_path / 'video.mp4'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (96, 96))
    assert writer.isOpened()
    for value in range(20):
        writer.write(np.full((96, 96, 3), value * 10, dtype=np.uint8))
    writer.release()
    archive = tmp_path / 'media.zip'
    with ZipFile(archive, 'w') as z:
        z.write(path, 'videos/clip.mp4')
        z.writestr('download.py', 'raise RuntimeError("do not execute")')
    zipped = clip_frames({'kind': 'zip_video', 'archive': str(archive), 'member': 'videos/clip.mp4', 'id': 'clip'})
    direct = clip_frames({'kind': 'video', 'filepath': str(path), 'id': 'clip'})
    assert len(zipped) == len(direct) == 12
    for a, b in zip(zipped, direct):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_external_report_measures_auc_eer_without_tuning_threshold():
    report = score_report([{'label': 'real', 'score': .1}, {'label': 'fake', 'score': .9}])
    assert report['roc_auc'] == 1
    assert report['eer'] == pytest.approx(0)
    assert 'fixed 0.5' in report['threshold_policy']
    constant = score_report([{'label': 'real', 'score': .5}, {'label': 'fake', 'score': .5}])
    assert constant['eer'] == pytest.approx(.5)


def test_external_report_requires_both_classes():
    assert score_report([{'label': 'fake', 'score': .9}])['status'] == 'both_classes_required'
