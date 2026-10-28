import json
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest

from ml.prepare_sdfvd import csv_audit
from ml.train_temporal import selection_score, source_balanced_weights


def test_source_balancing_gives_small_corpus_and_both_classes_equal_weight():
    entries = ([{'dataset': 'large', 'label': 'real'}] * 10
               + [{'dataset': 'large', 'label': 'fake'}] * 30
               + [{'dataset': 'small', 'label': 'real'}] * 2
               + [{'dataset': 'small', 'label': 'fake'}] * 2)
    weights = source_balanced_weights(entries)
    cells = {(row['dataset'], row['label']) for row in entries}
    totals = [sum(w for w, row in zip(weights, entries) if (row['dataset'], row['label']) == cell)
              for cell in cells]
    np.testing.assert_allclose(totals, [len(entries) / 4] * 4)


def test_validation_macro_prevents_large_corpus_hiding_small_corpus_failure():
    labels = np.array([0, 1] * 50 + [0, 1])
    predictions = np.array([0, 1] * 50 + [1, 0])
    rows = [{'dataset': 'large'}] * 100 + [{'dataset': 'small'}] * 2
    assert selection_score(labels, predictions, rows, macro_datasets=True) == pytest.approx(.5)
    assert selection_score(labels, predictions, rows) > .98


def test_duplicate_csv_archive_is_counted_once_and_excluded_from_visual_training(tmp_path):
    paths = []
    for index in (3, 5):
        path = tmp_path / f'archive{index}.zip'
        with ZipFile(path, 'w') as archive:
            archive.writestr('Deepfake_Misinformation_Dataset.csv',
                             'CONTENT TYPE,MODEL FAKE PROBABILITY SCORE,LABEL BINARY\nvideo,0.99,1\n')
        paths.append(path)
    receipts = csv_audit(paths)
    assert receipts[0]['duplicate_of'] is None
    assert receipts[1]['duplicate_of'] == str(paths[0].resolve())
    assert all('Not used for visual training' in row['decision'] for row in receipts)
