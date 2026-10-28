from io import BytesIO
import csv
import zipfile
import xml.etree.ElementTree as ET

from ml.prepare_immfnd import NS, prepare, read_xlsx


def workbook(rows):
    root = ET.Element('worksheet', xmlns=NS['m'])
    sheet = ET.SubElement(root, 'sheetData')
    for number, values in enumerate(rows, 1):
        row = ET.SubElement(sheet, 'row', r=str(number))
        for index, value in enumerate(values):
            if value is None:
                continue
            cell = ET.SubElement(row, 'c', r=chr(65 + index) + str(number), t='inlineStr')
            ET.SubElement(ET.SubElement(cell, 'is'), 't').text = value
    payload = BytesIO()
    with zipfile.ZipFile(payload, 'w') as zipped:
        zipped.writestr('xl/worksheets/sheet1.xml', ET.tostring(root))
    return payload.getvalue()


def test_xlsx_sparse_columns_do_not_shift_labels():
    payload = workbook([['claim', 'unused', 'label'], ['A sample claim', None, 'fake']])
    assert list(read_xlsx(payload)) == [{'claim': 'A sample claim', 'unused': '', 'label': 'fake'}]


def test_claim_only_preparation_preserves_test_and_removes_cross_split_overlap(tmp_path):
    archive = tmp_path / 'demo.zip'
    with zipfile.ZipFile(archive, 'w') as zipped:
        for split in ('train', 'validation', 'test'):
            for folder in ('Fake', 'Real'):
                label = folder.lower()
                rows = [['claim', 'label', 'language', 'url', 'content', 'date'],
                        [f'The fictional {split} {label} council approved this proposal', label, 'English',
                         f'https://example.test/{split}/{label}', 'FACT CHECK BODY REVEALS THE LABEL', '2020-01-01']]
                if folder == 'Fake':
                    rows.append(['The duplicate fictional claim belongs in the test split only', label, 'English',
                                 'https://example.test/repeated', 'Do not train on this body', '2020-01-01'])
                rows.append([None] * 6)
                zipped.writestr(f'IMMFND_Demo_Dataset/{split}/{folder}/{folder}_text.xlsx', workbook(rows))
    output = tmp_path / 'prepared'
    manifest = prepare(archive, output)
    assert manifest['audit_counts']['excluded_duplicate_claim_or_source_url'] == 2
    assert manifest['audit_counts']['ignored_empty_spreadsheet_rows'] == 6
    assert manifest['counts']['train']['total'] == 2
    assert manifest['counts']['test']['total'] == 3
    all_text = set()
    for split in ('test', 'validation', 'train'):
        with (output / f'{split}.csv').open(encoding='utf8', newline='') as handle:
            rows = list(csv.DictReader(handle))
        texts = {row['text'] for row in rows}
        assert not all_text & texts
        all_text.update(texts)
        assert all('BODY' not in row['text'] for row in rows)
        assert all(set(row) == {'text', 'label'} for row in rows)
