"""Prepare English claim-only examples from the supplied IMMFND Excel sheets.

Read XLSX as zipped XML, without evaluating formulas or requiring Office.
Fact-check article bodies disclose the answer, so only the claim is a feature.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

from ml.text_cleaning import clean_text


NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def read_xlsx(payload):
    with zipfile.ZipFile(BytesIO(payload)) as book:
        strings = []
        if 'xl/sharedStrings.xml' in book.namelist():
            strings = [''.join(node.itertext()) for node in
                       ET.fromstring(book.read('xl/sharedStrings.xml')).findall('m:si', NS)]
        root = ET.fromstring(book.read('xl/worksheets/sheet1.xml'))
        header = None
        for row in root.findall('m:sheetData/m:row', NS):
            cells = {}
            for cell in row.findall('m:c', NS):
                letters = re.match(r'[A-Z]+', cell.get('r', '')).group()
                value = cell.find('m:v', NS)
                text = value.text if value is not None and value.text else ''
                if cell.get('t') == 's' and text:
                    text = strings[int(text)]
                elif cell.get('t') == 'inlineStr':
                    text = ''.join(cell.find('m:is', NS).itertext())
                cells[letters] = text
            if header is None:
                header = cells
            else:
                yield {name: cells.get(column, '') for column, name in header.items()}


def key(text):
    return hashlib.sha256(' '.join(re.findall(r'\w+', text.casefold())).encode()).hexdigest()


def prepare(archive, output):
    archive, output = Path(archive), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose an empty output directory to preserve previous splits.')
    counts, languages, members = Counter(), Counter(), []
    rows = defaultdict(list)
    with zipfile.ZipFile(archive) as zipped:
        for split in ('train', 'validation', 'test'):
            for folder, label in [('Fake', 'fake'), ('Real', 'real')]:
                name = f'IMMFND_Demo_Dataset/{split}/{folder}/{folder}_text.xlsx'
                payload = zipped.read(name)
                members.append({'member': name, 'sha256': hashlib.sha256(payload).hexdigest()})
                for row in read_xlsx(payload):
                    if not any(row.get(field) for field in ('url', 'claim', 'content', 'label')):
                        counts['ignored_empty_spreadsheet_rows'] += 1
                        continue
                    counts['input_rows'] += 1
                    languages[row.get('language', '')] += 1
                    if row.get('label', '').strip().casefold() != label:
                        counts['excluded_label_mismatch'] += 1
                        continue
                    if row.get('language', '').strip().casefold() != 'english':
                        counts['excluded_non_english'] += 1
                        continue
                    claim = clean_text(row.get('claim', ''))
                    if len(claim.split()) < 4 or re.search(r'[\u0900-\u0dff]', claim):
                        counts['excluded_short_or_non_english_script'] += 1
                        continue
                    rows[split].append({'text': claim, 'label': label, 'url': row.get('url', ''),
                                        'date': row.get('date', ''), 'key': key(claim), 'member': name})
    labels = defaultdict(set)
    for examples in rows.values():
        for row in examples:
            labels[row['key']].add(row['label'])
    conflicts = {identity for identity, values in labels.items() if len(values) > 1}
    seen, urls = set(), set()
    output.mkdir(parents=True, exist_ok=True)
    split_counts = {}
    for split in ('test', 'validation', 'train'):
        clean = []
        for row in rows[split]:
            if row['key'] in conflicts:
                counts['excluded_conflicting_claims'] += 1
                continue
            if row['key'] in seen or (row['url'] and row['url'] in urls):
                counts['excluded_duplicate_claim_or_source_url'] += 1
                continue
            seen.add(row['key'])
            if row['url']:
                urls.add(row['url'])
            clean.append(row)
        if set(row['label'] for row in clean) != {'fake', 'real'}:
            raise ValueError('Every prepared split must retain both classes.')
        with (output / f'{split}.csv').open('w', encoding='utf8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['text', 'label'])
            writer.writeheader()
            writer.writerows({key: row[key] for key in ('text', 'label')} for row in clean)
        (output / f'{split}-provenance.jsonl').write_text(
            '\n'.join(json.dumps(row, ensure_ascii=False) for row in clean) + '\n', encoding='utf8')
        split_counts[split] = dict(Counter(row['label'] for row in clean), total=len(clean))
    with archive.open('rb') as handle:
        sha = hashlib.file_digest(handle, 'sha256').hexdigest()
    manifest = {'name': 'IMMFND demo: English claim-only classification', 'version': sha[:12],
        'archive_path': str(archive.resolve()), 'archive_sha256': sha, 'members': members,
        'audit_counts': dict(counts), 'original_language_counts': dict(languages), 'counts': split_counts,
        'languages': ['en'], 'feature': 'claim only; no title, fact-check body, source, category, or label metadata',
        'split_policy': 'Preserve supplied split membership. Deduplicate normalized claims and source URLs, test > validation > train; remove contradictory labels.',
        'license': 'No license file included in the supplied demo ZIP; original source publishers retain their content.',
        'limitations': ['Historical fact-check claims and news claims, not a live current-affairs knowledge base.',
                        'Only English examples are prepared for the existing English BERT model.',
                        'Folder/row fake-real annotations are dataset labels, not independently reverified facts.',
                        'No image-manipulation labels are inferred from fake-news labels; image files remain untouched.',
                        'Exact deduplication does not establish independent events or near-duplicate-free splits.']}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--output', default='data/immfnd-english-claims')
    args = parser.parse_args()
    prepare(args.archive, args.output)
