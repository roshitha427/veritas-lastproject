"""Fit a local retrieval model from dated fact checks, without external APIs.

Input JSON list or JSONL: title, claim, url (HTTPS), publisher, rating,
reviewed_at (ISO date), optionally text (the source's evidence/reasoning).
Never use model-generated reasoning as independently verified evidence.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from backend.store import now


def build(input_path, output, version):
    input_path, output = Path(input_path), Path(output)
    if input_path.stat().st_size > 25 * 1024 * 1024:
        raise ValueError('Use a corpus of 25 MB or less per version for this local prototype.')
    raw = input_path.read_text(encoding='utf-8-sig')
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()] if input_path.suffix.lower() == '.jsonl' else json.loads(raw)
    if not isinstance(rows, list) or not 2 <= len(rows) <= 20000:
        raise ValueError('The corpus must be a list of 2–20,000 evidence records.')
    documents, seen = [], set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Each evidence record must be an object.')
        required = ('title', 'claim', 'url', 'publisher', 'reviewed_at')
        if any(not isinstance(row.get(key), str) or not row[key].strip() for key in required):
            raise ValueError('Each record needs title, claim, url, publisher and reviewed_at strings.')
        record = {key: str(row.get(key, '')).strip() for key in (*required, 'rating', 'text')}
        parsed = urlparse(record['url'])
        if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or len(record['url']) > 2000:
            raise ValueError('Evidence URLs must be original HTTPS source links without credentials.')
        datetime.fromisoformat(record['reviewed_at'].replace('Z', '+00:00'))
        if any(len(record[key]) > limit for key, limit in [('title', 1000), ('claim', 12000), ('text', 40000), ('rating', 500), ('publisher', 200)]):
            raise ValueError('An evidence field exceeds its supported length.')
        identity = (record['url'], ' '.join(record['claim'].lower().split()))
        if identity not in seen:
            documents.append(record)
            seen.add(identity)
    if len(documents) < 2:
        raise ValueError('At least two distinct evidence records are required.')
    # Character n-grams can index English/Hindi/Telugu text without an English tokenizer.
    # This is within-language lexical retrieval, not automatic cross-language translation.
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=100000, sublinear_tf=True)
    matrix = vectorizer.fit_transform([' '.join((row['title'], row['claim'], row['claim'], row['text'])) for row in documents])
    card = {'name': 'Local evidence retrieval model', 'version': version, 'document_count': len(documents),
        'built_at': now(), 'source_sha256': hashlib.sha256(raw.encode()).hexdigest(),
        'latest_reviewed_at': max(datetime.fromisoformat(row['reviewed_at'].replace('Z', '+00:00')).date().isoformat() for row in documents),
        'method': 'Character TF-IDF and cosine similarity', 'evaluation': None,
        'scope': 'Retrieve archived source records. No automatic current-truth, entailment or cross-language guarantee.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp')
    joblib.dump({'format_version': 1, 'vectorizer': vectorizer, 'matrix': matrix, 'documents': documents, 'card': card}, temporary)
    temporary.replace(output)
    output.with_name('model-card.json').write_text(json.dumps(card, indent=2), encoding='utf8')
    return card


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', default='models/evidence/index.joblib')
    parser.add_argument('--version', required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.input, args.output, args.version), indent=2))
