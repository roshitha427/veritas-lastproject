"""Offline lexical retrieval over an administrator-built, versioned evidence corpus."""
import json
import pickle
from pathlib import Path
import threading

import joblib
import numpy as np

from backend.store import now

DEFAULT_PATH = Path(__file__).resolve().parents[1] / 'models' / 'evidence' / 'index.joblib'
_lock = threading.Lock()
_cache = {}


def evidence_ready(path=None):
    return Path(path or DEFAULT_PATH).is_file()


def verify_local(text, sources, path=None):
    from backend.verification import trusted_url

    result = {'provider': 'Local evidence model', 'provider_status': 'not_configured',
        'status': 'unverified', 'checked_at': now(), 'evidence': [],
        'explanation': 'The local evidence library has not been built. Import dated fact-check records with their original source URLs. No external verification service was called.'}
    if not text.strip():
        result.update(provider_status='no_claim', explanation='Add a claim or caption to search the local evidence library.')
        return result
    path = Path(path or DEFAULT_PATH)
    if not path.is_file():
        return result
    try:
        stamp = (str(path.resolve()), path.stat().st_mtime_ns)
        with _lock:
            bundle = _cache.get(stamp)
            if bundle is None:
                # Only load artifacts built locally by ml.build_evidence, never uploaded pickle files.
                bundle = joblib.load(path)
                _cache.clear()
                _cache[stamp] = bundle
        if bundle.get('format_version') != 1:
            raise ValueError('Unknown local evidence format')
        card = bundle['card']
        query = bundle['vectorizer'].transform([text[:12000]])
        scores = (bundle['matrix'] @ query.T).toarray().ravel()
        for index in np.argsort(-scores):
            if scores[index] < .18:
                break
            record = bundle['documents'][int(index)]
            if not trusted_url(record['url'], sources):
                continue
            result['evidence'].append({key: record.get(key, '') for key in (
                'title', 'claim', 'url', 'publisher', 'rating', 'reviewed_at')})
            result['evidence'][-1].update(kind='local_fact_check',
                snippet=record.get('text', '')[:4000],
                relevance_score=round(float(scores[index]), 4),
                relationship='Retrieved from a local library. Compare the exact claim and date; the source rating belongs to the archived claim.')
            if len(result['evidence']) >= 5:
                break
        result.update(provider_status='searched', status='needs_review' if result['evidence'] else 'unverified',
            corpus={'version': card['version'], 'documents': card['document_count'], 'built_at': card['built_at'],
                    'latest_reviewed_at': card.get('latest_reviewed_at'), 'method': card['method']},
            explanation='Related archived fact checks were found by the local retrieval model. Read the original source and compare the claim, date and context. The library does not update itself.'
                if result['evidence'] else 'No related record from an enabled trusted source was found in the local library. Missing coverage does not establish truth or falsehood.')
    except (OSError, ValueError, KeyError, TypeError, EOFError, IndexError, AttributeError, pickle.UnpicklingError):
        result.update(provider_status='unavailable', explanation='The local evidence index could not be loaded. Rebuild it from the original JSON or JSONL corpus. No external verification service was called.')
    return result
