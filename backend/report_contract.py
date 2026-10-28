"""One recorded report view for UI, PDF, history and source-grounded questions.

This projection never retrieves evidence, changes verdicts, renumbers saved
citations, or derives factual conclusions from model scores.
"""
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
import re
from urllib.parse import urlparse

from backend.verification import source_record_key

SCHEMA_VERSION = 1


def _rows(value):
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _ids(value):
    return list(dict.fromkeys(number for number in value if type(number) is int and number > 0)) if isinstance(value, list) else []


def _url(value):
    try:
        parsed = urlparse(value or '')
        return value if parsed.scheme == 'https' and parsed.hostname and not parsed.username and not parsed.password else None
    except (TypeError, ValueError):
        return None


def build_report_contract(report):
    result = report.get('result') or {}
    summary = result.get('report_summary') or {}
    verification = result.get('verification') or {}
    review = summary.get('verification_review') or {}
    inspection = result.get('media_inspection') or {}
    ocr = result.get('ocr_review') or {}
    media = report.get('type') in ('image', 'video')
    media_only = media and summary.get('mode') == 'media_only'
    submitted = str(report.get('content') or '')
    checked = '' if media_only else str(result.get('claim_text') or review.get('claim') or submitted
        or (result.get('media_text_analysis') or {}).get('content')
        or (ocr.get('corrected_text') if ocr.get('user_reviewed') else '') or '')
    original_ocr = str(ocr.get('original_text', inspection.get('visible_text') or result.get('ocr_text') or ''))
    prior = result.get('report_contract') or {}
    prior_ids = {}
    for source in _rows(prior.get('sources')):
        if type(source.get('id')) is int and source['id'] > 0:
            prior_ids.setdefault(source_record_key(source), source['id'])

    sources = []
    reserved = set()
    primary = _rows(summary.get('sources'))
    evidence = _rows(verification.get('evidence'))
    audits = _rows(verification.get('checked_sources')) or _rows(review.get('checked_sources'))
    audit_by_url = {}
    for row in audits:
        audit_by_url.setdefault(source_record_key(row), []).append(row)
    evidence_urls = {source_record_key(row) for row in evidence}

    def record(item, number):
        key = source_record_key(item)
        checks = audit_by_url.get(key, [])
        rejected = next((row for row in checks if row.get('accepted') is False), None)
        accepted_check = next((row for row in checks if row.get('accepted') is True), None)
        original_url = item.get('url') or ''
        valid_url = _url(original_url)
        metadata_only = item.get('retrieval_status') == 'metadata_only'
        if rejected:
            accepted, reason = False, rejected.get('reason') or 'Excluded by the recorded evidence check.'
        elif not valid_url:
            accepted, reason = False, 'No usable original HTTPS source link was recorded.'
        elif metadata_only:
            accepted, reason = False, 'Only search metadata was saved; no source passage was checked.'
        elif accepted_check or key in evidence_urls:
            accepted = True
            reason = (accepted_check or {}).get('reason') or 'Saved as a retrieved passage for this report; relevance alone is not proof of the claim.'
        else:
            accepted, reason = None, 'Legacy source citation; no completed relevance check is recorded.'
        source_key = sha256((str(number) + '\0' + key).encode()).hexdigest()[:20]
        excerpt = str(item.get('snippet') or item.get('excerpt') or '')
        return {'id': number, 'source_key': source_key, 'url': valid_url,
                'title': str(item.get('title') or ''), 'publisher': str(item.get('publisher') or ''),
                'excerpt': excerpt, 'accepted': accepted, 'acceptance_reason': str(reason),
                'retrieval_status': item.get('retrieval_status') or ('saved_passage' if excerpt else 'not_recorded'),
                'source_type': item.get('source_type') or item.get('kind') or 'source',
                'published_at': item.get('published_at') or item.get('reviewed_at'),
                'source_claim': str(item.get('claim') or ''), 'rating': item.get('rating'),
                'reliability': item.get('reliability'), 'relationship': item.get('relationship'),
                'reviewed_at': item.get('reviewed_at'), 'retrieved_at': item.get('retrieved_at') or item.get('fetched_at'),
                'cited': False}

    # Summary IDs are authoritative for existing prose such as "[11]". Even
    # duplicate/ambiguous IDs are retained and flagged rather than rewritten.
    for position, item in enumerate(primary, 1):
        number = item.get('id') if type(item.get('id')) is int and item['id'] > 0 else position
        reserved.add(number)
        sources.append(record(item, number))

    for position, item in enumerate(evidence + audits, 1):
        key = source_record_key(item)
        matches = [row for row in sources if source_record_key(row) == key]
        if matches:
            # Prefer the full saved passage over the audit's truncated excerpt.
            for row in matches:
                if not row['excerpt'] and (item.get('snippet') or item.get('excerpt')):
                    row['excerpt'] = str(item.get('snippet') or item.get('excerpt'))
            continue
        requested = item.get('id')
        number = requested if type(requested) is int and requested > 0 and requested not in reserved else prior_ids.get(key)
        if type(number) is not int or number < 1 or number in reserved:
            number = position if not primary and position not in reserved else max(reserved, default=0) + 1
        reserved.add(number)
        sources.append(record(item, number))

    claims = []
    raw_claims = _rows(summary.get('claims'))
    if not raw_claims and checked:
        raw_claims = [{'text': checked, 'verdict': summary.get('verdict', 'unverified'),
                       'reason': summary.get('reason') or '',
                       'supporting_evidence': summary.get('supporting_evidence') or [],
                       'contradicting_evidence': summary.get('contradicting_evidence') or [],
                       'missing_evidence': summary.get('missing_evidence') or []}]
    cited = set(_ids(summary.get('summary_source_ids')) + _ids(summary.get('source_ids')))
    for text in (summary.get('summary'), summary.get('reason')):
        cited.update(int(number) for number in re.findall(r'\[(\d+)\]', str(text or '')) if int(number) > 0)
    counts = Counter(row['id'] for row in sources)
    usable_ids = {row['id'] for row in sources if row['accepted'] is True and counts[row['id']] == 1}
    for index, row in enumerate(raw_claims, 1):
        supporting = _ids(row.get('supporting_evidence'))
        contradicting = _ids(row.get('contradicting_evidence'))
        related = _ids(row.get('related_evidence'))
        citations = set(supporting + contradicting + related)
        citations.update(int(number) for number in re.findall(r'\[(\d+)\]', str(row.get('reason') or '')) if int(number) > 0)
        cited.update(citations)
        claims.append({'id': 'claim-' + str(index), 'text': str(row.get('text') or ''),
                       'verdict': row.get('verdict', 'unverified'), 'reason': str(row.get('reason') or ''),
                       'supporting_source_ids': supporting, 'contradicting_source_ids': contradicting,
                       'related_source_ids': related, 'missing_evidence': deepcopy(row.get('missing_evidence') or []),
                       'unresolved_source_ids': sorted(citations - usable_ids)})
    for source in sources:
        source['cited'] = source['id'] in cited
        source['citation_usable'] = source['id'] in usable_ids
    unresolved = sorted(cited - usable_ids)
    warnings = []
    if unresolved:
        warnings.append('Some saved citation IDs are missing, excluded, unreviewed, or ambiguous. The saved wording and verdict are preserved; those references are not usable evidence links.')
    if checked and ocr.get('user_reviewed') and not str(ocr.get('corrected_text') or '').strip() and not submitted.strip():
        warnings.append('The reviewed OCR is empty, but an older checked claim remains saved. Reanalyze to reconcile this legacy record.')

    media_text = result.get('media_text_analysis') or {}
    contract = {
        'schema_version': SCHEMA_VERSION, 'report_id': report.get('id'),
        'media_type': report.get('type'), 'mode': 'media_only' if media_only else 'claim_verification',
        'claim': {'submitted_text': submitted, 'checked_text': checked,
                  'original_ocr_text': original_ocr, 'reviewed_ocr_text': str(ocr.get('corrected_text') or '') if ocr.get('user_reviewed') else None,
                  'user_reviewed': bool(ocr.get('user_reviewed')), 'ocr_uncertain': bool(inspection.get('ocr_uncertain'))},
        'verification': {'verdict': None if media_only else summary.get('verdict', 'unverified'),
                         'summary': str(review.get('summary') or summary.get('summary') or ''),
                         'reason': str(summary.get('reason') or ''),
                         'checked_at': summary.get('verification_timestamp') or verification.get('checked_at'),
                         'claims': claims, 'cited_source_ids': sorted(cited), 'unresolved_source_ids': unresolved},
        'sources': sources, 'excluded_sources': [deepcopy(row) for row in sources if row['accepted'] is False],
        'modules': {
            'text_pattern': deepcopy(result.get('text_model_assessment') or {}),
            'image_origin': deepcopy(result.get('media_origin') or {}),
            'media_text_pattern': deepcopy(media_text.get('text_model_assessment') or {}),
            'ai_writing': {'percentage': result.get('ai_percentage') if not media else media_text.get('ai_percentage'),
                           'assessment': deepcopy((result.get('ai_assessment') if not media else media_text.get('ai_assessment')) or {})},
            'temporal': deepcopy(result.get('temporal_assessment') or {}),
            'video': deepcopy(result.get('video_assessment') or {}),
        }, 'warnings': warnings,
    }
    encoded = json.dumps(contract, sort_keys=True, ensure_ascii=False, default=str, separators=(',', ':'))
    contract['revision'] = sha256(encoded.encode()).hexdigest()[:20]
    return contract


def attach_report_contract(report):
    """Attach a deterministic projection and qualify its evidence counts once."""
    if report.get('status') != 'complete' or not isinstance(report.get('result'), dict):
        return False
    contract = build_report_contract(report)
    changed = report['result'].get('report_contract') != contract
    if changed:
        report['result']['report_contract'] = contract
    from backend.evidence_metrics import evidence_metrics
    metrics = evidence_metrics(report['result'].get('report_summary'), contract)
    if report['result'].get('evidence_metrics') != metrics:
        report['result']['evidence_metrics'] = metrics
        changed = True
    return changed


def accepted_report_sources(report):
    """Source-grounded Q&A consumes the same approved rows as report links."""
    contract = build_report_contract(report)
    counts = Counter(source['id'] for source in contract['sources'])
    return [source for source in contract['sources'] if source['accepted'] is True and counts[source['id']] == 1]
