"""Count cited claim outcomes without treating classifier scores as truth."""
from collections import Counter
from urllib.parse import urlparse

from backend.claim_context import source_can_settle_current_claim


def evidence_metrics(summary, contract=None):
    """Percentages describe statement counts, never a probability or certainty."""
    summary = summary or {}
    claims = summary.get('claims')
    claims = claims if isinstance(claims, list) else []
    unassessed = summary.get('unassessed_claim_count', 0)
    unassessed = unassessed if type(unassessed) is int and unassessed >= 0 else 0
    total = len(claims) + unassessed
    sources = {}
    for index, source in enumerate(summary.get('sources') or [], 1):
        if isinstance(source, dict):
            source_id = source.get('id', index)
            if type(source_id) is int and source_id > 0:
                sources[source_id] = source
    qualified = None
    if contract is not None:
        rows = contract.get('sources') if isinstance(contract, dict) else []
        rows = [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
        counts = Counter(row.get('id') for row in rows if type(row.get('id')) is int and row['id'] > 0)
        qualified = {row['id']: row for row in rows if type(row.get('id')) is int and row['id'] > 0
                     and counts[row['id']] == 1 and row.get('accepted') is True
                     and row.get('citation_usable') is not False}

    def cited(claim, key):
        ids = claim.get(key)
        if not isinstance(ids, list) or not ids:
            return False
        for source_id in ids:
            if type(source_id) is not int or source_id not in sources:
                return False
            source = sources[source_id]
            if qualified is not None:
                qualified_source = qualified.get(source_id)
                if qualified_source is None:
                    return False
                from backend.verification import source_record_key
                if source_record_key(source) != source_record_key(qualified_source):
                    return False
            try:
                url = urlparse(source.get('url') or '')
                if url.scheme != 'https' or not url.hostname or url.username or url.password:
                    return False
            except (TypeError, ValueError):
                return False
            if not (source.get('snippet') or (source.get('claim') and source.get('rating'))):
                return False
            if not source_can_settle_current_claim(claim.get('text', ''), source):
                return False
        return True

    supported = contradicted = 0
    for claim in claims:
        if not isinstance(claim, dict) or claim.get('conflict'):
            continue
        if claim.get('verdict') == 'true' and not claim.get('contradicting_evidence') and cited(claim, 'supporting_evidence'):
            supported += 1
        elif claim.get('verdict') == 'false' and not claim.get('supporting_evidence') and cited(claim, 'contradicting_evidence'):
            contradicted += 1

    def percentage(count):
        return round(count * 100 / total, 1) if total else None

    unresolved = total - supported - contradicted
    verdict = summary.get('verdict')
    agreement = (percentage(supported) if verdict == 'true' else
                 percentage(contradicted) if verdict == 'false' else None)
    return {
        'schema_version': 1,
        'source_policy_version': 1 if contract is not None else 0,
        'status': 'assessed' if claims else 'not_assessed',
        'supported_percentage': percentage(supported),
        'contradicted_percentage': percentage(contradicted),
        'unverified_percentage': percentage(unresolved),
        'verdict_agreement_percentage': agreement if claims else None,
        'supported_claim_count': supported,
        'contradicted_claim_count': contradicted,
        'unverified_claim_count': unresolved,
        'assessed_claim_count': len(claims),
        'unassessed_claim_count': unassessed,
        'total_claim_count': total,
        'note': ('Percentages count statements supported, contradicted, or unresolved by the cited evidence assessment. '
                 'Unassessed statements remain unresolved. Evidence agreement is the share matching the overall verdict, '
                 'not certainty or model confidence.'),
    }
