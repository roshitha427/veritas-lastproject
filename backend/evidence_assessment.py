"""Conservative, auditable verdicts derived from retrieved evidence records."""
from datetime import datetime, timezone
import re

from backend.verification import STOPWORDS
from backend.claim_context import normalize_places, source_can_settle_current_claim


VERDICTS = {'true', 'false', 'unverified'}
_TOKENS = re.compile(r"[\w'-]+", re.UNICODE)
_REVIEW_LABELS = {
    'true': 'true', 'correct': 'true', 'accurate': 'true',
    'false': 'false', 'incorrect': 'false', 'wrong': 'false',
    'fabricated': 'false', 'fake': 'false',
}


def claim_tokens(text):
    # Keep negation: dropping "not" can make opposite claims look identical.
    ignored = STOPWORDS - {'not', 'no', 'never'}
    return {token.casefold() for token in _TOKENS.findall(normalize_places(text))
            if len(token) > 1 and token.casefold() not in ignored}


def split_candidate_claims(text, limit=5):
    """Return sentence-sized candidates for long, multi-sentence submissions."""
    from backend.source_routing import normalize_search_submission
    text = normalize_search_submission(text)
    content = re.sub(r'[ \t\r]+', ' ', str(text or '')).strip()
    if not content:
        return []
    sentences = [part.strip(' \t\r\n\u201c\u201d\"') for part in
                 re.split(r'(?<=[.!?])\s+|\n+', content) if part.strip()]
    if len(sentences) < 2:
        return [content[:2000]]
    # Short statements still carry material claims ("The Earth is flat.").
    # Keep every sentence so a long supported statement cannot hide a short
    # unsupported one. Nonfactual fragments can conservatively remain unverified.
    return [sentence[:2000] for sentence in sentences[:limit]]


def contextual_candidate_claims(text, limit=5):
    """Keep a dependent follow-up attached to the statement it refers to.

    The returned positions still match split_candidate_claims. Original claim
    text remains unchanged in assessments; only search/relevance queries gain
    their existing antecedent, never invented names, dates, or facts.
    """
    claims = split_candidate_claims(text, limit)
    queries, antecedent = [], None
    dependent = re.compile(
        r'^(?:(?:it|its|they|their|these|those|this|that)\b|'
        r'(?:the|this|that)\s+(?:message|post|announcement|notice|claim|report|video|image)\b|'
        r'(?:parents|residents|students|children|citizens|patients|staff|employees|viewers)\s+'
        r'(?:are|were|have been|should|must|were told|are advised)\b)', re.I)
    for candidate in claims:
        if antecedent and dependent.search(candidate):
            queries.append(antecedent + ' ' + candidate)
        else:
            queries.append(candidate)
            antecedent = candidate
    return queries


def _candidate_count(text):
    from backend.source_routing import normalize_search_submission
    text = normalize_search_submission(text)
    content = re.sub(r'[ \t\r]+', ' ', str(text or '')).strip()
    if not content:
        return 0
    sentences = [part.strip() for part in re.split(r'(?<=[.!?])\s+|\n+', content) if part.strip()]
    if len(sentences) < 2:
        return 1
    return len(sentences)


def _rating_verdict(rating):
    normalized = re.sub(r'[^a-z]+', ' ', str(rating or '').casefold()).strip()
    return _REVIEW_LABELS.get(normalized)


def _same_claim(submitted, reviewed):
    left, right = claim_tokens(submitted), claim_tokens(reviewed)
    if len(left) < 3 or len(right) < 3:
        return False
    # Lexical similarity alone lets opposite statements or changed quantities
    # inherit a review's rating. Those require passage comparison instead.
    negations = {'not', 'no', 'never', 'neither', 'nor', 'without'}
    def negative(text, tokens):
        return bool(tokens & negations or re.search(r"n['\u2019]t\b", text, re.I))
    if negative(submitted, left) != negative(reviewed, right):
        return False
    if set(re.findall(r'\d+(?:[.,]\d+)*', submitted)) != set(re.findall(r'\d+(?:[.,]\d+)*', reviewed)):
        return False
    overlap = len(left & right)
    # Require a close match, including negation and material names/numbers.
    return overlap >= 3 and overlap / max(len(left), len(right)) >= 0.78


def _fresh_for_current_claim(claim, reviewed_at, days=90):
    if not re.search(r'\b(current|currently|today|now|latest|this\s+week|this\s+month)\b', claim, re.I):
        return True
    if not reviewed_at:
        return False
    try:
        value = datetime.fromisoformat(str(reviewed_at).replace('Z', '+00:00'))
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - value.astimezone(timezone.utc)).days
        return 0 <= age_days <= days
    except (TypeError, ValueError):
        return False


def assess_claim(claim, evidence, retrieved_ids=None):
    """Use only close-matching explicit fact-check ratings for TRUE/FALSE."""
    if retrieved_ids is None:
        retrieved_ids = list(range(1, len(evidence or []) + 1))
    matched = []
    for index, item in enumerate(evidence or [], start=1):
        if not isinstance(item, dict) or not item.get('claim') or not item.get('rating'):
            continue
        if not source_can_settle_current_claim(claim, item):
            continue
        if not _same_claim(claim, item.get('claim')):
            continue
        verdict = _rating_verdict(item.get('rating'))
        if not verdict or not _fresh_for_current_claim(claim, item.get('reviewed_at')):
            continue
        matched.append((index, verdict, item))

    distinct = {verdict for _, verdict, _ in matched}
    if len(distinct) > 1:
        return {
            'verdict': 'unverified',
            'reason': 'Close-matching fact-check records give conflicting TRUE and FALSE ratings. The disagreement is unresolved, so this report does not choose either verdict.',
            'supporting_evidence': [index for index, verdict, _ in matched if verdict == 'true'],
            'contradicting_evidence': [index for index, verdict, _ in matched if verdict == 'false'],
            'conflict': True,
        }
    if distinct:
        verdict = next(iter(distinct))
        ids = [index for index, _, _ in matched]
        ratings = ', '.join(sorted({str(item.get('rating')) for _, _, item in matched}))
        reason = (f'{len(matched)} close-matching fact-check record(s) rated the submitted claim "{ratings}". '
                  'This verdict reflects those published reviews; check their dates and cited evidence for context.')
        return {
            'verdict': verdict, 'reason': reason,
            'supporting_evidence': ids if verdict == 'true' else [],
            'contradicting_evidence': ids if verdict == 'false' else [],
            'conflict': False,
        }

    ids = list(retrieved_ids)
    if ids:
        reason = ('Related reporting or background excerpts were retrieved, but none was a fresh, close-matching '
                  'fact-check with an unambiguous rating. They do not establish this claim as true or false.')
    else:
        reason = ('No relevant source excerpts were retrieved, so there is not enough evidence to establish this '
                  'claim as true or false. Missing coverage does not mean the claim is false.')
    return {'verdict': 'unverified', 'reason': reason, 'supporting_evidence': [],
            'contradicting_evidence': [], 'conflict': False}


def assess_submission(text, evidence):
    """Assess each candidate and use a cautious article-level aggregate."""
    claims = []
    all_candidate_count = _candidate_count(text)
    candidates = split_candidate_claims(text)
    for candidate in candidates:
        candidate_terms = claim_tokens(candidate)
        related_ids = []
        for index, item in enumerate(evidence or [], start=1):
            item_terms = claim_tokens(' '.join(str(item.get(key) or '')
                for key in ('title', 'claim', 'snippet', 'rating')))
            if candidate_terms and len(candidate_terms & item_terms) >= min(2, len(candidate_terms)):
                related_ids.append(index)
        assessed = assess_claim(candidate, evidence, related_ids)
        claims.append({'text': candidate, **assessed, 'related_evidence': related_ids,
                       'missing_evidence': [] if assessed['verdict'] != 'unverified' else [
                           'A current, directly matching fact-check or primary source that settles the claim.'
                       ]})

    verdicts = {item['verdict'] for item in claims}
    overall = next(iter(verdicts)) if len(verdicts) == 1 else 'unverified'
    # An article receives TRUE/FALSE only if every extracted candidate has that same settled verdict.
    if overall != 'unverified' and len(claims) > 1 and any(item['verdict'] == 'unverified' for item in claims):
        overall = 'unverified'
    reason = (' '.join(item['reason'] for item in claims[:3]))[:1200]
    unassessed = max(0, all_candidate_count - len(candidates))
    if unassessed:
        overall = 'unverified'
        reason = (f'Only {len(candidates)} factual statements were assessed; {unassessed} additional statement(s) '
                  'exceeded the report limit and remain unverified. ' + reason)
    if len(claims) > 1 and overall == 'unverified':
        reason = 'The submission contains multiple factual statements; they do not all have the same settled verdict. ' + reason
    supporting = sorted({number for item in claims for number in item['supporting_evidence']})
    contradicting = sorted({number for item in claims for number in item['contradicting_evidence']})
    return {'verdict': overall, 'claims': claims, 'reason': reason[:1500],
            'supporting_evidence': supporting, 'contradicting_evidence': contradicting,
            'unassessed_claim_count': unassessed}
