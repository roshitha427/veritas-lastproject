"""Explain an unresolved claim from recorded searches and source checks only."""
import re

from backend.claim_context import needs_current_sources
from backend.report_summary import filter_relevant_evidence
from backend.evidence_assessment import split_candidate_claims, _candidate_count
from backend.source_routing import is_health_claim

REVIEW_VERSION = 2


def annotate_verification_review(result, claim):
    summary = dict(result.get('report_summary') or {})
    if summary.get('verdict') != 'unverified' or summary.get('mode') == 'media_only' or not str(claim or '').strip():
        return result
    verification = dict(result.get('verification') or {})
    if 'evidence' not in verification:
        verification['evidence'] = summary.get('sources') or []
    if not verification.get('checked_sources'):
        # Legacy OCR searches may have indexed mastheads and wrapped fragments.
        # Recheck the saved pages against today's complete claim candidates.
        verification['evidence'] = [dict(row) for row in verification.get('evidence') or [] if isinstance(row, dict)]
        for row in verification['evidence']:
            row.pop('candidate_claim_indices', None)
            row.pop('relevant_claim_indices', None)
    verification = filter_relevant_evidence(claim, verification)
    searched_claim = verification.get('normalized_claim') or str(claim)
    checked = [dict(row) for row in verification.get('checked_sources', []) if isinstance(row, dict)]
    accepted = [row for row in checked if row.get('accepted')]
    rejected = [row for row in checked if not row.get('accepted')]
    # Keep saved citation IDs and original records, but make exclusion explicit
    # for legacy consumers as well as the canonical report projection.
    from backend.verification import source_record_key
    rejected_by_key = {source_record_key(row): row for row in rejected}
    if isinstance(summary.get('sources'), list):
        summary['sources'] = [dict(row, accepted=False,
            acceptance_reason=rejected_by_key[source_record_key(row)].get('reason'))
            if isinstance(row, dict) and source_record_key(row) in rejected_by_key else row
            for row in summary['sources']]
    providers = [dict(row) for row in verification.get('providers') or [] if isinstance(row, dict)]
    if not providers and (verification.get('provider') or verification.get('provider_status')):
        providers = [{'name': verification.get('provider') or 'Source search',
                      'status': verification.get('provider_status') or 'not_recorded'}]
    searches = []
    for search in verification.get('claim_searches') or []:
        if not isinstance(search, dict):
            continue
        searches.append({key: search[key] for key in ('claim_index', 'claim', 'provider_status',
                         'search_query', 'query_statuses', 'searched', 'skip_reason') if key in search})
    if not searches:
        for query in verification.get('search_queries') or ([verification['search_query']] if verification.get('search_query') else []):
            if isinstance(query, str):
                searches.append({'search_query': query, 'provider_status': verification.get('provider_status', 'not_recorded')})

    reasons = []
    def reason(text):
        if text and text not in reasons:
            reasons.append(text)
    outages = {'unavailable', 'timeout', 'network_error', 'quota_or_auth_unavailable', 'busy', 'not_configured', 'no_sources'}
    for provider in providers:
        status = provider.get('status')
        if status in outages:
            reason(f"{provider.get('name', 'Source search')} could not complete a usable search ({str(status).replace('_', ' ')}).")
    if rejected:
        for row in rejected:
            reason(row.get('reason'))
    if not accepted:
        reason('No retrieved passage directly addresses enough of the claim to establish support or contradiction. Missing coverage does not mean the claim is false.')
    else:
        reason('Relevant passages were retrieved, but they do not settle every material part of the submission with a consistent supported or contradicted outcome.')
    claims = summary.get('claims') or []
    if any(row.get('conflict') or (row.get('supporting_evidence') and row.get('contradicting_evidence'))
           for row in claims if isinstance(row, dict)):
        reason('The recorded claim assessments contain conflicting evidence; the disagreement remains unresolved.')
    skipped = [row for row in searches if row.get('searched') is False]
    if skipped:
        reason('Some statements have no completed live search; the report cannot settle the whole submission.')
    # Old OCR reports counted publisher banners as claims. Recompute coverage
    # using the same normalized candidates used by the current search path.
    unassessed = max(0, _candidate_count(searched_claim) - len(split_candidate_claims(searched_claim)))
    if unassessed:
        reason(f'{unassessed} additional statement(s) exceeded the assessment limit and remain unchecked.')
    inspection = result.get('media_inspection') or {}
    ocr_review = result.get('ocr_review') or {}
    if inspection.get('ocr_uncertain') and not ocr_review.get('user_reviewed'):
        reason('The automatic transcription contains uncertain text; reading errors may change the claim or reduce relevant source matches.')

    quoted = searched_claim.strip()
    quoted = quoted if len(quoted) <= 220 else quoted[:217] + '...'
    needed = []
    if re.search(r'\b(?:school|schools|college|colleges)\b', claim, re.I) and re.search(
            r'\b(?:clos(?:e|ed|ing|ure|ures)|shut|suspended|suspension)\b', claim, re.I):
        needed.append('A dated official education-department or district notice naming the affected locations, schools, closure dates, and any reopening instructions.')
    elif re.search(r'\b(free|giveaway|give away|giving away|offer|prize|winner|win)\b', claim, re.I):
        needed.append('An original announcement from the named company or person specifying the offer, dates, eligibility, and terms, or a reliable fact-check examining that same offer.')
    elif is_health_claim(claim):
        needed.append('Clinical research or guidance from a recognised health authority directly assessing the claimed effect on human health or lifespan.')
    elif re.search(r'\b(science|scientific|physics|chemistry|chemical|biology|dna|gravity|earth|planet|moon|orbit)\b', claim, re.I):
        needed.append('Published research, scientific measurements, or an authoritative scientific explanation directly assessing the claimed mechanism or effect.')
    else:
        needed.append(f'A primary-source record or reliable fact-check directly supporting or contradicting: “{quoted}” Include the material names, dates, and quantities.')
    if needs_current_sources(claim):
        needed.append('A dated recent announcement or report addressing this event; an old article or a recent retrieval date alone cannot establish current facts.')
    if inspection.get('ocr_performed') or result.get('ocr_text'):
        needed.append('The original post, page, or complete video with its publication date and context. Text on an image alone does not confirm the event or offer.')
        if inspection.get('ocr_uncertain') and not ocr_review.get('user_reviewed'):
            needed.append('A legible original source or accessible original article showing the exact wording of uncertain passages.')

    if checked:
        overview = (f'The recorded search returned {len(checked)} candidate page(s); '
                    f'{len(accepted)} contained passages relevant to this claim and {len(rejected)} were excluded. '
                    'The remaining evidence does not establish a factual verdict.')
    elif any(row.get('status') == 'searched' for row in providers):
        overview = 'The recorded searches returned no usable source passage addressing this claim. The report cannot confirm or refute it.'
    else:
        overview = 'No completed source check with usable passages is recorded for this claim. The report cannot confirm or refute it.'
    summary['verification_review'] = {
        'policy_version': REVIEW_VERSION, 'claim': searched_claim, 'summary': overview,
        'checked_at': verification.get('checked_at'), 'providers': providers, 'searches': searches,
        'checked_sources': checked, 'relevant_source_urls': [row.get('url') for row in accepted if row.get('url')],
        'unassessed_claim_count': unassessed,
        'insufficiency_reasons': reasons[:7], 'additional_evidence_needed': needed,
        'scope_note': 'Only searches and returned pages recorded for this report are listed. Suggested portals are follow-up resources, not sources already checked.',
    }
    result['report_summary'] = summary
    result['verification'] = verification
    return result
