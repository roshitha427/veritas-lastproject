"""Upgrade saved model-only settlements without fetching or inventing evidence."""
from copy import deepcopy

from backend.evidence_assessment import assess_submission
from backend.report_summary import filter_relevant_evidence, summarize_report


def repair_unqualified_evidence(result, claim, kind, verifier):
    summary = result.get('report_summary') or {}
    if (summary.get('evidence_policy_version', 0) >= 1
            or summary.get('provider') != 'Local evidence comparison'
            or (summary.get('gemini_analysis') or {}).get('status') == 'completed'):
        return False
    qualification = verifier.settlement_qualification()
    if qualification.get('qualified'):
        return False
    verification = deepcopy(result.get('verification') or {'evidence': summary.get('sources') or []})
    verification = filter_relevant_evidence(claim, verification)
    assessment = assess_submission(claim, verification.get('evidence') or [])
    assessment['evidence_model'] = {'status': 'experimental_review_only',
                                  'settlement_enabled': False, 'qualification': qualification}
    corrected = summarize_report(claim, kind, result, verification,
                                 api_key=None, local_assessment=assessment)
    corrected['evidence_policy_version'] = 1
    corrected.setdefault('technical_warnings', []).append(
        'An earlier local-model verdict was rechecked against the saved source records. '
        'The experimental comparison model is not approved to settle claims independently.')
    result.setdefault('retired_evidence_comparison', {
        'verdict': summary.get('verdict'), 'provider': summary.get('provider'),
        'reason': summary.get('reason'), 'claims': deepcopy(summary.get('claims') or [])})
    result.update(report_summary=corrected, verification=verification, evidence_assessment=assessment)
    return True
