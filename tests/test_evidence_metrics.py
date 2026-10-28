from copy import deepcopy

import pytest

from backend.evidence_metrics import evidence_metrics
from tests.conftest import register


def summary(verdict='true'):
    return {'schema_version': 5, 'verdict': verdict,
            'claims': [{'text': 'GIST was established in 2008.', 'verdict': verdict,
                        'supporting_evidence': [1] if verdict == 'true' else [],
                        'contradicting_evidence': [1] if verdict == 'false' else []}],
            'sources': [{'id': 1, 'url': 'https://gist.edu.in/gist/about-gist/',
                         'snippet': 'GIST was established in 2008.'}]}


@pytest.mark.parametrize('verdict,key', [('true', 'supported_percentage'), ('false', 'contradicted_percentage')])
def test_all_cited_claims_match_verdict_gives_full_agreement(verdict, key):
    result = evidence_metrics(summary(verdict))
    assert result['status'] == 'assessed'
    assert result[key] == 100
    assert result['verdict_agreement_percentage'] == 100
    assert result['unverified_percentage'] == 0


def test_mixed_and_unassessed_claims_prevent_full_support():
    data = summary()
    data['verdict'] = 'unverified'
    data['claims'].append(summary('false')['claims'][0])
    data['unassessed_claim_count'] = 2
    result = evidence_metrics(data)
    assert result['supported_percentage'] == 25
    assert result['contradicted_percentage'] == 25
    assert result['unverified_percentage'] == 50
    assert result['total_claim_count'] == 4
    assert result['verdict_agreement_percentage'] is None
    # Even an inconsistent saved overall TRUE cannot inflate skipped statements.
    data['verdict'] = 'true'
    assert evidence_metrics(data)['verdict_agreement_percentage'] == 25


@pytest.mark.parametrize('ids', [[99], ['1'], [True], [1, 99], [], None])
def test_missing_or_invalid_citations_do_not_produce_full_agreement(ids):
    data = summary()
    data['claims'][0]['supporting_evidence'] = ids
    result = evidence_metrics(data)
    assert result['supported_percentage'] == 0
    assert result['verdict_agreement_percentage'] == 0


@pytest.mark.parametrize('change', ['metadata', 'no_excerpt', 'no_url', 'conflict', 'stale'])
def test_ineligible_or_conflicting_sources_remain_unresolved(change):
    data = summary()
    if change == 'metadata':
        data['sources'][0]['retrieval_status'] = 'metadata_only'
    elif change == 'no_excerpt':
        data['sources'][0]['snippet'] = ''
    elif change == 'no_url':
        data['sources'][0]['url'] = ''
    elif change == 'conflict':
        data['claims'][0]['contradicting_evidence'] = [1]
    elif change == 'stale':
        data['claims'][0]['text'] = 'GIST announced a new course today.'
        data['sources'][0]['published_at'] = '2008-01-01T00:00:00Z'
    result = evidence_metrics(data)
    assert result['supported_percentage'] == 0
    assert result['unverified_percentage'] == 100


def test_bare_verdict_or_model_only_explanation_has_no_evidence_percentage():
    for data in [None, {'verdict': 'true'}, {'verdict': 'unverified',
                 'gemini_model_assessment': {'verdict': 'true', 'status': 'completed'}}]:
        result = evidence_metrics(data)
        assert result['status'] == 'not_assessed'
        assert result['supported_percentage'] is None
        assert result['verdict_agreement_percentage'] is None


def test_opening_saved_true_report_updates_evidence_metrics_without_changing_model_scores(client, app):
    auth = register(client)
    original = summary()
    saved = app.extensions['store'].save('reports', {
        'owner_id': auth['user']['id'], 'type': 'text', 'status': 'complete',
        'content': 'GIST was established in 2008.',
        'result': {'real_percentage': 62., 'fake_percentage': 38., 'confidence': 62.,
                   'ai_percentage': 12.2, 'report_summary': original,
                   'verification': {'evidence': deepcopy(original['sources']),
                                    'checked_sources': [dict(source, accepted=True) for source in original['sources']]}}})
    first = client.get('/api/reports/' + saved['id']).json
    result = first['result']
    assert result['evidence_metrics']['supported_percentage'] == 100
    assert result['evidence_metrics']['verdict_agreement_percentage'] == 100
    assert result['confidence'] == result['real_percentage'] == 62
    assert result['fake_percentage'] == 38
    assert result['ai_percentage'] == 12.2
    second = client.get('/api/reports/' + saved['id']).json
    assert second['metrics_updated_at'] == first['metrics_updated_at']
    assert second['result'] == result
    assert result['report_summary']['claims'] == original['claims']


def test_new_text_analysis_has_cited_evidence_metrics(app, monkeypatch):
    import backend.analysis as module
    analyzer = app.extensions['analyzer']
    monkeypatch.setattr(analyzer, 'text', lambda text: (.38, {'status': 'ready'}))
    monkeypatch.setattr(module, 'verify_submission', lambda *a, **k: {
        'evidence': summary()['sources'],
        'checked_sources': [dict(source, accepted=True) for source in summary()['sources']]})
    monkeypatch.setattr(module, 'summarize_report', lambda *a, **k: summary())
    result = analyzer.analyze('text', 'GIST was established in 2008.', None, 'en', [])
    assert result['evidence_metrics']['supported_percentage'] == 100
    assert result['evidence_metrics']['verdict_agreement_percentage'] == 100
    assert result['confidence'] == 62


def test_pdf_retains_full_evidence_with_only_the_three_model_metrics(monkeypatch):
    import backend.pdf_report as module
    rows, paragraphs = [], []
    original_table, original_paragraph = module.Table, module.Paragraph
    def capture_table(data, *args, **kwargs):
        rows.extend(data)
        return original_table(data, *args, **kwargs)
    def capture_paragraph(text, *args, **kwargs):
        paragraphs.append(text)
        return original_paragraph(text, *args, **kwargs)
    monkeypatch.setattr(module, 'Table', capture_table)
    monkeypatch.setattr(module, 'Paragraph', capture_paragraph)
    report = {'id': 'cited-true', 'type': 'text', 'created_at': '2026-10-23',
              'result': {'report_summary': summary(), 'confidence': 62.,
                         'verification': {'evidence': summary()['sources'],
                                          'checked_sources': [dict(source, accepted=True) for source in summary()['sources']]},
                         'real_percentage': 62., 'fake_percentage': 38.}}
    before = deepcopy(report)
    assert module.report_pdf(report).startswith(b'%PDF')
    assert rows == [['Measure', 'Result'], ['Real news pattern score', '62.0%'],
                    ['Fake news pattern score', '38.0%'], ['Text model confidence', '62.0%']]
    assert any('Original source passage: GIST was established in 2008.' in text for text in paragraphs)
    assert any('https://gist.edu.in/gist/about-gist/' in text for text in paragraphs)
    assert report == before
