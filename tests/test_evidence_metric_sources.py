from copy import deepcopy
import time

import pytest

from backend.evidence_metrics import evidence_metrics
from backend.report_contract import attach_report_contract, build_report_contract
from tests.conftest import register


def saved_report(verdict='true', state='accepted'):
    source = {'id': 11, 'url': 'https://example.gov.in/bridge', 'title': 'Bridge record',
              'snippet': 'The Riverford bridge opened in 1985.', 'kind': 'official_source_snapshot'}
    summary = {'schema_version': 5, 'verdict': verdict, 'summary': 'See the saved record [11].',
        'sources': [source], 'summary_source_ids': [11], 'claims': [
            {'text': 'The Riverford bridge opened in 1985.', 'verdict': verdict,
             'supporting_evidence': [11] if verdict == 'true' else [],
             'contradicting_evidence': [11] if verdict == 'false' else []}]}
    verification = {'evidence': [source], 'checked_sources': [dict(source, accepted=True)]}
    if state == 'excluded':
        verification['checked_sources'][0].update(accepted=False, reason='Different event.')
    elif state == 'unknown':
        verification = {'evidence': []}
    elif state == 'ambiguous':
        duplicate = dict(source, url='https://example.gov.in/other-bridge')
        summary['sources'].append(duplicate)
        verification['evidence'].append(duplicate)
        verification['checked_sources'].append(dict(duplicate, accepted=True))
    return {'id': 'qualified-metrics', 'status': 'complete', 'type': 'text',
        'content': 'The Riverford bridge opened in 1985.',
        'result': {'report_summary': summary, 'verification': verification,
                   'real_percentage': 62., 'fake_percentage': 38., 'confidence': 62., 'ai_percentage': 12.2}}


@pytest.mark.parametrize('verdict', ['true', 'false'])
@pytest.mark.parametrize('state', ['excluded', 'ambiguous', 'unknown'])
def test_only_accepted_unambiguous_sources_can_count_a_claim_outcome(verdict, state):
    report = saved_report(verdict, state)
    summary = report['result']['report_summary']
    original = deepcopy(summary)
    contract = build_report_contract(report)
    metrics = evidence_metrics(summary, contract)
    assert metrics['supported_percentage'] == metrics['contradicted_percentage'] == 0
    assert metrics['unverified_percentage'] == 100
    assert metrics['verdict_agreement_percentage'] == 0
    assert metrics['source_policy_version'] == 1
    assert summary == original


def test_qualifying_counts_requires_source_url_to_match_the_saved_citation():
    report = saved_report()
    contract = build_report_contract(report)
    assert evidence_metrics(report['result']['report_summary'], contract)['supported_percentage'] == 100
    contract['sources'][0]['url'] = 'https://example.gov.in/different-record'
    assert evidence_metrics(report['result']['report_summary'], contract)['supported_percentage'] == 0


def test_saved_report_refresh_qualifies_once_without_repeated_storage_writes(client, app, monkeypatch):
    auth = register(client)
    report = saved_report(state='excluded')
    report['owner_id'] = auth['user']['id']
    # Simulate the earlier unqualified metric saved by an old analysis pipeline.
    report['result']['evidence_metrics'] = evidence_metrics(report['result']['report_summary'])
    assert report['result']['evidence_metrics']['supported_percentage'] == 100
    store = app.extensions['store']
    store.save('reports', report)
    first = client.get('/api/reports/' + report['id']).json
    assert first['result']['evidence_metrics']['supported_percentage'] == 0
    assert first['result']['report_summary']['verdict'] == 'true'
    assert first['result']['confidence'] == first['result']['real_percentage'] == 62.
    assert first['result']['fake_percentage'] == 38.
    assert first['result']['ai_percentage'] == 12.2
    writes = []
    original_save = store.save
    def save(collection, document):
        if collection == 'reports':
            writes.append(document['id'])
        return original_save(collection, document)
    monkeypatch.setattr(store, 'save', save)
    second = client.get('/api/reports/' + report['id']).json
    assert second['result'] == first['result']
    assert second['metrics_updated_at'] == first['metrics_updated_at']
    assert client.get('/api/reports').status_code == 200
    assert client.get('/api/reports/' + report['id'] + '/pdf').status_code == 200
    assert writes == []


def test_worker_persists_qualified_metrics_before_the_first_api_read(client, app, monkeypatch):
    auth = register(client)
    result = saved_report(state='excluded')['result']
    result['evidence_metrics'] = evidence_metrics(result['report_summary'])
    monkeypatch.setattr(app.extensions['analyzer'], 'analyze', lambda *args, **kwargs: deepcopy(result))
    response = client.post('/api/analyze', data={'type': 'text', 'text': 'The Riverford bridge opened in 1985.'},
        headers={'X-CSRF-Token': auth['csrf_token']})
    assert response.status_code == 202
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        stored = app.extensions['store'].one('reports', {'id': response.json['id']})
        if stored['status'] in ('complete', 'failed'):
            break
        time.sleep(.01)
    assert stored['status'] == 'complete'
    assert stored['result']['evidence_metrics']['supported_percentage'] == 0
    assert stored['result']['evidence_metrics']['unverified_percentage'] == 100
    assert not attach_report_contract(stored)
