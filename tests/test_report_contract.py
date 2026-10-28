from copy import deepcopy

from backend.report_contract import attach_report_contract, build_report_contract, accepted_report_sources
from backend.rag import answer_from_report


def linked_report():
    good = {'id': 11, 'url': 'https://example.gov.in/bridge', 'title': 'Bridge record',
            'snippet': 'The Riverford bridge opened in 1985.', 'kind': 'official_source_snapshot'}
    bad = {'id': 3, 'url': 'https://example.org/unrelated', 'title': 'Other story',
           'snippet': 'The Riverford bridge opened in 2020.'}
    return {'id': 'linked-report', 'type': 'text', 'status': 'complete', 'content': 'The Riverford bridge opened in 1985.',
        'result': {'confidence': 62., 'verification': {'evidence': [good], 'checked_sources': [
            dict(good, accepted=True, reason='Matches the exact claim.'),
            dict(bad, accepted=False, reason='Unrelated page.') ]},
        'report_summary': {'schema_version': 5, 'verdict': 'true', 'sources': [bad, good],
            'summary': 'The record supports this date [11].', 'summary_source_ids': [11],
            'claims': [{'text': 'The Riverford bridge opened in 1985.', 'verdict': 'true',
                        'reason': 'See the dated bridge record [11].', 'supporting_evidence': [11]}]}}}


def test_contract_preserves_saved_ids_verdict_scores_and_is_idempotent():
    report = linked_report()
    original = deepcopy(report['result']['report_summary'])
    assert attach_report_contract(report)
    contract = report['result']['report_contract']
    assert contract['verification']['claims'][0]['supporting_source_ids'] == [11]
    assert contract['verification']['unresolved_source_ids'] == []
    assert [row['id'] for row in accepted_report_sources(report)] == [11]
    assert report['result']['confidence'] == 62.
    assert report['result']['report_summary'] == original
    assert not attach_report_contract(report)


def test_ambiguous_source_ids_are_not_resolved_to_the_first_page():
    report = linked_report()
    good = report['result']['verification']['evidence'][0]
    duplicate = dict(good, url='https://example.gov.in/different-bridge')
    report['result']['report_summary']['sources'].append(duplicate)
    report['result']['verification']['evidence'].append(duplicate)
    contract = build_report_contract(report)
    assert contract['verification']['unresolved_source_ids'] == [11]
    assert all(not row['citation_usable'] for row in contract['sources'] if row['id'] == 11)
    assert accepted_report_sources(report) == []


def test_report_questions_use_the_same_accepted_source_ids_as_the_report():
    report = linked_report()
    answer = answer_from_report('When did the Riverford bridge open?', report)
    assert answer['status'] == 'source_excerpts'
    assert [row['source_id'] for row in answer['sources']] == [11]
    assert answer['sources'][0]['citation_label'] == 'S11'
    assert '1985' in answer['sources'][0]['text']
    assert all('unrelated' not in row['url'] for row in answer['sources'])
