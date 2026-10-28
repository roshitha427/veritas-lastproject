from types import SimpleNamespace
from unittest.mock import Mock, patch
import json
import requests
import pytest

from backend.evidence_assessment import assess_submission
from backend.report_summary import filter_relevant_evidence, summarize_report


CLAIM = 'Apollo 11 mission landed on the Moon in 1969.'


def test_lifespan_claim_requires_health_outcome_evidence_not_a_recipe():
    from backend.report_summary import _source_relevance
    claim = 'Eating fruit everyday makes humans immortal.'
    recipe = {'url': 'https://example.gov/recipes/fruit-cake', 'title': 'Fruit cake recipe',
              'reliability': 'official_primary_source',
              'snippet': 'Eating fruit everyday is easy. Humans can make this fruit cake with flour and water.'}
    assert _source_relevance(claim, recipe)['relevant'] is False
    study = {'url': 'https://example.gov/research/fruit', 'title': 'Fruit intake and human life expectancy',
             'reliability': 'official_primary_source',
             'snippet': 'Eating fruit daily was studied in humans. The study measured mortality and life expectancy; it did not establish unlimited lifespan.'}
    assert _source_relevance(claim, study)['relevant'] is True


@pytest.mark.parametrize('status,expected', [
    ('unavailable', 'unavailable'),
    ('timeout', 'unavailable'),
    ('network_error', 'unavailable'),
    ('busy', 'busy'),
    ('quota_or_auth_unavailable', 'credential or usage-limit'),
    ('not_configured', 'No source provider is configured'),
    ('no_sources', 'No supported sources are enabled'),
])
def test_failed_source_search_is_not_described_as_a_completed_search_without_matches(status, expected):
    result = summarize_report(CLAIM, 'text', {}, {'provider_status': status, 'evidence': []})
    assert result['verdict'] == 'unverified'
    assert expected in result['summary']
    assert 'No relevant source excerpts were retrieved' not in result['reason']
    assert result['sources'] == [] and result['summary_source_ids'] == []
    if status == 'quota_or_auth_unavailable':
        assert any('credentials' in warning for warning in result['technical_warnings'])


def test_successful_empty_search_remains_a_coverage_gap():
    result = summarize_report(CLAIM, 'text', {}, {'provider_status': 'searched', 'evidence': []})
    assert result['verdict'] == 'unverified'
    assert 'No directly relevant source excerpt' in result['summary']
    assert 'Missing coverage does not mean the claim is false' in result['reason']
    assert not any(word in result['summary'] for word in ('credential', 'busy', 'unavailable'))


def test_generated_explanation_cannot_hide_a_recorded_source_outage():
    response = model_response('unverified', summary='No source matched this input.',
                              claim_assessments=[{'verdict': 'unverified',
                                  'reason': 'No source supports a factual verdict.',
                                  'supporting_source_ids': [], 'contradicting_source_ids': []}])
    with patch('backend.report_summary.requests.post', return_value=response):
        result = summarize_report(CLAIM, 'text', {}, {'provider_status': 'unavailable', 'evidence': []},
                                  api_key='test-gemini-key-12345')
    assert result['gemini_analysis']['status'] == 'completed'
    assert result['verdict'] == 'unverified'
    assert 'source search service was unavailable' in result['summary']
    assert result['sources'] == [] and result['summary_source_ids'] == []


def test_gemini_explanation_is_kept_when_related_sources_do_not_settle_input():
    source = {'url': 'https://www.reuters.com/space/apollo', 'title': 'Apollo 11 mission in 1969',
              'snippet': 'Background about Apollo 11 and the Moon landing in 1969.', 'kind': 'source_article'}
    payload = model_response('unverified', summary='The available passage is background, not a complete verification.').json()
    generated = json.loads(payload['candidates'][0]['content']['parts'][0]['text'])
    generated['model_only_assessment'] = {'verdict': 'true', 'answer': 'Apollo 11 landed on the Moon in July 1969.',
        'reason': 'This is a best-effort explanation from model knowledge, not a source-verified verdict.'}
    payload['candidates'][0]['content']['parts'][0]['text'] = json.dumps(generated)
    response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)
    with patch('backend.report_summary.requests.post', return_value=response):
        result = summarize_report(CLAIM, 'text', {}, {'evidence': [source]}, api_key='test-api-key-12345')
    assert result['verdict'] == 'unverified'
    assert result['gemini_model_assessment']['answer'] == generated['model_only_assessment']['answer']
    assert result['gemini_model_assessment']['status'] == 'completed'


def test_rejected_evidence_citations_do_not_discard_separate_gemini_explanation():
    payload = model_response('true', support=[99]).json()
    generated = json.loads(payload['candidates'][0]['content']['parts'][0]['text'])
    generated['model_only_assessment'] = {'verdict': 'true', 'answer': 'Apollo 11 landed on the Moon in July 1969.',
        'reason': 'This explanation is based on model knowledge and has not been source-verified.'}
    payload['candidates'][0]['content']['parts'][0]['text'] = json.dumps(generated)
    response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)
    with patch('backend.report_summary.requests.post', return_value=response):
        result = summarize_report(CLAIM, 'text', {}, {'evidence': []}, api_key='test-api-key-12345')
    assert result['verdict'] == 'unverified'
    assert result['gemini_analysis']['status'] == 'response_rejected'
    assert result['gemini_model_assessment']['answer']


def test_no_answer_has_a_plain_language_message():
    result = summarize_report(CLAIM, 'text', {}, {'evidence': []})
    message = result['gemini_model_assessment']['user_message']
    assert 'reliable information' in message
    assert 'improve our coverage' in message


def test_official_office_directory_can_contradict_alleged_holder():
    from backend.report_summary import _source_relevance
    source = {'kind': 'official_source_snapshot', 'reliability': 'official_primary_source',
              'title': 'Example Pradesh leadership', 'snippet': 'Aruna Example is the Chief Minister of Example Pradesh.'}
    relevance = _source_relevance('current Example Pradesh cm is Bala Wrong', source)
    assert relevance['relevant'] is True
    assert relevance['reason'] == 'office_and_jurisdiction_match'
    assert not _source_relevance('current Another State cm is Bala Wrong', source)['relevant']


def review(rating='True', claim=CLAIM, reviewed_at='2026-10-18T00:00:00Z'):
    return {
        'kind': 'fact_check_review', 'source_type': 'published_fact_check',
        'claim': claim, 'rating': rating, 'reviewed_at': reviewed_at,
        'title': 'Apollo 11 fact check', 'publisher': 'Reuters',
        'url': 'https://www.reuters.com/fact-check/apollo-11',
        'snippet': 'The mission landed on the Moon in July 1969.',
    }


def model_response(verdict='true', support=None, contradict=None, summary=None,
                   claim_assessments=None, summary_ids=None):
    support = support or []
    contradict = contradict or []
    summary_ids = summary_ids or sorted(set(support + contradict))
    claim_assessments = claim_assessments or [{
        'verdict': verdict,
        'reason': 'The cited excerpt directly addresses the submitted claim.',
        'supporting_source_ids': support,
        'contradicting_source_ids': contradict,
    }]
    generated = {
        'verdict': verdict,
        'headline': 'Assessment based on retrieved evidence',
        'summary': summary or 'The retrieved excerpt directly addresses the claim [1].',
        'reason': 'The cited excerpt directly addresses the submitted claim [1].',
        'summary_source_ids': summary_ids,
        'claim_assessments': claim_assessments,
        'key_points': [],
    }
    return SimpleNamespace(raise_for_status=lambda: None,
        json=lambda: {'candidates': [{'content': {'parts': [{'text': json.dumps(generated)}]}}]})


def test_explicit_matching_fact_check_can_support_true_or_false():
    assert assess_submission(CLAIM, [review('True')])['verdict'] == 'true'
    result = assess_submission(CLAIM, [review('False')])
    assert result['verdict'] == 'false'
    assert result['contradicting_evidence'] == [1]


def test_evidence_grounded_overall_explanation_can_exceed_the_old_one_line_limit():
    summary = ('The mission archive records the Apollo 11 Moon landing in 1969 and directly supports the submitted statement [1]. ' * 6).strip()
    assert 500 < len(summary) <= 1200
    with patch('backend.report_summary.requests.post', return_value=model_response('true', support=[1], summary=summary)) as request:
        result = summarize_report(CLAIM, 'text', {}, {'evidence': [review()]}, api_key='test-api-key-12345')
    assert result['gemini_analysis']['status'] == 'completed'
    assert result['verdict'] == 'true'
    assert result['summary'] == summary
    assert result['summary_source_ids'] == [1]
    prompt = request.call_args.kwargs['json']['contents'][0]['parts'][0]['text']
    assert '100-160 words' in prompt
    assert 'never pad the report or invent details' in prompt


def test_related_reporting_or_ambiguous_rating_is_unverified():
    article = review('Mostly false')
    article['kind'] = 'related_reporting'
    article['snippet'] = 'Apollo 11 mission coverage.'
    result = assess_submission(CLAIM, [article])
    assert result['verdict'] == 'unverified'
    assert 'do not establish' in result['reason']


def test_irrelevant_article_with_shared_country_topic_is_filtered():
    verification = {'provider_status':'searched','evidence':[
        {'kind':'related_reporting','title':'India Constitution video goes viral',
         'url':'https://www.reuters.com/world/india-example','publisher':'Reuters',
         'snippet':'A video about the Constitution of India was shared online.'}]}
    filtered = filter_relevant_evidence('The Sun will stop producing light for 72 hours next week.', verification)
    assert filtered['evidence'] == []
    assert filtered['filtered_irrelevant_count'] == 1


def test_material_year_mismatch_does_not_make_article_relevant():
    verification = {'provider_status':'searched','evidence':[
        {'kind':'related_reporting','title':'Indian Constitution timeline',
         'url':'https://legislative.gov.in/example','publisher':'Government of India',
         'snippet':'The Constitution came into force on 26 January 1951.'}]}
    filtered = filter_relevant_evidence("India's Constitution came into effect on 26 January 1950.", verification)
    assert filtered['evidence'] == []


def test_conflicting_direct_reviews_remain_unverified():
    result = assess_submission(CLAIM, [review('True'), review('False')])
    assert result['verdict'] == 'unverified'
    assert result['claims'][0]['conflict'] is True
    assert result['supporting_evidence'] == [1]
    assert result['contradicting_evidence'] == [2]


def test_old_fact_check_cannot_settle_a_current_claim():
    claim = 'Apollo 11 is currently the latest mission to land on the Moon.'
    result = assess_submission(claim, [review('True', claim, '2020-01-01T00:00:00Z')])
    assert result['verdict'] == 'unverified'


def test_article_with_multiple_claims_needs_each_claim_settled():
    article = CLAIM + ' The Eiffel Tower is located in Berlin since 1889.'
    result = assess_submission(article, [review('True')])
    assert len(result['claims']) == 2
    assert result['claims'][0]['verdict'] == 'true'
    assert result['claims'][1]['verdict'] == 'unverified'
    assert result['verdict'] == 'unverified'


def test_article_claim_limit_cannot_hide_unassessed_statements():
    article = ' '.join([CLAIM] * 6)
    result = assess_submission(article, [review()])
    assert len(result['claims']) == 5
    assert result['unassessed_claim_count'] == 1
    assert result['verdict'] == 'unverified'


def test_negation_is_kept_when_matching_fact_checked_claims():
    claimed = 'Apollo 11 mission did not land on the Moon in 1969.'
    evidence = [review('True', CLAIM)]
    assert assess_submission(claimed, evidence)['verdict'] == 'unverified'


def test_close_review_cannot_transfer_rating_across_negation_or_changed_numbers():
    claim = 'The Riverford bridge opened to the public in January 1985.'
    evidence = [review('True', claim)]
    assert assess_submission(claim, evidence)['verdict'] == 'true'
    for changed in [claim.replace('opened', 'never opened'),
                    claim.replace('opened', "didn't open"),
                    claim.replace('1985', '1986')]:
        assert assess_submission(changed, evidence)['verdict'] == 'unverified'


def test_short_statement_cannot_disappear_between_longer_supported_claims():
    article = CLAIM + ' The Earth is flat. ' + CLAIM
    result = assess_submission(article, [review('True')])
    assert len(result['claims']) == 3
    assert result['claims'][1]['text'] == 'The Earth is flat.'
    assert result['claims'][1]['verdict'] == 'unverified'
    assert result['verdict'] == 'unverified'


def test_short_statements_past_claim_limit_are_counted_as_unassessed():
    result = assess_submission(' '.join([CLAIM] * 5) + ' The Earth is flat.', [review('True')])
    assert result['unassessed_claim_count'] == 1
    assert result['verdict'] == 'unverified'


def test_text_without_evidence_still_asks_gemini_for_an_unverified_reason():
    verification = {'provider': 'Public search', 'provider_status': 'searched', 'evidence': []}
    response = model_response('unverified', summary='No relevant source excerpts were found, so the claim remains unverified.',
                              claim_assessments=[{'verdict': 'unverified',
                                  'reason': 'No retrieved source addresses this claim.',
                                  'supporting_source_ids': [], 'contradicting_source_ids': []}], summary_ids=[])
    with patch('backend.report_summary.requests.post', return_value=response) as post:
        result = summarize_report(CLAIM, 'text', {}, verification, api_key='test-gemini-key-12345')
    post.assert_called_once()
    prompt = post.call_args.kwargs['json']['contents'][0]['parts'][0]['text']
    assert 'No relevant source excerpts were retrieved' in prompt
    assert result['verdict'] == 'unverified'
    assert result['summary'].startswith('No relevant source excerpts')
    assert result['gemini_analysis']['status'] == 'completed'
    assert result['sources'] == []


def test_non_text_without_evidence_does_not_call_gemini():
    verification = {'provider': 'Public search', 'provider_status': 'searched', 'evidence': []}
    with patch('backend.report_summary.requests.post') as post:
        result = summarize_report(CLAIM, 'image', {}, verification, api_key='test-gemini-key-12345')
    post.assert_not_called()
    assert result['verdict'] == 'unverified'
    assert result['gemini_analysis']['status'] == 'not_called_no_evidence'
    assert result['sources'] == []


def test_gemini_can_assess_relevant_reporting_when_no_published_review_exists():
    evidence = [{'kind': 'related_reporting', 'title': 'Apollo 11 overview', 'publisher': 'Reuters',
                 'url': 'https://www.reuters.com/world/example',
                 'snippet': 'Apollo 11 landed on the Moon in July 1969.'}]
    response = model_response('true', support=[1])
    with patch('backend.report_summary.requests.post', return_value=response) as post:
        result = summarize_report(CLAIM, 'text', {}, {'provider_status': 'searched', 'evidence': evidence}, 'test-gemini-key-12345')
    prompt = post.call_args.kwargs['json']['contents'][0]['parts'][0]['text']
    assert 'not a verdict you must repeat' in prompt
    assert result['verdict'] == 'true'
    assert result['supporting_evidence'] == [1]
    assert result['gemini_analysis']['status'] == 'completed'
    assert result['provider'] == 'Gemini evidence synthesis'


def test_gemini_can_mark_false_when_retrieved_source_directly_contradicts_claim():
    evidence = [{'kind': 'related_reporting', 'title': 'Apollo 11 correction', 'publisher': 'Reuters',
                 'url': 'https://www.reuters.com/world/example',
                 'snippet': 'Apollo 11 did not land on the Moon in 1969.'}]
    response = model_response('false', contradict=[1], summary='The source directly contradicts the claim [1].')
    with patch('backend.report_summary.requests.post', return_value=response):
        result = summarize_report(CLAIM, 'text', {}, {'provider_status': 'searched', 'evidence': evidence}, 'test-gemini-key-12345')
    assert result['verdict'] == 'false'
    assert result['contradicting_evidence'] == [1]
    assert result['claims'][0]['verdict'] == 'false'


def test_gemini_verdict_without_matching_citations_is_rejected():
    response = model_response('true', support=[])
    evidence = [{'kind': 'related_reporting', 'title': 'Apollo 11 overview', 'publisher': 'Reuters',
                 'url': 'https://www.reuters.com/world/example', 'snippet': 'Apollo 11 mission coverage.'}]
    with patch('backend.report_summary.requests.post', return_value=response):
        result = summarize_report(CLAIM, 'text', {}, {'provider_status': 'searched', 'evidence': evidence}, 'test-gemini-key-12345')
    assert result['verdict'] == 'unverified'
    assert result['gemini_analysis']['status'] == 'response_rejected'


def test_unverified_insufficient_evidence_summary_can_have_no_citations():
    response = model_response('unverified', summary='The retrieved excerpts do not settle the claim.')
    with patch('backend.report_summary.requests.post', return_value=response):
        result = summarize_report(CLAIM, 'text', {}, {'evidence': [review()]}, 'test-gemini-key-12345')
    assert result['gemini_analysis']['status'] == 'completed'
    assert result['verdict'] == 'unverified'
    assert result['summary_source_ids'] == []


def test_invented_summary_citation_still_rejected_for_unverified_claim():
    response = model_response('unverified', summary_ids=[99])
    with patch('backend.report_summary.requests.post', return_value=response):
        result = summarize_report(CLAIM, 'text', {}, {'evidence': [review()]}, 'test-gemini-key-12345')
    assert result['gemini_analysis']['status'] == 'response_rejected'
    assert result['gemini_diagnostics']['validation_error'] == 'summary_citation_ids_invalid'


def test_gemini_must_assess_every_claim_and_aggregate_consistently():
    claim = CLAIM + ' The Eiffel Tower is located in Berlin since 1889.'
    response = model_response('unverified', support=[1], claim_assessments=[
        {'verdict': 'true', 'reason': 'Source [1] supports this claim.',
         'supporting_source_ids': [1], 'contradicting_source_ids': []},
        {'verdict': 'unverified', 'reason': 'The source does not address this claim.',
         'supporting_source_ids': [], 'contradicting_source_ids': []},
    ])
    evidence = [{'kind': 'related_reporting', 'title': 'Apollo 11', 'publisher': 'Reuters',
                 'url': 'https://www.reuters.com/world/example',
                 'snippet': 'Apollo 11 landed on the Moon in July 1969.'}]
    with patch('backend.report_summary.requests.post', return_value=response):
        result = summarize_report(claim, 'text', {}, {'provider_status': 'searched', 'evidence': evidence}, 'test-gemini-key-12345')
    assert result['verdict'] == 'unverified'
    assert [item['verdict'] for item in result['claims']] == ['true', 'unverified']
    assert result['gemini_analysis']['status'] == 'completed'


def test_gemini_summary_cannot_drop_valid_source_ids_or_change_verdict():
    response = model_response('true', support=[1], summary='A published review rates this claim true [1].')
    with patch('backend.report_summary.requests.post', return_value=response):
        result = summarize_report(CLAIM, 'text', {}, {'checked_at': '2026-10-19T00:00:00Z',
            'provider_status': 'searched', 'evidence': [review()]}, 'test-gemini-key-12345')
    assert result['verdict'] == 'true'
    assert result['summary_source_ids'] == [1]
    assert result['gemini_analysis']['status'] == 'completed'


def test_historical_primary_source_can_settle_without_hardcoded_claim_rule():
    claim = "India's Constitution came into effect on 26 January 1950."
    source = {'kind':'related_reporting','title':'The Constitution of India',
        'publisher':'Legislative Department, Government of India',
        'url':'https://legislative.gov.in/constitution-of-india',
        'snippet':'The Constitution came into force on 26 January 1950.',
        'reliability':'official_primary_source'}
    response = model_response('true', support=[1],
        summary='The official legal source says the Constitution came into force on 26 January 1950 [1].')
    with patch('backend.report_summary.requests.post', return_value=response) as post:
        result = summarize_report(claim, 'text', {}, {'provider_status':'searched','evidence':[source]},
                                  'test-gemini-key-12345')
    prompt = post.call_args.kwargs['json']['contents'][0]['parts'][0]['text']
    assert 'https://legislative.gov.in/constitution-of-india' in prompt
    assert 'official_primary_source' in prompt
    assert result['verdict'] == 'true'
    assert result['summary_source_ids'] == [1]


def test_gemini_request_uses_structured_output_and_validates_api_model_config():
    response = model_response('true', support=[1])
    source = review()
    with patch('backend.report_summary.requests.post', return_value=response) as post:
        result = summarize_report(CLAIM, 'text', {}, {'provider_status':'searched','evidence':[source]},
                                  'test-gemini-key-12345')
    request_json = post.call_args.kwargs['json']
    assert 'temperature' not in request_json['generationConfig']
    assert request_json['generationConfig']['responseFormat']['text']['mimeType'] == 'APPLICATION_JSON'
    schema = request_json['generationConfig']['responseFormat']['text']['schema']
    assert schema['properties']['claim_assessments']['minItems'] == 1
    assert schema['properties']['verdict']['enum'] == ['true','false','unverified']
    assert result['gemini_analysis']['status'] == 'completed'
    with patch('backend.report_summary.requests.post') as post:
        invalid = summarize_report(CLAIM, 'text', {}, {'provider_status':'searched','evidence':[source]},
                                   'test-gemini-key-12345', model='../bad')
    post.assert_not_called()
    assert invalid['gemini_analysis']['status'] == 'invalid_model_config'


def test_gemini_quota_safety_and_empty_responses_have_distinct_statuses():
    quota = Mock(status_code=429)
    quota.raise_for_status.side_effect = requests.HTTPError(response=quota)
    quota.json.return_value = {'error':{'status':'RESOURCE_EXHAUSTED','message':'quota exceeded'}}
    with patch('backend.report_summary.requests.post', return_value=quota) as post:
        result = summarize_report(CLAIM, 'text', {}, {'provider_status':'searched','evidence':[review()]},
                                  'test-gemini-key-12345')
    assert post.call_count == 1
    assert result['gemini_analysis']['status'] == 'quota_exceeded'
    assert result['gemini_diagnostics']['http_status'] == 429

    safety = SimpleNamespace(raise_for_status=lambda: None,
        json=lambda: {'candidates':[{'finishReason':'SAFETY','content':{'parts':[]}}]})
    with patch('backend.report_summary.requests.post', return_value=safety) as post:
        result = summarize_report(CLAIM, 'text', {}, {'provider_status':'searched','evidence':[review()]},
                                  'test-gemini-key-12345')
    assert post.call_count == 1
    assert result['gemini_analysis']['status'] == 'safety_blocked'
    assert result['gemini_diagnostics']['finish_reason'] == 'SAFETY'

    empty = SimpleNamespace(raise_for_status=lambda: None,
        json=lambda: {'candidates':[{'finishReason':'STOP','content':{'parts':[]}}]})
    with patch('backend.report_summary.requests.post', return_value=empty):
        result = summarize_report(CLAIM, 'text', {}, {'provider_status':'searched','evidence':[review()]},
                                  'test-gemini-key-12345')
    assert result['gemini_analysis']['status'] == 'empty_response'
    assert result['gemini_diagnostics']['candidate_content_returned'] is False


def test_malformed_gemini_json_is_not_retried_and_uses_safe_fallback():
    malformed = SimpleNamespace(raise_for_status=lambda: None,
        json=lambda: {'candidates': [{'content': {'parts': [{'text': '{broken'}]}}]})
    with patch('backend.report_summary.requests.post', return_value=malformed) as post:
        result = summarize_report(CLAIM, 'text', {},
            {'provider_status': 'searched', 'evidence': [review()]}, 'test-gemini-key-12345')
    assert post.call_count == 1
    assert result['verdict'] == 'true'
    assert result['gemini_analysis']['status'] == 'malformed'
    assert result['gemini_diagnostics']['candidate_content_returned'] is True


def test_gemini_network_failure_preserves_evidence_assessment_and_reports_warning():
    with patch('backend.report_summary.requests.post', side_effect=requests.Timeout('timed out')) as post:
        result = summarize_report(CLAIM, 'text', {},
            {'provider_status': 'searched', 'evidence': [review()]}, 'test-gemini-key-12345')
    assert post.call_count == 2
    assert result['verdict'] == 'true'
    assert result['gemini_analysis']['status'] == 'timeout'
    assert result['gemini_diagnostics']['error_category'] == 'timeout'
    assert any('Gemini generation did not complete' in item for item in result['technical_warnings'])
