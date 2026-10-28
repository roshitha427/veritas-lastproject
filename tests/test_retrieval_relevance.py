"""Claim-level retrieval regressions with synthetic publisher passages.

These fixtures exercise evidence relevance, not model accuracy or real promotions.
"""
from types import SimpleNamespace

import pytest

from backend.evidence_assessment import split_candidate_claims, assess_submission
from backend.report_summary import filter_relevant_evidence, _source_relevance
from backend.source_routing import normalize_search_submission, route_sources
from backend.verification import merge_source_record, verify_submission


CLAIM = 'Elon Musk is giving away free Tesla vehicles to everyone in 2025.'


def source(path, passage, title='Elon Musk news', **kwargs):
    return {'url': 'https://www.reuters.com/' + path, 'publisher': 'Reuters',
            'title': title, 'snippet': passage, 'retrieval_status': 'article_excerpt_retrieved', **kwargs}


@pytest.mark.parametrize('page', [
    source('greetings', 'Elon Musk sent Happy New Year greetings for 2025 and thanked his followers.'),
    source('different-video', 'Elon Musk appeared in a video about Tesla vehicle manufacturing in 2025.'),
    source('programme', 'Elon Musk discusses Tesla cars, space and his business career on a BBC programme.'),
    source('headline-only', 'The programme follows a reporter through the factory.', title=CLAIM),
    source('metadata', CLAIM, title=CLAIM, retrieval_status='metadata_only'),
])
def test_same_person_or_matching_headline_cannot_replace_claim_specific_passage(page):
    report = filter_relevant_evidence(CLAIM, {'provider_status': 'searched', 'evidence': [page]})
    assert report['evidence'] == []
    assert report['status'] == 'unverified'
    checked = report['checked_sources'][0]
    assert checked['accepted'] is False
    assert checked['reason'] and '_' not in checked['reason']
    assert checked['claim_indices'] == [0]


@pytest.mark.parametrize('passage', [
    'Elon Musk is giving away free Tesla vehicles to everyone in 2025.',
    'Musk offered complimentary Tesla cars to all residents in a 2025 giveaway.',
    'The 2025 free Tesla promotion attributed to Elon Musk is a scam; no such giveaway was announced.',
])
def test_direct_passage_paraphrase_and_counterevidence_are_all_relevant(passage):
    report = filter_relevant_evidence(CLAIM, {'evidence': [source('promotion', passage)]})
    assert len(report['evidence']) == 1
    assert report['checked_sources'][0]['accepted'] is True
    # Relevance must not manufacture a TRUE/FALSE verdict.
    assert report.get('status') not in {'true', 'false', 'verified'}


def test_headline_mentioning_free_tesla_cars_still_requires_giveaway_evidence():
    headline = 'Elon Musk Announces Free Tesla Cars for Everyone in 2025'
    near_miss = source('different-video', 'Elon Musk appeared in a video about Tesla vehicle manufacturing in 2025.')
    assert not _source_relevance(headline, near_miss)['relevant']
    assert _source_relevance(headline, source('direct-promotion', CLAIM))['relevant']


def test_rejected_sources_and_reasons_survive_repeated_filtering_without_reintroduction():
    pages = [source('greetings', 'Elon Musk sent New Year greetings in 2025.'),
             source('promotion', CLAIM)]
    first = filter_relevant_evidence(CLAIM, {'evidence': pages})
    second = filter_relevant_evidence(CLAIM, first)
    assert second == first
    assert [row['url'] for row in second['evidence']] == [pages[1]['url']]
    assert len(second['checked_sources']) == 2
    assert second['filtered_irrelevant_count'] == 1


def test_numerical_counterevidence_remains_available_but_not_direct_support():
    match = _source_relevance('The population of Arborville is 120000.',
                             source('census', 'The population of Arborville is 150000.'))
    assert match['relevant'] is True
    assert match['values_match'] is False


def test_current_source_rejection_explains_missing_fresh_evidence():
    report = filter_relevant_evidence('Elon Musk is giving away free Tesla vehicles today.', {
        'evidence': [source('old-promotion', 'Elon Musk is giving away free Tesla vehicles.',
                            published_at='2020-01-01T00:00:00Z')]})
    assert report['evidence'] == []
    assert report['checked_sources'][0]['reason_code'] == 'not_current_enough'


@pytest.mark.parametrize('metadata_field', ['matched_excerpt', 'title'])
def test_official_office_search_metadata_cannot_rescue_an_unrelated_original_passage(metadata_field):
    claim = 'The current chief minister of Example Pradesh is Daya Wrong.'
    page = {'url': 'https://example.gov.in/irrigation',
            'title': 'Agricultural irrigation statistics',
            'snippet': 'Farm irrigation canals provide water to agricultural fields during summer.',
            'retrieval_status': 'article_excerpt_retrieved',
            metadata_field: 'The current chief minister of Example Pradesh is Bala Sample.'}
    match = _source_relevance(claim, page)
    assert match['relevant'] is False
    assert match['reason'] != 'office_and_jurisdiction_match'
    report = filter_relevant_evidence(claim, {'evidence': [page]})
    assert report['evidence'] == []
    assert report['checked_sources'][0]['accepted'] is False


def test_original_official_directory_can_supply_a_different_holder_as_counterevidence():
    claim = 'The current chief minister of Example Pradesh is Daya Wrong.'
    page = {'url': 'https://example.gov.in/directory', 'title': 'Department directory',
            'snippet': 'The current chief minister of Example Pradesh is Bala Sample.',
            'retrieval_status': 'article_excerpt_retrieved'}
    match = _source_relevance(claim, page)
    assert match['relevant'] is True
    assert match['reason'] == 'office_and_jurisdiction_match'


def test_ocr_banners_do_not_consume_claim_slots_and_wrapped_headline_is_reassembled():
    ocr = ('Global News\nBREAKING NEWS\nElon Musk Announces\nFree Tesla Cars for\n'
           'Everyone in 2025\nTech billionaire says he will give\napay free Tesla vehicles to all\n'
           'world citizens next year.\nFAKE NEWS\nGLOBAL NEWS')
    claims = split_candidate_claims(ocr)
    assert len(claims) == 1
    assert 'Elon Musk Announces Free Tesla Cars for Everyone in 2025' in claims[0]
    assert 'Global News' not in claims[0]
    # Query normalization must not silently rewrite uncertain OCR words.
    assert 'apay' in claims[0]
    report = filter_relevant_evidence(ocr, {'evidence': [source('programme', 'Global News programme on BBC.')]})
    assert report['evidence'] == []
    assert report['normalized_claim'] == normalize_search_submission(ocr)
    assert len(assess_submission(ocr, [])['claims']) == 1
    assert split_candidate_claims('The Earth is flat.\nThe Moon is a star.') == [
        'The Earth is flat.', 'The Moon is a star.']


def test_all_five_candidates_get_search_attempts_and_auditable_query_statuses(monkeypatch):
    candidates = ['Arborville opened a railway in 1985.', 'Birchford opened a school in 1986.',
                  'Cedarford opened a library in 1987.', 'Daleside opened a bridge in 1988.',
                  'Elmford opened a hospital in 1989.']
    called = []
    def search(claim, *args):
        called.append(claim)
        return {'provider': 'Tavily', 'provider_status': 'searched',
                'query_statuses': [{'query': claim, 'status': 'searched',
                                    'requested_domains': ['reuters.com'], 'request_attempted': True}],
                'evidence': [source(str(len(called)), claim)]}
    monkeypatch.setattr('backend.tavily_search.search_tavily', search)
    report = verify_submission(' '.join(candidates), 'en', [{'domain': 'reuters.com'}],
                               provider='tavily', tavily_key='fixture-only')
    assert called == candidates
    assert [row['claim_index'] for row in report['claim_searches']] == list(range(5))
    assert all(row['query_statuses'][0]['requested_domains'] == ['reuters.com'] for row in report['claim_searches'])
    assert [row['candidate_claim_indices'] for row in report['evidence']] == [[0], [1], [2], [3], [4]]


def test_source_merge_retains_new_snapshot_metadata_and_all_candidate_associations():
    old = source('same', CLAIM * 10, retrieved_at='2026-09-01T00:00:00Z',
                 published_at='2026-08-01T00:00:00Z', candidate_claim_indices=[0],
                 retrieval_method='cached_page')
    fresh = source('same', 'A new, shorter original page passage.', retrieved_at='2026-10-24T00:00:00Z',
                   published_at=None, candidate_claim_indices=[4], retrieval_method='direct_official_page_fetch')
    merged = merge_source_record(old, fresh)
    assert merged['snippet'] == fresh['snippet']
    assert merged['retrieved_at'] == fresh['retrieved_at']
    assert merged['published_at'] is None
    assert merged['retrieval_method'] == fresh['retrieval_method']
    assert merged['candidate_claim_indices'] == [0, 4]
    assert old['candidate_claim_indices'] == [0]


def test_tavily_audit_records_actual_requested_domains_not_all_configured_publishers(monkeypatch):
    from backend.tavily_search import search_tavily
    requests = []
    def post(*args, **kwargs):
        requests.append(kwargs['json'])
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None,
                               json=lambda: {'results': []})
    monkeypatch.setattr('backend.tavily_search.requests.post', post)
    sources = [{'domain': 'gov.in'}, {'domain': 'reuters.com'}, {'domain': 'bbc.com'}]
    report = search_tavily('The government opened a bridge in 1985.', sources, 'audit-domain-fixture')
    assert len(requests) == 2
    by_query = {row['query']: row for row in report['query_statuses']}
    for request in requests:
        assert by_query[request['query']]['requested_domains'] == request['include_domains']
    assert any(row['requested_domains'] == ['gov.in'] for row in by_query.values())
    assert report.get('checked_sources', []) == []


def test_promotional_search_uses_core_claim_and_claim_specific_follow_up():
    route = route_sources('Is it true that ' + CLAIM, [{'domain': 'reuters.com'}])
    assert route['topic'] == 'promotion'
    assert route['queries'][0] == CLAIM
    assert route['queries'][1].endswith('fact check official announcement')


@pytest.mark.parametrize('passage', [
    'Rahul Gandhi discussed his faith and questioned the caste claims made about a political rival.',
    'A man standing behind Rahul Gandhi was falsely accused of slapping Hardik Patel.',
    'Questioning Priyanka Gandhi about her affidavit, a spokesperson mentioned the National Herald case.',
])
def test_questioning_requires_the_same_person_action_and_named_case(passage):
    claim = 'Rahul Gandhi was questioned by ED in the National Herald case.'
    report = filter_relevant_evidence(claim, {'evidence': [source('other-event', passage)]})
    assert report['evidence'] == []
    assert report['checked_sources'][0]['reason_code'] == 'different_event_or_predicate'


def test_unrelated_article_cannot_use_exact_claim_from_explicit_related_card_as_evidence():
    claim = 'Rahul Gandhi was questioned by ED in the National Herald case.'
    page = source('dengue', 'Doctors described the treatment of dengue and cautioned against unsupported remedies.\n'
                  'Related Fact Check\n' + claim,
                  title='This medicine cannot cure dengue in 48 hours')
    report = filter_relevant_evidence(claim, {'evidence': [page]})
    assert report['evidence'] == []
    assert claim not in report['checked_sources'][0]['excerpt']
    assert report['checked_sources'][0]['accepted'] is False


def test_unmarked_distant_card_does_not_override_different_article_title_and_lead():
    claim = 'Rahul Gandhi was questioned by ED in the National Herald case.'
    page = source('dengue', ('Doctors described the treatment of dengue and cautioned against unsupported remedies. ' * 9)
                  + claim, title='This medicine cannot cure dengue in 48 hours')
    report = filter_relevant_evidence(claim, {'evidence': [page]})
    assert report['evidence'] == []
    assert report['checked_sources'][0]['reason_code'] == 'different_article_topic'


def test_main_claim_passage_and_generic_primary_record_remain_available():
    claim = 'Rahul Gandhi was questioned by ED in the National Herald case.'
    passage = 'The Enforcement Directorate interrogated Rahul Gandhi in connection with the National Herald case.'
    assert _source_relevance(claim, source('case', passage,
        title='National Herald investigation: Rahul Gandhi appears before agency'))['relevant']
    primary = dict(source('case', passage, title='Department records and publications'),
                   url='https://fixture.gov.in/records', reliability='official_primary_source')
    assert _source_relevance(claim, primary)['relevant']
    ordinary = source('case', passage + ' The lawyer recommended caution within the case discussion.')
    report = filter_relevant_evidence(claim, {'evidence': [ordinary]})
    assert report['evidence'][0]['snippet'] == ordinary['snippet']


def test_related_cards_removed_from_accepted_source_before_any_evidence_model_sees_it():
    claim = 'Rahul Gandhi was questioned by ED in the National Herald case.'
    page = source('case', claim + '\nRelated Stories\nAn unrelated headline.', title='National Herald case update')
    report = filter_relevant_evidence(claim, {'evidence': [page]})
    assert report['evidence'][0]['snippet'] == claim
    assert report['evidence'][0]['excluded_page_sections'] == 'related_or_recommended_content'
