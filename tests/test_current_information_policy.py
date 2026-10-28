from backend.claim_context import normalize_places, needs_current_sources, source_can_settle_current_claim
from backend.report_summary import apply_knowledge_policy
from backend.source_routing import route_sources
from backend.store import now


def test_current_offices_and_death_claims_need_live_sources_but_history_does_not():
    assert needs_current_sources('current tamilnadu chief minister vijay is dead')
    assert needs_current_sources('Who is the chief minister of Telangana?')
    assert needs_current_sources('Actor Example is alive')
    assert not needs_current_sources('Example was the chief minister in 1995')
    assert not needs_current_sources('The Constitution came into effect on 26 January 1950')


def test_state_aliases_and_routes_preserve_jurisdiction():
    sources = [{'domain': 'gov.in'}, {'domain': 'nic.in'}]
    assert normalize_places('current AP cm') == 'current andhra pradesh cm'
    assert route_sources('current AP chief minister', sources)['domains'] == ['ap.gov.in', 'apa.neva.gov.in']
    assert route_sources('current tamilnadu chief minister vijay is dead', sources)['jurisdiction'] == 'tamil nadu'
    assert route_sources('Telangana government announced a scheme', sources)['domains'] == ['telangana.gov.in']
    assert route_sources('Karnataka chief minister is Example', sources)['topic'] == 'government'


def test_chief_minister_typo_routes_to_current_office_sources_without_inventing_holder():
    from backend.source_routing import directory_urls
    sources = [{'domain': 'gov.in'}, {'domain': 'nic.in'}]
    claim = 'current AP cheif minister is Daya Wrong'
    normalized = normalize_places(claim)
    assert normalized == 'current andhra pradesh chief minister is Daya Wrong'
    plan = route_sources(claim, sources)
    assert plan['topic'] == 'government'
    assert plan['jurisdiction'] == 'andhra pradesh'
    assert plan['domains'] == ['ap.gov.in', 'apa.neva.gov.in']
    assert any('andhra pradesh chief minister current official' in query for query in plan['queries'])
    assert 'https://apa.neva.gov.in/Home/' in directory_urls(claim)
    assert needs_current_sources('AP cheif minister is Daya Wrong')
    assert not needs_current_sources('Daya Wrong was AP cheif minister in 1995')


def test_misspelled_office_accepts_fresh_official_counterevidence_and_rejects_stale_snapshots():
    from backend.report_summary import _source_relevance
    claim = 'current Example Pradesh cheif minister is Daya Wrong'
    source = {'title': 'Example Pradesh leadership', 'url': 'https://example.gov.in/leadership',
              'reliability': 'official_primary_source', 'retrieval_status': 'article_excerpt_retrieved',
              'retrieved_at': now(), 'snippet': 'Example Pradesh leadership directory. '
              'Aruna Example is the Chief Minister. Bala Sample is the Deputy Chief Minister.'}
    relevance = _source_relevance(claim, source)
    assert relevance['relevant']
    assert relevance['reason'] == 'office_and_jurisdiction_match'
    assert source_can_settle_current_claim(claim, source)
    stale = dict(source, retrieval_status='imported_source_excerpt', retrieved_at='2020-01-01', fetched_at=now())
    assert not source_can_settle_current_claim(claim, stale)
    # Relevance is not a verdict and does not replace the submitted name.
    assert 'verdict' not in relevance
    assert 'Daya Wrong' not in source['snippet']


def test_science_and_institution_routes_respect_disabled_sources():
    sources = [{'domain': d} for d in ['nasa.gov', 'isro.gov.in', 'who.int', 'gov.in', 'ac.in']]
    sources.append({'domain': 'nasa.gov', 'enabled': False})
    science = route_sources('The Earth is flat', sources)
    assert science['topic'] == 'space_and_earth_science'
    assert 'nasa.gov' not in science['domains']
    assert 'isro.gov.in' in science['domains']
    assert route_sources('Dengue can be cured with antibiotics', sources)['domains'] == ['who.int', 'icmr.gov.in', 'mohfw.gov.in']
    assert route_sources('Riverford College announced a holiday', sources)['topic'] == 'institution'
    assert route_sources('Example Hospital announced a new department', sources)['topic'] == 'institution'


def test_office_listing_or_old_debunk_cannot_settle_a_new_death_claim():
    claim = 'current tamilnadu chief minister vijay is dead'
    assert not source_can_settle_current_claim(claim, {'source_role': 'office_directory',
        'snippet': 'Vijay is Chief Minister', 'published_at': now()})
    assert not source_can_settle_current_claim(claim, {'snippet': 'The rumour of his death is false.',
        'published_at': '2020-01-01', 'retrieved_at': now(), 'retrieval_status': 'article_excerpt_retrieved'})
    assert source_can_settle_current_claim(claim, {'snippet': 'An official statement says the death rumour is false.',
        'published_at': now()})


def test_saved_stale_knowledge_is_removed_without_changing_evidence_verdict():
    old = {'schema_version': 5, 'verdict': 'unverified',
           'gemini_model_assessment': {'status': 'completed', 'verdict': 'false',
               'answer': 'Vijay is alive and Stalin is the current Chief Minister.', 'reason': 'Old model knowledge'}}
    corrected = apply_knowledge_policy(old, 'current tamilnadu chief minister vijay is dead')
    assert corrected['verdict'] == 'unverified'
    assert corrected['gemini_model_assessment']['status'] == 'needs_current_sources'
    assert not corrected['gemini_model_assessment']['answer']
    assert 'Stalin' not in str(corrected)
    assert old['gemini_model_assessment']['answer']  # do not mutate input records


def test_stable_background_explanation_remains_available():
    answer = {'status': 'completed', 'answer': 'Water consists of hydrogen and oxygen.', 'reason': 'Basic chemistry.'}
    assert apply_knowledge_policy({'verdict': 'unverified', 'gemini_model_assessment': answer},
                                 'What is water made of?')['gemini_model_assessment'] == answer


def test_settled_current_verdict_does_not_show_a_conflicting_unverified_panel():
    summary = apply_knowledge_policy({'verdict':'true', 'gemini_model_assessment':{
        'status':'completed', 'answer':'Old model answer', 'reason':'Stored knowledge'}},
        'Who is the current chief minister?')
    assert summary['gemini_model_assessment'] is None


def test_official_profile_reads_adjacent_name_and_role_headings_not_navigation():
    from backend.verification import _ArticleParser, _office_heading_excerpt
    parser = _ArticleParser()
    parser.feed('<title>Example Pradesh Government</title><nav>Chief Minister old navigation</nav>'
                '<h1>Chief Minister</h1><h2>Aruna Fictional</h2><h3>Honourable Chief Minister</h3>'
                '<footer>Old copyright and menu</footer>')
    excerpt = _office_heading_excerpt(parser, 'Example Pradesh chief minister')
    assert 'Aruna Fictional' in excerpt
    assert 'old navigation' not in excerpt
    assert 'Old copyright' not in excerpt
