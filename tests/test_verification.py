from unittest.mock import Mock, patch
from types import SimpleNamespace
import pytest
import requests
from backend.verification import trusted_url, verify, verify_submission, gdelt_query, verify_wikipedia


SOURCES=[{'domain':'reuters.com','enabled':True}]


def test_trusted_domains_require_https_and_real_subdomain():
    assert trusted_url('https://www.reuters.com/world/example',SOURCES)
    assert not trusted_url('https://reuters.com.evil.test',SOURCES)
    assert not trusted_url('http://reuters.com',SOURCES)
    assert not trusted_url('https://reuters.com', [{'domain':'reuters.com','enabled':False}])


def test_official_office_headings_precede_generic_location_paragraphs(monkeypatch):
    from backend.verification import _article_excerpt
    paragraph = 'Example Pradesh has many historic buildings and parks that welcome visitors throughout the year. '
    html = ('<title>Example Pradesh Legislative Assembly</title>'
            '<meta property="article:published_time" content="2026-10-27T00:00:00Z">'
            '<nav><h2>Chief Minister obsolete navigation</h2></nav>'
            '<p>' + paragraph * 30 + '</p>'
            '<h2>Deputy Chief Minister</h2><h3>Bala Sample</h3>'
            '<h2>Chief Minister</h2><h3>Aruna Example</h3>'
            '<h2>Speaker</h2><h3>Chandra Fictional</h3>')
    calls = []
    def get(url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(status_code=200, headers={'Content-Type': 'text/html'}, encoding='utf-8',
                               close=lambda: None, iter_content=lambda chunk_size: [html.encode()])
    monkeypatch.setattr('backend.verification.requests.get', get)
    excerpt, published = _article_excerpt('https://example.gov.in/leadership',
                                          'Example Pradesh chief minister', [{'domain': 'gov.in'}])
    assert excerpt.startswith('Example Pradesh Legislative Assembly')
    assert 'Deputy Chief Minister Bala Sample Chief Minister Aruna Example' in excerpt
    assert 'obsolete navigation' not in excerpt
    assert 'Daya Wrong' not in excerpt
    assert len(excerpt) <= 1800 and published == '2026-10-27T00:00:00Z'
    assert len(calls) == 1 and calls[0][1]['allow_redirects'] is False


@pytest.mark.parametrize('role', ['Deputy Chief Minister', 'Vice Chief Minister',
                                 'Former Chief Minister', 'Ex-Chief Minister'])
def test_other_office_roles_are_not_selected_as_the_chief_minister_heading(role):
    from backend.verification import _ArticleParser, _office_heading_excerpt
    parser = _ArticleParser()
    parser.feed(f'<title>Example Pradesh leadership</title><h2>{role}</h2><h3>Bala Sample</h3>')
    assert _office_heading_excerpt(parser, 'Example Pradesh chief minister') is None


def test_official_page_extraction_rejects_redirects_and_disabled_sources(monkeypatch):
    from backend.verification import _article_excerpt
    calls = []
    def get(url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(status_code=302, headers={'Location': 'https://untrusted.example/leadership'},
                               close=lambda: None)
    monkeypatch.setattr('backend.verification.requests.get', get)
    url = 'https://example.gov.in/leadership'
    assert _article_excerpt(url, 'Example Pradesh chief minister', [{'domain': 'gov.in'}]) == (None, None)
    assert _article_excerpt(url, 'Example Pradesh chief minister',
                            [{'domain': 'gov.in'}, {'domain': 'example.gov.in', 'enabled': False}]) == (None, None)
    assert len(calls) == 1 and calls[0][1]['allow_redirects'] is False


def test_search_match_is_evidence_for_review_not_automatic_verification():
    fixture = {'status':'needs_review', 'provider_status':'searched', 'evidence':[
        {'claim':'A different claim', 'url':'https://reuters.com/fact-check/example',
         'title':'Related review', 'rating':'False'}]}
    with patch('backend.factcheck_archives.verify_public', return_value=fixture), \
         patch('backend.verification.verify_wikipedia',return_value={'name':'Reference', 'status':'unavailable'}), \
         patch('backend.verification._article_excerpt',return_value=(None,None)):
        result=verify('Some unverified claim','en',SOURCES,provider='public')
    assert result['status']=='needs_review'
    assert result['evidence'][0]['rating']=='False'
    assert result['provider_status']=='searched'



def test_no_api_key_is_explicit():
    result=verify('Some claim','en',SOURCES,provider='none')
    assert result['status']=='unverified' and result['evidence']==[]
    assert result['provider_status']=='not_configured'


def test_keyless_reporting_has_no_fact_check_rating_or_verified_verdict():
    response=Mock()
    response.json.return_value={'articles':[{'url':'https://www.reuters.com/world/example','title':'Related reporting',
                                          'seendate':'20261018T090000Z'}]}
    with patch('backend.verification.requests.get',return_value=response) as get, \
         patch('backend.verification._gdelt_last_request',0), \
         patch('backend.verification._gdelt_cache',{}), \
         patch('backend.verification._article_excerpt',return_value=(None,None)):
        result=verify('India election results','en',SOURCES,provider='gdelt')
    assert result['provider']=='GDELT DOC 2.0'
    assert result['status']=='needs_review'
    assert result['evidence'][0]['kind']=='related_reporting'
    assert result['evidence'][0]['rating']==''
    assert 'key' not in get.call_args.kwargs['params']


def test_gdelt_doc_search_uses_a_three_month_window_and_source_record():
    response=Mock()
    response.json.return_value={'articles':[{'url':'https://www.reuters.com/world/example','title':'Andhra Pradesh cabinet update',
        'seen_at':'2026-10-18T09:00:00Z','context':'The state government said the Chief Minister would review the project.'}]}
    with patch('backend.verification.requests.get',return_value=response) as get, \
         patch('backend.verification._gdelt_last_request',0), \
         patch('backend.verification._gdelt_cache',{}), \
         patch('backend.verification._article_excerpt',return_value=(None,None)):
        result=verify('Andhra Pradesh Chief Minister reviews project','en',SOURCES,provider='gdelt')
    assert '/doc/doc' in get.call_args.args[0]
    assert get.call_args.kwargs['params']['timespan']=='3months'
    assert result['evidence'][0]['snippet'].startswith('The state government')
    assert result['evidence'][0]['publisher']=='www.reuters.com'
    assert result['evidence'][0]['rating']==''


def test_wikipedia_search_is_labeled_as_reference_not_fact_check():
    response=Mock()
    response.json.return_value={'query':{'search':[{'pageid':1159,'title':'Andhra Pradesh',
        'snippet':'Andhra Pradesh is a state in <span>India</span>.'}]}}
    with patch('backend.verification.requests.get',return_value=response):
        result=verify_wikipedia('Who is Chief Minister of Andhra Pradesh?')
    assert result['status']=='searched'
    assert result['evidence'][0]['kind']=='reference_article'
    assert 'not a fact-check verdict' in result['evidence'][0]['relationship']
    assert result['evidence'][0]['snippet']=='Andhra Pradesh is a state in India.'


def test_keyless_search_filters_untrusted_domains_and_invalid_urls():
    response=Mock()
    response.json.return_value={'articles':[{'url':'https://reuters.com.evil.test/example','title':'Bad link'},
                                          {'url':'http://www.reuters.com/example','title':'HTTP link'}]}
    with patch('backend.verification.requests.get',return_value=response), \
         patch('backend.verification._gdelt_last_request',0), \
         patch('backend.verification._gdelt_cache',{}):
        result=verify('Moon landing example','en',SOURCES,provider='gdelt')
    assert result['evidence']==[] and result['status']=='unverified'


def test_gdelt_outage_falls_back_to_public_fact_check_search():
    fallback = {'status':'needs_review', 'provider':'Public fact-check search',
        'provider_status':'searched', 'evidence':[{'url':'https://reuters.com/fact-check/example'}],
        'providers':[{'name':'Reuters archive', 'status':'searched'}],
        'explanation':'A related fact-check was found; review its original verdict.'}
    with patch('backend.verification.requests.get', side_effect=requests.Timeout), \
         patch('backend.verification._gdelt_last_request', 0), \
         patch('backend.verification._gdelt_cache', {}), \
         patch('backend.verification._article_excerpt',return_value=(None,None)), \
         patch('backend.factcheck_archives.verify_public', return_value=fallback):
        result=verify('India election results','en',SOURCES,provider='gdelt')
    assert result['provider']=='GDELT with public archive/RSS fallback'
    assert result['primary_provider_status']=='unavailable'
    assert result['providers'][0]=={'name':'GDELT DOC 2.0','status':'unavailable'}
    assert [item['url'] for item in result['evidence']]==[item['url'] for item in fallback['evidence']]
    assert result['status']=='needs_review'


def test_empty_recent_search_falls_back_to_archive_and_publisher_feeds():
    response=Mock()
    response.json.return_value={'articles':[]}
    fallback={'status':'needs_review','provider':'Public fact-check search','provider_status':'searched',
        'evidence':[{'kind':'fact_check_article','title':'Older review','publisher':'Reuters',
                     'url':'https://reuters.com/fact-check/older','snippet':'The claim was reviewed in context.'}],
        'providers':[{'name':'Reuters archive','status':'searched'}]}
    with patch('backend.verification.requests.get',return_value=response), \
         patch('backend.verification._gdelt_last_request',0), \
         patch('backend.verification._gdelt_cache',{}), \
         patch('backend.verification._article_excerpt',return_value=(None,None)), \
         patch('backend.factcheck_archives.verify_public',return_value=fallback):
        result=verify('India election results','en',SOURCES,provider='gdelt')
    assert result['status']=='needs_review'
    assert result['provider']=='GDELT with public archive/RSS fallback'
    assert result['evidence'][0]['url']=='https://reuters.com/fact-check/older'
    assert result['providers'][0]=={'name':'GDELT DOC 2.0','status':'searched'}


def test_submission_searches_each_independent_claim_and_tracks_sources():
    calls=[]
    def search(claim,*args,**kwargs):
        calls.append(claim)
        return {'provider':'GDELT DOC 2.0','provider_status':'searched',
                'checked_at':'2026-10-19T00:00:00Z','search_query':claim,
                'evidence':[{'kind':'related_reporting','url':f'https://www.reuters.com/{len(calls)}',
                    'title':claim,'snippet':claim,'publisher':'Reuters'}]}
    text='India adopted its Constitution in 1950. The Eiffel Tower opened in 1889.'
    with patch('backend.verification.verify',side_effect=search):
        result=verify_submission(text,'en',SOURCES,provider='gdelt')
    assert len(calls)==2
    assert result['provider_status']=='searched'
    assert [item['claim_index'] for item in result['claim_searches']]==[0,1]
    assert [item['candidate_claim_indices'] for item in result['evidence']]==[[0],[1]]


def test_keyless_query_uses_only_sanitized_terms_and_enabled_source_domains():
    query=gdelt_query('India election domain:evil.test (OR malware)',SOURCES)
    assert 'domain:reuters.com' in query and 'domain:evil.test' not in query
    primary_query=gdelt_query("India's Constitution came into effect on 26 January 1950", [
        {'domain':'legislative.gov.in','enabled':True}])
    assert '1950' in primary_query and '26' in primary_query
    assert gdelt_query('ఈ వార్త నిజమేనా',SOURCES) is None
