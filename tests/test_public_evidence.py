from unittest.mock import MagicMock, Mock, patch
from backend.public_evidence import read_feed, verify_rss
from backend.factcheck_archives import verify_public

SOURCES=[{'domain':'altnews.in','enabled':True},{'domain':'factly.in','enabled':True},{'domain':'bbc.com','enabled':True}]


def test_public_archive_returns_related_fact_check_without_inventing_rating():
    response=Mock()
    response.content=b'[]'
    response.json.return_value=[{'link':'https://www.altnews.in/example-moon-landing/',
        'title':{'rendered':'<strong>NASA moon landing</strong> claim examined'},
        'excerpt':{'rendered':'<p>Context about the moon landing</p>'},'date_gmt':'2026-10-18T09:00:00'}]
    with patch('backend.factcheck_archives.requests.get',return_value=response) as get, \
         patch('backend.factcheck_archives._cache',{}):
        result=verify_public('NASA moon landing',SOURCES)
    assert result['status']=='needs_review' and result['provider_status']=='searched'
    assert result['evidence'][0]['kind']=='fact_check_article'
    assert result['evidence'][0]['rating']==''
    assert '<strong>' not in result['evidence'][0]['title']
    assert 'key' not in get.call_args.kwargs['params']


def test_public_archive_failure_falls_back_to_recent_feeds():
    fallback={'provider_status':'searched','evidence':[{'url':'https://www.bbc.com/news/example',
        'kind':'related_reporting','title':'Moon landing report','publisher':'BBC','rating':''}], 'fetched_at':'2026-10-18T09:00:00Z'}
    with patch('backend.factcheck_archives.search_archive',return_value=None), \
         patch('backend.factcheck_archives.verify_rss',return_value=fallback), \
         patch('backend.factcheck_archives._cache',{}):
        result=verify_public('Moon landing report',SOURCES)
    assert result['status']=='needs_review'
    assert result['evidence'][0]['kind']=='related_reporting'
    assert result['providers'][0]['status']=='unavailable'
    assert result['providers'][-1]['status']=='searched'


def test_public_search_no_matches_remains_unverified():
    fallback={'provider_status':'searched','evidence':[]}
    with patch('backend.factcheck_archives.search_archive',return_value=[]), \
         patch('backend.factcheck_archives.verify_rss',return_value=fallback), \
         patch('backend.factcheck_archives._cache',{}):
        result=verify_public('Example unknown claim',SOURCES)
    assert result['status']=='unverified' and not result['evidence']
    assert result['provider_status']=='searched'


def test_rss_parser_reads_only_metadata_and_links():
    xml=b'''<?xml version="1.0"?><rss><channel><item><title>NASA moon landing explained</title>
    <link>https://www.bbc.com/news/example</link><description>Related public reporting</description>
    <pubDate>Sun, 18 Oct 2026 09:00:00 GMT</pubDate></item></channel></rss>'''
    context=MagicMock()
    response=context.__enter__.return_value
    response.iter_content.return_value=[xml]
    with patch('backend.public_evidence.requests.get',return_value=context):
        records=read_feed(('BBC','bbc.com','https://feeds.bbci.co.uk/news/world/rss.xml'))
    assert records[0]['title']=='NASA moon landing explained'
    assert records[0]['reviewed_at']=='2026-10-18T09:00:00+00:00'
    assert records[0]['rating']=='' and records[0]['kind']=='related_reporting'


def test_rss_cache_does_not_bypass_disabled_sources():
    record={'kind':'related_reporting','title':'NASA moon landing','url':'https://www.bbc.com/news/example',
        'publisher':'BBC','rating':'','_keywords':{'nasa','moon','landing'},'_title_keywords':{'nasa','moon','landing'}}
    with patch('backend.public_evidence.read_feed',return_value=[record]) as fetch, \
         patch('backend.public_evidence._cache',{}):
        first=verify_rss('NASA moon landing',[{'domain':'bbc.com','enabled':True}])
        second=verify_rss('NASA moon landing',[{'domain':'bbc.com','enabled':True}])
        disabled=verify_rss('NASA moon landing',[{'domain':'bbc.com','enabled':False}])
    assert first['status']=='needs_review' and second['cached']
    assert fetch.call_count==1
    assert disabled['evidence']==[] and disabled['provider_status']=='no_sources'
