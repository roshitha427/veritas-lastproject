from copy import deepcopy
from urllib.parse import urlparse

import pytest

from backend.source_recommendations import source_recommendations, with_source_recommendations
from backend.verification import DEFAULT_SOURCES
from tests.conftest import register


def sources():
    return [{'domain': urlparse(url).hostname.removeprefix('www.'), 'enabled': True}
            for _, url, _ in DEFAULT_SOURCES]


def recommendations(claim, **summary):
    return source_recommendations(claim, dict(verdict='unverified', **summary), sources())


@pytest.mark.parametrize('claim', [
    'Eating an apple every day will make you immortal.',
    'Dietary supplements prevent humans from aging.',
    'Eating vegetables makes people live forever.',
    'Human lifespan can be increased by nutrition.',
])
def test_nutrition_and_lifespan_claims_route_to_health_sources_without_assigning_a_verdict(claim):
    from backend.source_routing import route_sources
    plan = route_sources(claim, sources())
    assert plan['topic'] == 'health'
    assert 'nih.gov' in plan['domains']
    assert plan['queries'][0] == claim
    assert len(plan['queries']) == 2
    assert all('false' not in query.lower() for query in plan['queries'])
    summary = {'verdict': 'unverified', 'sources': []}
    before = deepcopy(summary)
    found = source_recommendations(claim, summary, sources())
    assert [row['url'] for row in found] == ['https://www.who.int/']
    assert 'nutrition' in found[0]['description']
    assert summary == before


@pytest.mark.parametrize('claim', [
    'Apple iPhone battery lifespan doubles after an update.',
    'MacBook batteries have a longer lifespan.',
    'The lifespan of the Sun is measured in billions of years.',
])
def test_device_and_astronomical_lifespans_do_not_route_to_health(claim):
    from backend.source_routing import is_health_claim, route_sources
    assert not is_health_claim(claim)
    assert route_sources(claim, sources())['topic'] != 'health'
    assert not any(row['url'] == 'https://www.who.int/' for row in recommendations(claim))


def test_unverified_government_claim_gets_direct_portals_without_retrieved_evidence():
    found = recommendations('The Government of India announced a new national scheme today.')
    urls = [item['url'] for item in found]
    assert 'https://www.india.gov.in/' in urls
    assert 'https://www.pib.gov.in/factcheck.aspx' in urls
    assert all(url.startswith('https://') and '?' not in url for url in urls)
    assert all(item.get('description') and item.get('name') for item in found)
    assert all('snippet' not in item and 'id' not in item for item in found)
    pib = next(item for item in found if item['name'] == 'PIB Fact Check')
    assert 'central government' in pib['description']
    assert 'Government of India' in pib['description']


@pytest.mark.parametrize('claim,url', [
    ('Current AP chief minister announced a scheme.', 'https://www.ap.gov.in/'),
    ('Telangana government announced a holiday.', 'https://www.telangana.gov.in/'),
    ('Current tamilnadu chief minister', 'https://www.tn.gov.in/'),
    ('Does Example College in India have autonomous status?', 'https://www.ugc.gov.in/'),
    ('GIST is in Gangavaram, Nellore.', 'https://gist.edu.in/gist/gist-home/'),
    ('geethanjali institute of science and technology has announced a holiday.', 'https://gist.edu.in/gist/gist-home/'),
    ('This hospital offers a cure for dengue.', 'https://www.who.int/'),
    ('Earth is flat.', 'https://www.nasa.gov/'),
    ('A chemical freezes at this temperature.', 'https://www.nist.gov/'),
    ('Climate change is a hoax.', 'https://www.noaa.gov/'),
])
def test_topic_and_jurisdiction_choose_relevant_primary_portals(claim, url):
    assert url in [item['url'] for item in recommendations(claim)]


def test_generic_gist_word_and_unrelated_college_do_not_recommend_gist_institution():
    for claim in ['What is the gist of this government policy?', 'Example College announced a holiday.']:
        assert not any('gist.edu.in' in item['url'] for item in recommendations(claim))


def test_indian_weather_claim_prioritizes_official_forecasts_and_honors_disables():
    claim = 'A cyclone will hit Nellore tomorrow.'
    assert recommendations(claim)[0]['url'] == 'https://mausam.imd.gov.in/'
    found = source_recommendations(claim, {'verdict': 'unverified'},
                                  sources() + [{'domain': 'imd.gov.in', 'enabled': False}])
    assert not any('imd.gov.in' in item['url'] for item in found)


def test_disabled_sources_are_excluded_even_with_enabled_parent_domain():
    settings = sources() + [{'domain': 'pib.gov.in', 'enabled': False},
                            {'domain': 'telangana.gov.in', 'enabled': False}]
    found = source_recommendations('Telangana government announced a scheme.', {'verdict': 'unverified'}, settings)
    assert found == []
    central = source_recommendations('The Government of India announced a scheme.', {'verdict': 'unverified'}, settings)
    assert [item['url'] for item in central] == ['https://www.india.gov.in/']
    assert source_recommendations('Government announcement', {'verdict': 'unverified'}, []) == []


@pytest.mark.parametrize('verdict,mode,claim', [('true', None, 'Government announcement'),
    ('false', None, 'Government announcement'), ('unverified', 'media_only', 'Government announcement'),
    ('unverified', None, '')])
def test_no_suggestions_for_settled_reports_or_missing_claims(verdict, mode, claim):
    assert source_recommendations(claim, {'verdict': verdict, 'mode': mode}, sources()) == []


def test_recommendations_do_not_change_verdict_sources_or_evidence_metrics():
    original = {'verdict': 'unverified', 'sources': [], 'claims': [], 'summary_source_ids': []}
    before = deepcopy(original)
    updated = with_source_recommendations(original, 'Government of India announcement', sources())
    assert original == before
    assert updated['recommended_sources']
    assert {key: updated[key] for key in before} == before
    assert updated['recommendation_policy_version'] == 2
    assert with_source_recommendations(updated, 'Government of India announcement', sources()) == updated


@pytest.mark.parametrize('claim,expected', [
    ('Telangana government announced a holiday.', ['https://www.telangana.gov.in/']),
    ('GIST is in Gangavaram, Nellore.', ['https://gist.edu.in/gist/gist-home/']),
    ('Does Example College in India have autonomous status?', ['https://www.ugc.gov.in/']),
    ('GIST college has autonomous status.', ['https://gist.edu.in/gist/gist-home/', 'https://www.ugc.gov.in/']),
    ('This hospital offers a cure for dengue.', ['https://www.who.int/']),
    ('Earth is flat.', ['https://www.nasa.gov/']),
    ('ISRO launched Chandrayaan yesterday.', ['https://www.isro.gov.in/']),
    ('A chemical freezes at this temperature.', ['https://www.nist.gov/']),
    ('A cyclone will hit Nellore tomorrow.', ['https://mausam.imd.gov.in/']),
    ('A hurricane will hit Florida tomorrow.', ['https://www.noaa.gov/']),
    ('The weather forecast predicts a storm in Florida.', ['https://www.noaa.gov/']),
    ('Climate change is a hoax.', ['https://www.ipcc.ch/', 'https://www.noaa.gov/']),
    ('Earth climate change has stopped.', ['https://www.ipcc.ch/', 'https://www.noaa.gov/']),
    ('Tesla is giving away free cars this week.', ['https://www.tesla.com/blog']),
    ('Government of India announcement', ['https://www.india.gov.in/', 'https://www.pib.gov.in/factcheck.aspx']),
    ('Schools in Andhra Pradesh closed because of a cyclone.', ['https://www.ap.gov.in/', 'https://mausam.imd.gov.in/']),
    ('Telangana government announced a holiday because of rain.', ['https://www.telangana.gov.in/', 'https://mausam.imd.gov.in/']),
])
def test_recommendations_stay_within_the_relevant_primary_source_remit(claim, expected):
    found = recommendations(claim)
    assert [item['url'] for item in found] == expected
    for item in found:
        assert claim[:30] in item['description']
        assert 'check ' in item['description']
        assert len(item['description']) <= 220


@pytest.mark.parametrize('claim', [
    'Example College announced a holiday.',
    'Does Example College have autonomous status?',
    'Example College in Spain has autonomous status.',
    'Example College in Kenya has autonomous status.',
    'A school is now autonomous.',
    'Example College received engineering accreditation.',
    'This hospital opened yesterday.',
    'A cyclone will hit Example City tomorrow.',
    'The Canada government announced a new scheme.',
    'The US government announced a new scheme.',
    'The Spain prime minister announced a new scheme.',
    'The UK parliament announced a new scheme.',
    'Government announcement',
    'The central government announced a scheme.',
    'An unnamed company is giving away free money.',
    'A famous actor died yesterday.',
    'The home team won the football match.',
    'This biology experiment was completed yesterday.',
    'There is no matching official portal for this statement.',
])
def test_no_generic_or_unrelated_links_when_a_direct_remit_is_unknown(claim):
    assert recommendations(claim) == []


def test_updated_policy_replaces_legacy_generic_recommendations_without_changing_evidence():
    original = {'verdict': 'unverified', 'sources': [], 'summary_source_ids': [7],
                'recommended_sources': [{'name': 'AFP Fact Check', 'url': 'https://factcheck.afp.com/'}],
                'recommendation_policy_version': 1}
    before = deepcopy(original)
    updated = with_source_recommendations(original, 'A famous actor died yesterday.', sources())
    assert original == before
    assert updated['recommended_sources'] == []
    assert updated['recommendation_policy_version'] == 2
    assert updated['verdict'] == before['verdict']
    assert updated['sources'] == before['sources']
    assert updated['summary_source_ids'] == before['summary_source_ids']


def test_saved_unverified_report_gets_portals_without_external_requests(client, app, monkeypatch):
    auth = register(client)
    store = app.extensions['store']
    report = store.save('reports', {'owner_id': auth['user']['id'], 'status': 'complete', 'type': 'text',
        'content': 'Telangana government announced a new scheme.',
        'result': {'confidence': 62., 'fake_percentage': 38.,
                   'report_summary': {'schema_version': 5, 'verdict': 'unverified', 'sources': [], 'claims': []}}})
    def unexpected(*args, **kwargs):
        pytest.fail('Adding manual-check links must not make an external request')
    monkeypatch.setattr('requests.get', unexpected)
    monkeypatch.setattr('requests.post', unexpected)
    first = client.get('/api/reports/' + report['id']).json
    result = first['result']
    summary = result['report_summary']
    assert any(item['url'] == 'https://www.telangana.gov.in/' for item in summary['recommended_sources'])
    assert summary['verdict'] == 'unverified'
    assert summary['sources'] == []
    assert result['confidence'] == 62
    assert result['fake_percentage'] == 38
    again = client.get('/api/reports/' + report['id']).json
    assert again['metrics_updated_at'] == first['metrics_updated_at']
    store.save('sources', {'name': 'Disabled Telangana', 'domain': 'telangana.gov.in', 'enabled': False})
    refreshed = client.get('/api/reports/' + report['id']).json['result']['report_summary']
    assert not any('telangana.gov.in' in item['url'] for item in refreshed['recommended_sources'])


def test_new_analysis_attaches_manual_check_links(app, monkeypatch):
    import backend.analysis as module
    analyzer = app.extensions['analyzer']
    monkeypatch.setattr(analyzer, 'text', lambda text: (.38, {'status': 'ready'}))
    monkeypatch.setattr(module, 'verify_submission', lambda *a, **k: {'evidence': []})
    monkeypatch.setattr(module, 'summarize_report', lambda *a, **k: {'verdict': 'unverified', 'sources': []})
    result = analyzer.analyze('text', 'The Government of India announced a scheme.', None, 'en', sources())
    assert result['report_summary']['recommended_sources']
    assert result['report_summary']['verdict'] == 'unverified'


@pytest.mark.parametrize('kind', ['text', 'image'])
def test_unverified_pdf_includes_direct_portals_separate_from_evidence(monkeypatch, kind):
    import backend.pdf_report as module
    from backend.pdf_report import report_pdf
    text = []
    original = module.Paragraph
    def capture(value, *args, **kwargs):
        text.append(value)
        return original(value, *args, **kwargs)
    monkeypatch.setattr(module, 'Paragraph', capture)
    summary = with_source_recommendations({'verdict': 'unverified', 'sources': []}, 'Government of India announcement', sources())
    pdf = report_pdf({'id': 'suggestion-test', 'type': kind, 'created_at': '2026-10-24',
                      'content': 'Government of India announcement', 'result': {'report_summary': summary}})
    assert pdf.startswith(b'%PDF')
    combined = '\n'.join(text)
    assert 'Where to check this claim' in combined
    assert 'https://www.india.gov.in/' in combined
    assert 'these links have not verified this claim' in combined
    assert 'Sources to review' not in combined
