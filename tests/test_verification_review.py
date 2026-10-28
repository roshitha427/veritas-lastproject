from copy import deepcopy

from backend.verification_review import annotate_verification_review
from tests.conftest import register


CLAIM = 'Elon Musk announced free Tesla cars for everyone in 2025.'


def result():
    return {'confidence': 51.9, 'ai_percentage': 51.9,
            'media_text_analysis': {'text_model_assessment': {'fake_percentage': 98.7}},
            'media_inspection': {'ocr_performed': True, 'visible_text': CLAIM, 'ocr_uncertain': True},
            'report_summary': {'schema_version': 5, 'verdict': 'unverified', 'sources': [], 'claims': []},
            'verification': {'provider': 'Tavily web search', 'provider_status': 'searched',
                'claim_searches': [{'claim_index': 0, 'claim': CLAIM, 'provider_status': 'searched',
                                   'query_statuses': [{'query': CLAIM, 'status': 'searched', 'requested_domains': ['reuters.com']}]}],
                'evidence': [
                    {'url': 'https://www.thestatesman.com/greetings', 'title': 'Happy 2025: New Year greetings',
                     'snippet': 'The new year brings joy and new opportunities for everyone in 2025.'},
                    {'url': 'https://factly.in/robot-video', 'title': 'Elon Musk robot haircut video',
                     'snippet': 'Elon Musk appears in a video getting a haircut from a robot. This is unrelated to a vehicle giveaway.'},
                    {'url': 'https://www.bbc.com/programmes/podcast', 'title': 'Global News Podcast',
                     'snippet': 'BBC brings you global news and headlines every day.'}]}}


def test_unverified_explanation_names_rejected_pages_and_required_offer_evidence():
    data = result()
    annotate_verification_review(data, CLAIM)
    review = data['report_summary']['verification_review']
    assert len(review['checked_sources']) == 3
    assert not any(row['accepted'] for row in review['checked_sources'])
    assert not data['verification']['evidence']
    assert all(row['reason'] for row in review['checked_sources'])
    assert any('eligibility' in needed and 'terms' in needed for needed in review['additional_evidence_needed'])
    assert any('original post' in needed for needed in review['additional_evidence_needed'])
    assert any('uncertain' in reason for reason in review['insufficiency_reasons'])
    assert review['searches'][0]['query_statuses'][0]['query'] == CLAIM
    assert review['relevant_source_urls'] == []
    assert data['report_summary']['verdict'] == 'unverified'
    assert data['ai_percentage'] == 51.9
    assert data['media_text_analysis']['text_model_assessment']['fake_percentage'] == 98.7
    again = deepcopy(data)
    annotate_verification_review(data, CLAIM)
    assert data == again


def test_outage_does_not_claim_configured_sources_were_checked():
    data = result()
    data['verification'] = {'provider': 'Tavily web search', 'provider_status': 'unavailable', 'evidence': []}
    annotate_verification_review(data, CLAIM)
    review = data['report_summary']['verification_review']
    assert not review['checked_sources']
    assert not review['searches']
    assert 'No completed source check' in review['summary']
    assert any('could not complete' in reason for reason in review['insufficiency_reasons'])


def test_timeless_health_claim_needs_clinical_evidence_without_an_announcement():
    claim = 'A dietary supplement makes people live forever.'
    data = {'report_summary': {'schema_version': 5, 'verdict': 'unverified'},
            'verification': {'provider': 'Source search', 'provider_status': 'searched', 'evidence': []}}
    annotate_verification_review(data, claim)
    needed = data['report_summary']['verification_review']['additional_evidence_needed']
    assert len(needed) == 1
    assert 'Clinical research' in needed[0]
    assert 'lifespan' in needed[0]
    assert 'announcement' not in needed[0]
    assert data['report_summary']['verdict'] == 'unverified'


def test_timeless_scientific_claim_needs_research_or_measurements():
    claim = 'Gravity causes objects to accelerate towards the Earth.'
    data = {'report_summary': {'schema_version': 5, 'verdict': 'unverified'},
            'verification': {'provider': 'Source search', 'provider_status': 'searched', 'evidence': []}}
    annotate_verification_review(data, claim)
    needed = data['report_summary']['verification_review']['additional_evidence_needed']
    assert 'scientific measurements' in needed[0]
    assert 'announcement' not in needed[0]


def test_legacy_ocr_fragment_indices_are_rechecked_against_complete_claim():
    data = result()
    raw = 'Global News\nBREAKING NEWS\nElon Musk Announces\nFree Tesla Cars for\nEveryone in 2025'
    data['report_summary']['unassessed_claim_count'] = 14
    for source in data['verification']['evidence']:
        source['candidate_claim_indices'] = [4]
    annotate_verification_review(data, raw)
    review = data['report_summary']['verification_review']
    assert 'BREAKING NEWS' not in review['claim']
    assert 'Elon Musk Announces Free Tesla Cars for Everyone in 2025' == review['claim']
    assert not any(row['accepted'] for row in review['checked_sources'])
    assert not any('No sufficiently specific claim wording' in row['reason'] for row in review['checked_sources'])
    assert review['unassessed_claim_count'] == 0
    assert not any('14 additional' in reason for reason in review['insufficiency_reasons'])


def test_unverified_saved_report_upgrade_does_not_search_again(client, app, monkeypatch):
    auth = register(client)
    store = app.extensions['store']
    saved = store.save('reports', {'owner_id': auth['user']['id'], 'type': 'image', 'status': 'complete',
                                  'content': CLAIM, 'result': result()})
    def unexpected(*args, **kwargs):
        raise AssertionError('Explaining saved source checks must not make another external request')
    monkeypatch.setattr('requests.get', unexpected)
    monkeypatch.setattr('requests.post', unexpected)
    first = client.get('/api/reports/' + saved['id']).json
    review = first['result']['report_summary']['verification_review']
    assert review['policy_version'] == 2
    assert len(review['checked_sources']) == 3
    assert not first['result']['verification']['evidence']
    second = client.get('/api/reports/' + saved['id']).json
    assert second['result']['report_summary']['verification_review'] == review
    assert second['metrics_updated_at'] == first['metrics_updated_at']


def test_pdf_explains_insufficiency_and_keeps_rejected_links_out_of_evidence(monkeypatch):
    import backend.pdf_report as module
    from backend.pdf_report import report_pdf
    data = result()
    annotate_verification_review(data, CLAIM)
    paragraphs = []
    original = module.Paragraph
    def capture(text, *args, **kwargs):
        paragraphs.append(text)
        return original(text, *args, **kwargs)
    monkeypatch.setattr(module, 'Paragraph', capture)
    pdf = report_pdf({'id': 'unverified-audit', 'type': 'image', 'created_at': '2026-10-24', 'content': CLAIM, 'result': data})
    assert pdf.startswith(b'%PDF')
    text = '\n'.join(paragraphs)
    assert 'What was checked' in text
    assert 'Evidence needed to settle the claim' in text
    assert 'Excluded from evidence: Global News Podcast' in text
    assert 'Sources to review' not in text
    assert 'eligibility' in text
