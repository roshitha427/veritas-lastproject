import hashlib
import json

import pytest

from backend.evidence_nli import EvidenceVerifier
from backend.report_summary import summarize_report


def source(text, **extra):
    return {'url': 'https://example.gov.in/bridge', 'kind': 'official_source_snapshot',
            'title': 'Riverford bridge opening', 'snippet': text, **extra}


def verifier(monkeypatch, tmp_path, probabilities, *, qualified=True):
    directory = tmp_path / 'fixture-evidence-model'
    directory.mkdir(exist_ok=True)
    files = {'config.json': b'{}', 'model.safetensors': b'unit-fixture-weights',
             'tokenizer_config.json': b'{}', 'tokenizer.json': b'{}', 'validation-report.json': b'{"fixture":true}'}
    for name, blob in files.items():
        (directory / name).write_bytes(blob)
    fingerprint = lambda blob: hashlib.sha256(blob).hexdigest()
    card = {'version': 'unit-fixture-v1', 'status': 'validated' if qualified else 'experimental',
            'locally_validated': qualified, 'deployment_ready': qualified,
            'runtime_validation': {'policy_version': 1, 'review_status': 'approved',
                'scope': 'english_claim_passage_comparison', 'threshold': .92,
                'model_version': 'unit-fixture-v1', 'threshold_selection_split': 'calibration',
                'evaluation_split': 'held_out_test', 'reviewer': 'Unit test fixture',
                'reviewed_at': '2020-01-01T00:00:00+00:00', 'evaluation_manifest_sha256': 'a' * 64,
                'model_files_sha256': {name: fingerprint(blob) for name, blob in files.items() if name != 'validation-report.json'},
                'evaluation_report': {'path': 'validation-report.json', 'sha256': fingerprint(files['validation-report.json'])}}}
    (directory / 'model-card.json').write_text(json.dumps(card), encoding='utf-8')
    model = EvidenceVerifier(directory)
    monkeypatch.setattr(model, 'predict', lambda pairs: probabilities[:len(pairs)])
    return model


def test_reviewed_qualified_passage_support_does_not_need_publisher_rating(monkeypatch, tmp_path):
    model = verifier(monkeypatch, tmp_path, [{'entailment': .99, 'contradiction': .005, 'neutral': .005}])
    evidence = {'evidence': [source('The Riverford bridge opened to the public in 1985.')], 'provider_status': 'searched'}
    claim = 'The Riverford bridge opened in 1985.'
    assessment = model.assess(claim, evidence)
    assert assessment['verdict'] == 'true'
    assert assessment['supporting_evidence'] == [1]
    summary = summarize_report(claim, 'text', {}, evidence, local_assessment=assessment)
    assert summary['verdict'] == 'true'
    assert summary['provider'] == 'Local evidence comparison'
    assert 'clearly rated fact-check' not in summary['summary']


def test_neutral_or_weak_evidence_stays_unverified(monkeypatch, tmp_path):
    for scores in [{'entailment': .5, 'contradiction': .2, 'neutral': .3},
                   {'entailment': .001, 'contradiction': .001, 'neutral': .998}]:
        model = verifier(monkeypatch, tmp_path, [scores])
        result = model.assess('The Riverford bridge opened in 1985.',
            {'evidence': [source('The Riverford bridge opened to the public in 1985.')]})
        assert result['verdict'] == 'unverified'


def test_current_office_claim_needs_fresh_evidence_not_import_date(monkeypatch, tmp_path):
    model = verifier(monkeypatch, tmp_path, [{'entailment': .99, 'contradiction': .005, 'neutral': .005}])
    result = model.assess('Current Example Pradesh chief minister is Aruna Example', {'evidence': [{
        'url': 'https://example.gov.in/leadership', 'kind': 'official_source_snapshot',
        'title': 'Example Pradesh leadership', 'snippet': 'Aruna Example is the Chief Minister of Example Pradesh.',
        'fetched_at': '2099-01-01T00:00:00Z'}]})
    assert result['verdict'] == 'unverified'


def test_conflicting_source_comparisons_do_not_choose_a_verdict(monkeypatch, tmp_path):
    model = verifier(monkeypatch, tmp_path, [
        {'entailment': .99, 'contradiction': .005, 'neutral': .005},
        {'entailment': .005, 'contradiction': .99, 'neutral': .005}])
    result = model.assess('The Riverford bridge opened in 1985.', {'evidence': [
        source('The Riverford bridge opened to the public in 1985.'),
        source('The Riverford bridge did not open to the public in 1985.')]})
    assert result['verdict'] == 'unverified'
    assert result['claims'][0]['conflict'] is True


def test_metadata_and_untrusted_sources_cannot_settle_a_claim(monkeypatch, tmp_path):
    model = verifier(monkeypatch, tmp_path, [{'entailment': .999, 'contradiction': 0., 'neutral': .001}])
    result = model.assess('The Riverford bridge opened in 1985.', {'evidence': [
        source('The Riverford bridge opened to the public in 1985.', retrieval_status='metadata_only'),
        {'url': 'https://random.example/story', 'snippet': 'The Riverford bridge opened to the public in 1985.'}]})
    assert result['verdict'] == 'unverified'


def test_quoted_viral_claim_is_not_treated_as_source_support(monkeypatch, tmp_path):
    model = verifier(monkeypatch, tmp_path, [{'entailment': .999, 'contradiction': 0., 'neutral': .001}])
    result = model.assess('The Riverford bridge opened in 1985.', {'evidence': [source(
        'An image purporting to show the Riverford bridge opened in 1985 is being shared online with users believing it.') ]})
    assert result['verdict'] == 'unverified'


def test_unrelated_passage_cannot_contradict_a_claim_based_on_title_alone(monkeypatch, tmp_path):
    model = verifier(monkeypatch, tmp_path, [{'entailment': 0., 'contradiction': .999, 'neutral': .001}])
    result = model.assess('The Riverford bridge opened in 1985.', {'evidence': [source(
        'This medicine does not cure dengue. The result of the medical trial is inconclusive.') ]})
    assert result['verdict'] == 'unverified'


def test_unvalidated_indic_scripts_are_not_mistaken_for_english():
    from ml.text_cleaning import content_language
    assert content_language('தமிழ்நாடு அரசு') == 'unsupported'
    assert content_language('বাংলা সংবাদ') == 'unsupported'
    assert content_language('తెలుగు వార్తలు') == 'te'


def test_ordinal_day_does_not_prevent_supported_passage_verdict(monkeypatch, tmp_path):
    model = verifier(monkeypatch, tmp_path, [{'entailment': .999, 'contradiction': 0., 'neutral': .001}])
    result = model.assess('The Riverford bridge opened on 26 January 1985.', {'evidence': [source(
        'The Riverford bridge opened on 26th January 1985.')]})
    assert result['verdict'] == 'true'


@pytest.mark.parametrize('relationship,candidate', [('entailment', 'true'), ('contradiction', 'false')])
def test_unreviewed_strong_scores_remain_experimental_candidates(monkeypatch, tmp_path, relationship, candidate):
    scores = {'entailment': .001, 'contradiction': .001, 'neutral': .001}
    scores[relationship] = .998
    model = verifier(monkeypatch, tmp_path, [scores], qualified=False)
    claim = 'The Riverford bridge opened in 1985.'
    evidence = {'evidence': [source('The Riverford bridge opened to the public in 1985.')]}
    result = model.assess(claim, evidence)
    assert model.ready() is True
    assert result['verdict'] == 'unverified'
    assert result['supporting_evidence'] == [] and result['contradicting_evidence'] == []
    diagnostic = result['evidence_model']
    assert diagnostic['status'] == 'experimental'
    assert diagnostic['settlement_enabled'] is False
    assert diagnostic['candidate_verdict'] == candidate
    assert diagnostic['comparisons'][0]['candidate_verdict'] == candidate
    assert diagnostic['comparisons'][0]['accepted_verdict'] == 'unverified'
    assert summarize_report(claim, 'text', {}, evidence, local_assessment=result)['verdict'] == 'unverified'


@pytest.mark.parametrize('rating,verdict', [('True', 'true'), ('False', 'false')])
def test_existing_published_factcheck_does_not_require_nli_release_qualification(monkeypatch, tmp_path, rating, verdict):
    model = verifier(monkeypatch, tmp_path, [], qualified=False)
    claim = 'The Riverford bridge opened in 1985.'
    evidence = {'evidence': [{'kind': 'fact_check_review', 'source_type': 'published_fact_check',
        'url': 'https://www.reuters.com/fact-check/bridge', 'publisher': 'Reuters',
        'claim': claim, 'rating': rating, 'reviewed_at': '2020-01-01T00:00:00Z',
        'title': 'Riverford bridge fact check', 'snippet': 'A published review of the Riverford bridge opening.'}]}
    result = model.assess(claim, evidence)
    assert result['verdict'] == verdict
    assert result['evidence_model']['comparisons'] == []
    assert result['evidence_model']['settlement_enabled'] is False
    assert 'published reviews' in result['reason']


def test_qualified_counterevidence_can_contradict_a_changed_material_value(monkeypatch, tmp_path):
    model = verifier(monkeypatch, tmp_path, [{'entailment': .001, 'contradiction': .998, 'neutral': .001}])
    result = model.assess('The Riverford bridge opened in 1986.',
                         {'evidence': [source('The Riverford bridge opened to the public in 1985.')]})
    assert result['verdict'] == 'false'
    assert result['contradicting_evidence'] == [1]
    assert result['evidence_model']['qualification']['qualified'] is True


@pytest.mark.parametrize('field,value', [
    ('scope', 'all_languages_and_all_claims'),
    ('threshold', .5),
    ('model_version', 'different-model'),
    ('threshold_selection_split', 'test'),
    ('evaluation_split', 'training'),
    ('review_status', 'draft'),
    ('reviewer', ''),
    ('reviewed_at', '2999-01-01T00:00:00Z'),
    ('evaluation_manifest_sha256', 'not-a-fingerprint'),
])
def test_review_contract_cannot_approve_a_different_or_unreviewed_release(monkeypatch, tmp_path, field, value):
    model = verifier(monkeypatch, tmp_path, [])
    card_path = model.directory / 'model-card.json'
    card = json.loads(card_path.read_text(encoding='utf-8'))
    card['runtime_validation'][field] = value
    card_path.write_text(json.dumps(card), encoding='utf-8')
    assert model.settlement_qualification()['qualified'] is False


def test_reviewed_fingerprints_are_checked_and_cached_approval_is_invalidated(monkeypatch, tmp_path):
    model = verifier(monkeypatch, tmp_path, [])
    assert model.settlement_qualification()['qualified'] is True
    (model.directory / 'model.safetensors').write_bytes(b'changed-unreviewed-weights')
    qualification = model.settlement_qualification()
    assert qualification['qualified'] is False
    assert 'fingerprint' in qualification['reason']


def test_model_flags_alone_do_not_qualify_missing_review_report(monkeypatch, tmp_path):
    model = verifier(monkeypatch, tmp_path, [])
    (model.directory / 'validation-report.json').unlink()
    assert model.ready() is True
    assert model.settlement_qualification()['qualified'] is False
