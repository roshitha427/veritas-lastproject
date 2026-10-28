from copy import deepcopy

from PIL import Image
import pytest

from backend.pdf_report import report_pdf
from backend.report_contract import attach_report_contract
from backend.report_media import make_image_preview
from tests.conftest import register


MODEL_LABELS = {
    'text': ['Real news pattern score', 'Fake news pattern score', 'Text model confidence'],
    'image': ['Real image pattern score', 'AI image pattern score', 'Image model confidence'],
    'video': ['Real video frame pattern score', 'AI video frame pattern score', 'Video frame model confidence'],
}


def report_for(kind='image'):
    source = {'id': 11, 'url': 'https://example.gov.in/bridge', 'title': 'Bridge record',
              'publisher': 'Official records', 'snippet': 'The Riverford bridge opened in 1985.'}
    report = {'id': 'compact-report', 'created_at': '2026-10-25', 'status': 'complete',
        'type': kind, 'content': 'The Riverford bridge opened in 1985.',
        'result': {'ai_percentage': 89.2, 'fake_percentage': 51.9, 'real_percentage': 48.1, 'confidence': 51.9,
            'media_origin': {'verdict': 'inconclusive'},
            'media_inspection': {'description': 'INTERNAL_SCENE_DETAILS', 'ocr_performed': True,
                                 'visible_text': 'ORIGINAL_OCR_DETAILS'},
            'verification': {'evidence': [source], 'checked_sources': [dict(source, accepted=True)],
                             'provider': 'INTERNAL_PROVIDER_DETAILS'},
            'report_summary': {'schema_version': 5, 'verdict': 'true',
                'summary': 'The dated source supports the claim [11].', 'reason': 'Official records give the date [11].',
                'sources': [source], 'summary_source_ids': [11],
                'claims': [{'text': 'The Riverford bridge opened in 1985.', 'verdict': 'true',
                            'reason': 'The record states the opening date [11].', 'supporting_evidence': [11]}]}}}
    attach_report_contract(report)
    return report


def test_default_canonical_pdf_has_one_verification_and_one_metric_table(monkeypatch):
    import backend.compact_pdf as module
    paragraphs, tables = [], []
    original_paragraph, original_table = module.Paragraph, module.Table
    def paragraph(text, *args, **kwargs):
        paragraphs.append(text)
        return original_paragraph(text, *args, **kwargs)
    def table(rows, *args, **kwargs):
        tables.append(rows)
        return original_table(rows, *args, **kwargs)
    monkeypatch.setattr(module, 'Paragraph', paragraph)
    monkeypatch.setattr(module, 'Table', table)
    report = report_for()
    before = deepcopy(report)
    pdf = report_pdf(report, detailed=False)
    text = '\n'.join(paragraphs)
    assert pdf.startswith(b'%PDF')
    assert text.count('Verification: TRUE') == 1
    assert len(tables) == 1
    assert tables[0] == [['Measure', 'Result'], ['Real image pattern score', '48.1%'],
                         ['AI image pattern score', '51.9%'], ['Image model confidence', '51.9%']]
    assert '[11] Official records' in text
    assert 'https://example.gov.in/bridge' in text
    assert not any(value in text for value in ('INTERNAL_SCENE_DETAILS', 'ORIGINAL_OCR_DETAILS', 'INTERNAL_PROVIDER_DETAILS'))
    assert report == before


@pytest.mark.parametrize('kind', ['text', 'image', 'video'])
def test_unresolved_compact_metrics_keep_the_selected_model_scores(kind):
    from backend.compact_pdf import related_metrics
    report = report_for(kind)
    summary = report['result']['report_summary']
    summary['verdict'] = summary['claims'][0]['verdict'] = 'unverified'
    summary['claims'][0]['supporting_evidence'] = []
    report['result']['media_text_analysis'] = {
        'text_model_assessment': {'real_percentage': 99., 'fake_percentage': 1., 'confidence': 99.}}
    attach_report_contract(report)
    heading, rows, _ = related_metrics(report, report['result']['report_contract'])
    assert heading == 'Model metrics'
    assert [label for label, _ in rows] == MODEL_LABELS[kind]
    assert [value for _, value in rows] == [48.1, 51.9, 51.9]


def test_unverified_default_export_explains_checked_sources_and_missing_evidence(monkeypatch):
    import backend.compact_pdf as module
    paragraphs = []
    original = module.Paragraph
    def paragraph(text, *args, **kwargs):
        paragraphs.append(text)
        return original(text, *args, **kwargs)
    monkeypatch.setattr(module, 'Paragraph', paragraph)
    report = report_for()
    summary = report['result']['report_summary']
    summary['verdict'] = summary['claims'][0]['verdict'] = 'unverified'
    summary['verification_review'] = {
        'summary': 'Two pages were checked; neither records the bridge opening date.',
        'checked_sources': [{'publisher': 'Town archive', 'accepted': False,
                             'url': 'https://example.gov.in/unrelated'}],
        'additional_evidence_needed': ['An original dated bridge opening notice.']}
    attach_report_contract(report)
    assert report_pdf(report, detailed=False).startswith(b'%PDF')
    text = '\n'.join(paragraphs)
    assert 'Two pages were checked; neither records the bridge opening date.' in text
    assert 'Sources checked: Town archive.' in text
    assert 'Evidence needed: An original dated bridge opening notice.' in text
    assert '<link href="https://example.gov.in/unrelated"' not in text
    assert 'Snapshot:' not in text


def test_detailed_export_keeps_the_recorded_ocr_and_module_audit(monkeypatch):
    import backend.pdf_report as module
    paragraphs = []
    original = module.Paragraph
    def paragraph(text, *args, **kwargs):
        paragraphs.append(text)
        return original(text, *args, **kwargs)
    monkeypatch.setattr(module, 'Paragraph', paragraph)
    pdf = report_pdf(report_for(), detailed=True)
    assert pdf.startswith(b'%PDF')
    assert 'ORIGINAL_OCR_DETAILS' in '\n'.join(paragraphs)
    assert 'INTERNAL_SCENE_DETAILS' in '\n'.join(paragraphs)
    assert 'The claim could not be verified.' not in '\n'.join(paragraphs)


@pytest.mark.parametrize('kind', ['text', 'image', 'video'])
def test_default_pdf_is_detailed_even_for_unverified_or_inconclusive_reports(monkeypatch, kind):
    import backend.pdf_report as module
    import backend.compact_pdf as compact
    import backend.concise_pdf as concise
    paragraphs, tables = [], []
    original_paragraph, original_table = module.Paragraph, module.Table
    def paragraph(text, *args, **kwargs):
        paragraphs.append(text)
        return original_paragraph(text, *args, **kwargs)
    def table(rows, *args, **kwargs):
        tables.append(rows)
        return original_table(rows, *args, **kwargs)
    def unexpected(*args, **kwargs):
        pytest.fail('The normal PDF download must retain detailed claim and evidence sections.')
    monkeypatch.setattr(module, 'Paragraph', paragraph)
    monkeypatch.setattr(module, 'Table', table)
    monkeypatch.setattr(compact, 'compact_report_pdf', unexpected)
    monkeypatch.setattr(concise, 'concise_report_pdf', unexpected)
    report = report_for(kind)
    summary = report['result']['report_summary']
    summary.update(verdict='unverified', reason='The available record does not settle the opening date.',
        verification_review={'insufficiency_reasons': ['No dated opening notice was retrieved.'],
            'additional_evidence_needed': ['The original dated bridge opening notice.']})
    summary['claims'][0].update(verdict='unverified', supporting_evidence=[])
    report['result']['media_text_analysis'] = {'content': 'Extracted transcript content.',
        'ai_percentage': 87., 'text_model_assessment': {
            'reason': 'Extracted-text classifier scope.', 'real_percentage': 99.,
            'fake_percentage': 1., 'confidence': 99.}}
    report['result']['temporal_assessment'] = {'status': 'ready', 'fake_percentage': 24.,
        'sampled_frames': 8, 'scope': 'Temporal review scope.'}
    report['result']['video_assessment'] = {'status': 'experimental',
        'synthetic_or_manipulated_percentage': 88., 'real_class_percentage': 12.}
    report['_image_preview'] = make_image_preview(Image.new('RGB', (100, 60), '#6454ab'))
    attach_report_contract(report)
    assert report_pdf(report).startswith(b'%PDF')
    text = '\n'.join(paragraphs)
    assert 'Claim checked: The Riverford bridge opened in 1985.' in text
    assert 'Why the evidence is insufficient' in text
    assert 'No dated opening notice was retrieved.' in text
    assert 'The original dated bridge opening notice.' in text
    assert 'Original source passage: The Riverford bridge opened in 1985.' in text
    assert 'https://example.gov.in/bridge' in text
    assert len(tables) == 1 and tables[0][0] == ['Measure', 'Result']
    assert [row[0] for row in tables[0][1:]] == MODEL_LABELS[kind]
    assert [row[1] for row in tables[0][1:]] == ['48.1%', '51.9%', '51.9%']
    if kind != 'text':
        assert 'ORIGINAL_OCR_DETAILS' in text
        assert 'INTERNAL_SCENE_DETAILS' in text
        assert 'Extracted transcript content.' in text
        assert 'Temporal review scope.' in text


@pytest.mark.parametrize('kind', ['text', 'image', 'video'])
def test_detailed_pdf_does_not_fill_missing_scores_from_other_models(monkeypatch, kind):
    import backend.pdf_report as module
    tables = []
    original = module.Table
    def table(rows, *args, **kwargs):
        tables.append(rows)
        return original(rows, *args, **kwargs)
    monkeypatch.setattr(module, 'Table', table)
    report = report_for(kind)
    result = report['result']
    result.update(real_percentage=None, fake_percentage=None, confidence=None,
                  ai_percentage=89.2, media_text_analysis={
                      'text_model_assessment': {'real_percentage': 99., 'fake_percentage': 1.,
                                                'confidence': 99.}})
    assert report_pdf(report).startswith(b'%PDF')
    assert len(tables) == 1 and tables[0][0] == ['Measure', 'Result']
    assert [row[0] for row in tables[0][1:]] == MODEL_LABELS[kind]
    assert [row[1] for row in tables[0][1:]] == [
        'Not assessed', 'Not assessed' if kind == 'text' else '89.2%', 'Not assessed']


def test_detailed_pdf_keeps_model_caveats_and_sourced_statistics_without_extra_accuracy_metrics(monkeypatch):
    import backend.pdf_report as module
    paragraphs = []
    original = module.Paragraph
    def paragraph(text, *args, **kwargs):
        paragraphs.append(text)
        return original(text, *args, **kwargs)
    monkeypatch.setattr(module, 'Paragraph', paragraph)
    report = report_for('text')
    report['result']['limitations'] = [
        'External claim benchmark: this article-trained model scored 48.4% on 500 unambiguous LIAR political claims. Its historical article accuracy does not establish claim verification accuracy.',
        'The visual classifier did not meet the requested 90% held-out accuracy target. Treat its scores as experimental.']
    source = report['result']['verification']['evidence'][0]
    source['snippet'] += ' The official record states that 90% of the work was complete before opening.'
    assert report_pdf(report).startswith(b'%PDF')
    text = '\n'.join(paragraphs)
    assert 'scored 48.4%' not in text
    assert 'requested 90%' not in text
    assert 'was evaluated on 500 unambiguous LIAR political claims' in text
    assert 'historical article accuracy does not establish claim verification accuracy' in text
    assert 'did not meet the requested held-out accuracy target' in text
    assert 'Treat its scores as experimental.' in text
    assert '90% of the work was complete' in text


@pytest.mark.parametrize('kind,embedded', [('image', True), ('video', False)])
def test_compact_export_embeds_only_private_image_preview(kind, embedded):
    report = report_for(kind)
    report['_image_preview'] = make_image_preview(Image.new('RGB', (200, 100), '#6454ab'))
    pdf = report_pdf(report, detailed=False)
    assert (b'/Subtype /Image' in pdf) is embedded


def test_compact_sources_never_promote_an_excluded_citation(monkeypatch):
    import backend.compact_pdf as module
    paragraphs = []
    original = module.Paragraph
    def paragraph(text, *args, **kwargs):
        paragraphs.append(text)
        return original(text, *args, **kwargs)
    monkeypatch.setattr(module, 'Paragraph', paragraph)
    report = report_for()
    report['result']['verification']['checked_sources'][0].update(accepted=False, reason='Unrelated page.')
    report_pdf(report, detailed=False)
    text = '\n'.join(paragraphs)
    assert '<link href="https://example.gov.in/bridge"' not in text
    assert 'Some saved citations are unavailable or excluded' in text
    assert report['result']['report_summary']['verdict'] == 'true'


def test_settled_default_pdf_links_verdict_evidence_instead_of_related_context(monkeypatch):
    import backend.compact_pdf as module
    paragraphs = []
    original = module.Paragraph
    def paragraph(text, *args, **kwargs):
        paragraphs.append(text)
        return original(text, *args, **kwargs)
    monkeypatch.setattr(module, 'Paragraph', paragraph)
    report = report_for()
    context = {'id': 1, 'url': 'https://example.org/context',
               'snippet': 'The Riverford bridge has a tourist information page.'}
    summary = report['result']['report_summary']
    summary['sources'].insert(0, context)
    summary['summary_source_ids'] = [1, 11]
    summary['claims'][0]['related_evidence'] = [1]
    report['result']['verification']['evidence'].append(context)
    attach_report_contract(report)
    assert 1 in report['result']['report_contract']['verification']['cited_source_ids']
    assert report_pdf(report, detailed=False).startswith(b'%PDF')
    text = '\n'.join(paragraphs)
    assert '<link href="https://example.gov.in/bridge"' in text
    assert '<link href="https://example.org/context"' not in text


def test_pdf_api_uses_detailed_default_and_explicit_compact_opt_in(client, app, monkeypatch):
    auth = register(client)
    report = report_for('text')
    report['owner_id'] = auth['user']['id']
    app.extensions['store'].save('reports', report)
    calls = []
    def render(current_report, detailed=False):
        calls.append((current_report['id'], detailed, current_report['result']['report_contract']['report_id']))
        return b'%PDF-test'
    monkeypatch.setattr('backend.app.report_pdf', render)
    assert client.get('/api/reports/compact-report/pdf').status_code == 200
    assert client.get('/api/reports/compact-report/pdf?details=1').status_code == 200
    assert client.get('/api/reports/compact-report/pdf?details=0').status_code == 200
    assert calls == [('compact-report', True, 'compact-report'), ('compact-report', True, 'compact-report'),
                     ('compact-report', False, 'compact-report')]
