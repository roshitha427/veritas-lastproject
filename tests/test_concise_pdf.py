import pytest

from backend.pdf_report import report_pdf


@pytest.mark.parametrize('detailed', [False, True])
def test_unverified_exports_include_general_explanation_as_plain_prose(monkeypatch, detailed):
    from backend.report_contract import attach_report_contract
    import backend.compact_pdf as compact
    import backend.concise_pdf as concise
    import backend.pdf_report as full
    messages = []
    original = full.Paragraph
    def capture(text, *args, **kwargs):
        messages.append(text)
        return original(text, *args, **kwargs)
    for module in (compact, concise, full):
        monkeypatch.setattr(module, 'Paragraph', capture)
    report = {'id': 'general-explanation', 'type': 'text', 'created_at': '2026-10-28',
        'content': 'A food is claimed to prevent human mortality.', 'result': {
            'verification': {'evidence': []},
            'report_summary': {'schema_version': 5, 'verdict': 'unverified',
                'summary': 'The retrieved sources do not settle this statement.',
                'gemini_model_assessment': {'status': 'completed', 'verdict': 'false',
                    'answer': 'A healthy diet does not establish human immortality. [99]'}}}}
    attach_report_contract(report)
    assert report_pdf(report, detailed=detailed).startswith(b'%PDF')
    text = '\n'.join(messages)
    assert 'A healthy diet does not establish human immortality.' in text
    assert 'This explanation uses general knowledge' in text
    assert '[99]' not in text
    assert 'Model-only verdict:' not in text
    assert 'Additional explanation (not source verified)' not in text


@pytest.mark.parametrize('status', ['needs_current_sources', 'network_error', 'not_configured'])
def test_general_explanation_does_not_display_unavailable_or_current_fact_answers(status):
    from backend.report_summary import general_knowledge_explanation
    summary = {'verdict': 'unverified', 'gemini_model_assessment': {
        'status': status, 'answer': 'An outdated answer that must not appear.'}}
    assert general_knowledge_explanation(summary) == ''


@pytest.mark.parametrize('kind,labels', [
    ('text', ['Real news pattern score', 'Fake news pattern score', 'Text model confidence']),
    ('image', ['Real image pattern score', 'AI image pattern score', 'Image model confidence']),
    ('video', ['Real video frame pattern score', 'AI video frame pattern score', 'Video frame model confidence']),
])
def test_unresolved_pdf_keeps_metrics_and_omits_model_faults(monkeypatch, kind, labels):
    import backend.concise_pdf as module
    import backend.pdf_report as pdf_module
    messages, tables = [], []
    original = module.Paragraph
    original_table = pdf_module.Table

    def capture(text, *args, **kwargs):
        messages.append(text)
        return original(text, *args, **kwargs)

    monkeypatch.setattr(module, 'Paragraph', capture)
    def table(rows, *args, **kwargs):
        tables.append(rows)
        return original_table(rows, *args, **kwargs)
    monkeypatch.setattr(pdf_module, 'Table', table)
    result = {'ai_percentage': 99.9, 'real_percentage': .1, 'confidence': 99.9,
              'media_inspection': {'description': 'A person beside a blue car.'},
              'media_origin': {'verdict': 'inconclusive', 'reason': 'Internal model failure.'},
              'report_summary': {'verdict': 'unverified', 'summary': 'Long technical failure.',
                                 'limitations': ['Local validation failed.']}}
    pdf = report_pdf({'id': 'concise-test', 'created_at': '2026-10-22', 'type': kind,
                      'title': 'Example', 'content': '', 'result': result}, detailed=False)
    assert pdf.startswith(b'%PDF')
    text = '\n'.join(messages)
    assert 'Internal model failure' not in text
    assert 'Long technical failure' not in text
    assert 'Local validation failed' not in text
    assert len(tables) == 1 and tables[0][0] == ['Measure', 'Result']
    assert [row[0] for row in tables[0][1:]] == labels
    assert [row[1] for row in tables[0][1:]] == [
        '0.1%', 'Not assessed' if kind == 'text' else '99.9%', '99.9%']
    if kind == 'text':
        assert 'UNVERIFIED' in text
    else:
        assert 'A person beside a blue car.' in text
        assert 'Origin inconclusive' in text


@pytest.mark.parametrize('kind,verdict', [('image', 'ai_generated'), ('image', 'likely_real'), ('image', 'ai_content_declared'), ('video', 'ai_generated'), ('text', 'true'), ('text', 'false')])
def test_resolved_reports_use_the_full_pdf(monkeypatch, kind, verdict):
    import backend.concise_pdf as module
    def unexpected(*args, **kwargs):
        pytest.fail('Resolved report was sent to the concise PDF renderer')
    monkeypatch.setattr(module, 'concise_report_pdf', unexpected)
    summary = {'mode': 'media_only', 'verdict': 'unverified'} if kind != 'text' else {'verdict': verdict}
    pdf = report_pdf({'id': 'resolved-test', 'created_at': '2026-10-22', 'type': kind,
                      'result': {'media_origin': {'verdict': verdict}, 'report_summary': summary}})
    assert pdf.startswith(b'%PDF')
