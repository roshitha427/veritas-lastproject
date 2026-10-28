from copy import deepcopy
from html import escape
import importlib

import pytest
from reportlab.platypus import Table

from backend.report_links import manual_check_links


def suggestion(name='Town records', url='https://records.example.gov/bridge',
               description='Check the dated bridge opening notice and its location.'):
    return {'name': name, 'url': url, 'description': description}


def summary_with(rows):
    return {'schema_version': 5, 'verdict': 'unverified',
            'summary': 'The available records do not settle the opening date.',
            'recommended_sources': rows}


@pytest.mark.parametrize('row', [
    None, 'https://records.example.gov', [], 7,
    {}, {'name': 'Town records', 'url': 'https://records.example.gov'},
    suggestion(name=None), suggestion(name=True), suggestion(name='  '),
    suggestion(description=None), suggestion(description=[]), suggestion(description='  '),
    suggestion(url=None), suggestion(url=123), suggestion(url='  '),
    suggestion(url='http://records.example.gov'), suggestion(url='javascript:alert(1)'),
    suggestion(url='//records.example.gov/bridge'), suggestion(url='https:///bridge'),
    suggestion(url='https://user@records.example.gov/bridge'),
    suggestion(url='https://user:password@records.example.gov/bridge'),
    suggestion(url='https://records.example.gov:444/bridge'),
    suggestion(url='https://records.example.gov:invalid/bridge'),
    suggestion(url='https://[broken/bridge'),
    suggestion(url='https://not a host.example/bridge'),
])
def test_manual_links_omit_malformed_or_unjustified_suggestions(row):
    assert manual_check_links(summary_with([row])) == []


@pytest.mark.parametrize('rows', [None, 'https://records.example.gov', {}, 123])
def test_manual_links_require_a_list_of_suggestions(rows):
    assert manual_check_links(summary_with(rows)) == []


def test_manual_links_deduplicate_normalized_destinations_and_limit_valid_links():
    first = suggestion(name='  Town records  ', url='https://Records.Example.gov:443/bridge/',
                       description='  Check the dated opening notice.  ')
    rows = [None, first, suggestion(url='https://records.example.gov/bridge#date'),
            suggestion(url='https://records.example.gov/bridge/'),
            suggestion(name='Archived notices', url='https://archive.example.gov/bridge'),
            suggestion(name='Town department', url='https://town.example.gov/bridges'),
            suggestion(name='Fourth destination', url='https://fourth.example.gov/bridge')]
    data = summary_with(rows)
    before = deepcopy(data)
    assert manual_check_links(data) == [
        {**first, 'name': 'Town records', 'description': 'Check the dated opening notice.'},
        rows[4], rows[5]]
    assert data == before


@pytest.mark.parametrize('verdict,mode', [('true', None), ('false', None),
                                        ('unverified', 'media_only')])
def test_manual_links_are_only_for_unverified_claims(verdict, mode):
    data = summary_with([suggestion()])
    data.update(verdict=verdict, mode=mode)
    assert manual_check_links(data) == []


def report_with_links(verdict='unverified', mode=None):
    summary = summary_with([
        suggestion(name='Town records & notices', url='https://records.example.gov/bridge?a=1&b=2',
                   description='Check the <dated> opening notice and the bridge location.'),
        suggestion(name='No justification', description=''),
        suggestion(name='Unsafe destination', url='http://unsafe.example/bridge'),
    ])
    summary.update(verdict=verdict, mode=mode)
    summary['sources'] = [{'id': 17, 'title': 'Unreviewed town article',
                           'url': 'https://unreviewed.example/bridge',
                           'snippet': 'A town article describes the bridge scenery.'}]
    return {'id': 'manual-check-report', 'created_at': '2026-10-27', 'status': 'complete',
            'type': 'image' if mode == 'media_only' else 'text',
            'title': 'Bridge opening claim', 'content': 'The bridge opened in 1985.',
            'result': {'real_percentage': 61., 'fake_percentage': 39., 'confidence': 61.,
                       'report_summary': summary}}


def render_with_paragraphs(monkeypatch, renderer, report):
    module = importlib.import_module('backend.' + renderer)
    paragraphs = []
    original = module.Paragraph
    def paragraph(text, style, *args, **kwargs):
        paragraphs.append((text, style))
        return original(text, style, *args, **kwargs)
    monkeypatch.setattr(module, 'Paragraph', paragraph)
    if renderer == 'pdf_report':
        pdf = module.report_pdf(report, detailed=True)
    elif renderer == 'compact_pdf':
        pdf = module.compact_report_pdf(report)
    else:
        pdf = module.concise_report_pdf(report, Table)
    assert pdf.startswith(b'%PDF')
    return paragraphs


@pytest.mark.parametrize('renderer', ['pdf_report', 'compact_pdf', 'concise_pdf'])
def test_pdf_manual_links_are_highlighted_with_readable_justifications(monkeypatch, renderer):
    report = report_with_links()
    before = deepcopy(report)
    paragraphs = render_with_paragraphs(monkeypatch, renderer, report)
    links = [(text, style) for text, style in paragraphs if style.name == 'CheckLink']
    reasons = [(text, style) for text, style in paragraphs if style.name == 'CheckReason']
    expected = report['result']['report_summary']['recommended_sources'][0]
    assert len(links) == len(reasons) == 1
    link, link_style = links[0]
    reason, reason_style = reasons[0]
    assert f'href="{escape(expected["url"], quote=True)}"' in link
    assert '<u>' + escape(expected['name']) + '</u>' in link
    assert reason == escape(expected['description'])
    assert link_style.fontName == 'Helvetica-Bold'
    assert link_style.fontSize >= 12 and link_style.backColor is not None
    assert link_style.borderWidth > 0
    assert reason_style.fontSize >= 10.5
    all_text = '\n'.join(text for text, _ in paragraphs)
    assert 'Where to check this claim' in all_text
    assert 'No justification' not in all_text and 'Unsafe destination' not in all_text
    assert 'href="https://unreviewed.example/bridge"' not in all_text
    assert report == before


@pytest.mark.parametrize('renderer', ['pdf_report', 'compact_pdf', 'concise_pdf'])
@pytest.mark.parametrize('verdict,mode', [('true', None), ('false', None),
                                        ('unverified', 'media_only')])
def test_pdf_omits_manual_suggestions_for_settled_or_media_only_reports(monkeypatch, renderer, verdict, mode):
    paragraphs = render_with_paragraphs(monkeypatch, renderer, report_with_links(verdict, mode))
    all_text = '\n'.join(text for text, _ in paragraphs)
    assert 'Where to check this claim' not in all_text
    assert 'Town records &amp; notices' not in all_text
    assert not any(style.name == 'CheckReason' for _, style in paragraphs)


def test_detailed_pdf_keeps_unreviewed_source_information_without_a_clickable_link(monkeypatch):
    paragraphs = render_with_paragraphs(monkeypatch, 'pdf_report', report_with_links())
    text = '\n'.join(value for value, _ in paragraphs)
    assert 'Source [17]: Unreviewed town article' in text
    assert 'Original source passage: A town article describes the bridge scenery.' in text
    assert 'Unreviewed legacy source:' in text
    assert 'href="https://unreviewed.example/bridge"' not in text
