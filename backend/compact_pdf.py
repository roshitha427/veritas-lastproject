"""Default report export: one verification, one related metric table and links."""
from html import escape
from io import BytesIO
from datetime import datetime, timezone
from urllib.parse import urlparse

from reportlab.lib import colors
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from backend.pdf_media import image_preview_flowables
from backend.report_contract import build_report_contract
from backend.report_metrics import model_metrics, format_percentage
from backend.report_links import manual_check_links
from backend.report_style import report_page_decoration, report_styles
from backend.report_summary import general_knowledge_explanation


def _brief(value, limit):
    text = ' '.join(str(value or '').split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(' ', 1)[0].rstrip('.,;:') + '...'


def _percentage(value):
    return format_percentage(value)


def _when(value):
    text = str(value or '')
    try:
        stamp = datetime.fromisoformat(text.replace('Z', '+00:00'))
        if 'T' not in text and ' ' not in text:
            return stamp.strftime('%d %b %Y')
        if stamp.tzinfo:
            return stamp.astimezone(timezone.utc).strftime('%d %b %Y, %H:%M UTC')
        return stamp.strftime('%d %b %Y, %H:%M')
    except ValueError:
        return _brief(text, 80)


def related_metrics(report, contract):
    """Use the same three classifier fields for every report type."""
    return ('Model metrics', model_metrics(report),
            'Model estimates do not verify factual claims. Confidence is not measured accuracy.')


def compact_report_pdf(report):
    contract = build_report_contract(report)
    result = report.get('result') or {}
    summary = result.get('report_summary') or {}
    verification = contract['verification']
    styles = report_styles()
    story = []

    def p(text, style='Normal'):
        if text:
            story.extend([Paragraph(escape(str(text)), styles[style]), Spacer(1, 5)])

    p('Verification report', 'Title')
    p('Report: ' + str(report.get('id') or '') + ' | ' + _when(report.get('created_at')), 'Small')
    if contract['mode'] == 'media_only':
        origin = result.get('media_origin') or {}
        p(origin.get('headline') or 'Media origin not established', 'Heading2')
        p(_brief(origin.get('reason') or summary.get('reason') or summary.get('summary'), 360))
    else:
        verdict = verification.get('verdict')
        label = str(verdict).upper() if verdict in ('true', 'false', 'unverified') else 'UNVERIFIED'
        p('Verification: ' + label, 'Heading2')
        p(_brief(contract['claim']['checked_text'], 280))
        review = summary.get('verification_review') or {}
        reason = (review.get('summary') if verdict == 'unverified' else None) or verification.get('reason') or verification.get('summary') or 'The available evidence does not settle this claim.'
        p(_brief(reason, 360))
        if report.get('type') == 'text':
            p(general_knowledge_explanation(summary))
        if verification.get('checked_at'):
            p('Checked: ' + _when(verification['checked_at']), 'Small')
        if verdict == 'unverified':
            checked = []
            for source in review.get('checked_sources') or []:
                if not isinstance(source, dict):
                    continue
                label = source.get('publisher') or source.get('title')
                if not label:
                    try:
                        label = urlparse(str(source.get('url') or '')).hostname
                    except ValueError:
                        label = None
                if label and label not in checked:
                    checked.append(str(label))
            if checked:
                names = '; '.join(_brief(label, 65) for label in checked[:3])
                p('Sources checked: ' + names + (f'; {len(checked) - 3} more in the full audit.' if len(checked) > 3 else '.'), 'Small')
            needed = review.get('additional_evidence_needed') or summary.get('missing_evidence') or []
            if isinstance(needed, list) and needed:
                p('Evidence needed: ' + _brief(needed[0], 280))

    image = image_preview_flowables(report, styles)
    for element in image:
        if hasattr(element, 'drawWidth') and hasattr(element, 'drawHeight'):
            scale = min(1, 288 / element.drawWidth, 160 / element.drawHeight)
            element.drawWidth *= scale
            element.drawHeight *= scale
    story.extend(image)

    heading, metrics, scope = related_metrics(report, contract)
    p(heading, 'Heading2')
    rows = [['Measure', 'Result']] + [[label, _percentage(value)] for label, value in metrics]
    table = Table(rows, colWidths=[335, 160])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#7054da')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TOPPADDING', (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
        ('LINEBELOW', (0, 0), (-1, -1), .5, colors.HexColor('#e6e1f0')),
    ]))
    story.extend([table, Spacer(1, 7)])
    p(scope, 'Small')

    sources = [source for source in contract['sources'] if source['accepted'] is True
               and source.get('citation_usable', True) and source.get('url')]
    verdict = verification.get('verdict')
    if verdict in ('true', 'false'):
        key = 'supporting_source_ids' if verdict == 'true' else 'contradicting_source_ids'
        outcome_ids = {number for claim in verification['claims'] if claim.get('verdict') == verdict
                       for number in claim.get(key, [])}
        summary_ids = summary.get('summary_source_ids')
        summary_ids = summary_ids if isinstance(summary_ids, list) else []
        primary_ids = outcome_ids or {number for number in summary_ids if type(number) is int and number > 0}
        sources = [source for source in sources if source['id'] in primary_ids]
    sources.sort(key=lambda source: not source['cited'])
    if sources:
        p('Related sources', 'Heading2')
        for source in sources[:3]:
            label = f"[{source['id']}] " + _brief(source.get('publisher') or source.get('title') or source['url'], 90)
            story.append(Paragraph(f'<link href="{escape(source["url"], quote=True)}" color="#53428b">{escape(label)}</link>', styles['Normal']))
            story.append(Spacer(1, 4))
    elif contract['mode'] != 'media_only':
        p('No usable source links were recorded for this claim.', 'Small')
    if verification.get('unresolved_source_ids'):
        p('Some saved citations are unavailable or excluded; see the detailed audit.', 'Small')
    suggestions = manual_check_links(summary)
    if suggestions:
        p('Where to check this claim', 'Heading2')
        for item in suggestions:
            story.append(Paragraph(f'<link href="{escape(item["url"], quote=True)}" color="#4a358c"><u>{escape(item["name"])}</u></link>', styles['CheckLink']))
            story.append(Spacer(1, 6))
            p(item['description'], 'CheckReason')
    p('Detailed evidence, OCR and module diagnostics are available in the full audit export.', 'Small')

    output = BytesIO()
    decoration = report_page_decoration(report)
    SimpleDocTemplate(output, pagesize=(595, 842), leftMargin=45, rightMargin=45,
                      topMargin=62, bottomMargin=52).build(story, onFirstPage=decoration, onLaterPages=decoration)
    return output.getvalue()
