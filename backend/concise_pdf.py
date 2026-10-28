"""Short user-facing exports for media and unverified claims."""
from html import escape
from io import BytesIO
from urllib.parse import urlparse

from reportlab.lib import colors
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, TableStyle
from backend.pdf_media import image_preview_flowables
from backend.report_contract import build_report_contract
from backend.report_metrics import model_metrics, format_percentage
from backend.report_links import manual_check_links
from backend.report_style import report_styles, report_page_decoration
from backend.report_summary import general_knowledge_explanation


def concise_report_pdf(report, table_factory):
    buffer = BytesIO()
    styles = report_styles()
    story = []
    result = report.get('result') or {}
    contract = build_report_contract(report)
    summary = result.get('report_summary') or {}
    review = summary.get('verification_review') or {}
    media = report.get('type') in ('image', 'video')
    origin = result.get('media_origin') or {}
    inspection = result.get('media_inspection') or {}
    ocr_review = result.get('ocr_review') or {}
    has_claim = media and summary.get('mode') != 'media_only' and bool(
        report.get('content') or result.get('claim_text') or review.get('claim')
        or summary.get('verdict') in ('true', 'false'))

    def p(text, style='Normal'):
        story.extend([Paragraph(escape(str(text)), styles[style]), Spacer(1, 8)])

    def cited(text):
        usable = {source['id']: source['url'] for source in contract['sources']
                  if source['accepted'] is True and source['id'] not in contract['verification']['unresolved_source_ids']}
        import re
        parts, cursor = [], 0
        for match in re.finditer(r'\[(\d+)\]', str(text or '')):
            parts.append(escape(str(text)[cursor:match.start()]))
            url = usable.get(int(match[1]))
            label = escape(match[0])
            parts.append(f'<link href="{escape(url, quote=True)}" color="#11594e">{label}</link>' if url else label)
            cursor = match.end()
        parts.append(escape(str(text or '')[cursor:]))
        story.extend([Paragraph(''.join(parts), styles['Normal']), Spacer(1, 8)])

    def table(rows):
        widget = table_factory(rows, colWidths=[320, 175])
        widget.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#11594e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('TOPPADDING', (0, 0), (-1, -1), 8), ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LINEBELOW', (0, 0), (-1, -1), .5, colors.HexColor('#d9e5e1'))]))
        story.extend([widget, Spacer(1, 12)])

    p('Verification report', 'Title')
    p(report.get('title') or report.get('filename') or 'Analysis result', 'Heading2')
    p('Report: ' + report['id'] + ' | ' + report['created_at'], 'Small')
    p('Report snapshot: ' + contract['revision'])
    def media_details():
        p('Description', 'Heading2')
        description = str((result.get('media_inspection') or {}).get('description') or 'Description unavailable.')
        p(description[:700] + ('…' if len(description) > 700 else ''))
        labels = {'inconclusive': 'Origin inconclusive', 'classification_unavailable': 'Origin not assessed',
                  'ai_generated': 'Likely AI-generated', 'likely_real': 'Likely real image',
                  'ai_content_declared': 'AI-generated content declared'}
        label = labels.get(origin.get('verdict'), 'Origin not assessed')
        if origin.get('basis') == 'content_credentials' and not origin.get('credential_verified'):
            label += ' (unverified declaration)'
        labels = result.get('media_labels') or {}
        p(labels.get('origin_label') or label)
        if labels.get('style'):
            p('Visual style: ' + labels['style'])
        if labels.get('manipulation'):
            p(labels['manipulation'])
        if labels.get('confirmed_origin'):
            p('Origin source: supplied reference label. Model scores are separate.')
        if origin.get('verdict') == 'inconclusive' and not labels.get('confirmed_origin'):
            p('The origin could not be confirmed. The metrics below summarize the visual assessment.')
        p('Model metrics', 'Heading2')
        table([['Measure', 'Result']] + [[label, format_percentage(value)] for label, value in model_metrics(report)])
        p('Image-origin scores estimate visual patterns. They do not establish whether the factual claim is true or false.')
        video = result.get('video_assessment') or {}
        if video.get('status') == 'experimental':
            p('Video model status: experimental')
            p(video.get('scope', ''))
            if video.get('training_example'):
                p('Training example: training fit, not independent accuracy.')
        temporal = result.get('temporal_assessment') or {}
        if temporal.get('status') == 'ready':
            p('Temporal manipulation analysis', 'Heading2')
            p('Status: ready')
            p(temporal.get('scope', ''))
        elif temporal.get('reason'):
            p('Temporal assessment: ' + str(temporal['reason']))
        media_text = result.get('media_text_analysis') or {}
        if media_text:
            p('Separate analysis of extracted text', 'Heading2')
            p(media_text.get('content', ''))
            news = media_text.get('text_model_assessment') or {}
            p('Text classifier status: ' + str(news.get('status', 'not assessed')))
            if news.get('status') != 'assessed':
                p(news.get('reason', ''))
            p('Text-pattern, writing-origin and temporal scores are separate model signals; none establishes the factual claim.')
    if media and not has_claim:
        story.extend(image_preview_flowables(report, styles))
        media_details()
    if not media:
        p('UNVERIFIED', 'Heading2')
        p(review.get('summary') or 'The available sources do not confirm or refute this claim.')
        general = general_knowledge_explanation(summary)
        if general:
            p(general)
    def text_model_details():
        p('Model metrics', 'Heading2')
        table([['Measure', 'Result']] + [[label, format_percentage(value)] for label, value in model_metrics(report)])
        p('Pattern scores, not verified facts.')
    if has_claim:
        verdict = summary.get('verdict') if summary.get('verdict') in ('true', 'false') else 'unverified'
        p('Claim: ' + verdict.upper(), 'Heading2')
        p('Claim checked: ' + contract['claim']['checked_text'])
        cited((review.get('summary') or 'The available sources do not confirm or refute this claim.') if verdict == 'unverified'
          else summary.get('summary', ''))
        if summary.get('reason'):
            cited('Reason: ' + str(summary['reason']))
        for claim in contract['verification']['claims']:
            p('Claim assessment: ' + str(claim['verdict']).upper(), 'Heading3')
            p(claim['text'])
            cited(claim['reason'])
            if claim['supporting_source_ids']:
                cited('Supporting evidence: ' + ', '.join(f'[{number}]' for number in claim['supporting_source_ids']))
            if claim['contradicting_source_ids']:
                cited('Contradicting evidence: ' + ', '.join(f'[{number}]' for number in claim['contradicting_source_ids']))
            for needed in claim['missing_evidence']:
                p('Evidence still needed: ' + str(needed))
    def ocr_findings():
        if not media or not (inspection.get('ocr_performed') or ocr_review.get('user_reviewed')):
            return
        p('OCR findings (transcription, not factual verification)', 'Heading2')
        p('User-reviewed text used for verification' if ocr_review.get('user_reviewed') else 'Automatic transcription')
        p(ocr_review.get('corrected_text', result.get('ocr_text', inspection.get('visible_text', ''))))
        if ocr_review.get('changed'):
            p('Original automatic transcription: ' + str(ocr_review.get('original_text') or ''))
        if inspection.get('ocr_uncertain'):
            p('Some original image text was uncertain. Review the transcription against the original image.')
        for span in inspection.get('uncertain_spans') or []:
            p(str(span.get('text', '')) + ': ' + str(span.get('reason', '')))
    if not has_claim:
        ocr_findings()
    if review:
        if not has_claim:
            p('Claim checked: ' + contract['claim']['checked_text'])
        p('Why this remains unverified', 'Heading2')
        for reason in review.get('insufficiency_reasons') or []:
            p('- ' + reason)
        p('Evidence needed to settle the claim', 'Heading2')
        for needed in review.get('additional_evidence_needed') or []:
            p('- ' + needed)
        p('What was checked', 'Heading2')
        p(review.get('scope_note', ''))
        for provider in review.get('providers') or []:
            p(str(provider.get('name', 'Source search')) + ': ' + str(provider.get('status', 'not recorded')).replace('_', ' '))
        for search in review.get('searches') or []:
            p('Search: ' + str(search.get('claim') or search.get('search_query') or '') + ' — ' + str(search.get('provider_status', 'not recorded')).replace('_', ' '))
        for source in review.get('checked_sources') or []:
            p(('Relevant passage: ' if source.get('accepted') else 'Excluded from evidence: ') + str(source.get('title') or source.get('publisher') or source.get('url')))
            p(source.get('reason', ''))
    for warning in contract['warnings']:
        p(warning)
    sources = [row for row in contract['sources'] if row['accepted'] is True]
    links = []
    for source in sources:
        url = source.get('url') or ''
        parsed = urlparse(url)
        if parsed.scheme == 'https' and parsed.hostname and not parsed.username and not parsed.password:
            links.append((url, f"[{source['id']}] " + str(source.get('publisher') or source.get('title') or parsed.hostname)))
        if len(links) == 3:
            break
    if links:
        p('Sources to review', 'Heading2')
        for url, label in links:
            story.append(Paragraph(f'<link href="{escape(url, quote=True)}" color="#11594e">{escape(label)}</link>', styles['Normal']))
            story.append(Spacer(1, 6))
    if summary.get('verdict') == 'unverified' and summary.get('mode') != 'media_only':
        suggestions = manual_check_links(summary)
        if suggestions:
            p('Where to check this claim', 'Heading2')
            p('Suggested places to check; these links have not verified this claim.')
            for item in suggestions:
                story.append(Paragraph(f'<link href="{escape(item["url"], quote=True)}" color="#4a358c"><u>{escape(item["name"])}</u></link>', styles['CheckLink']))
                p(item['description'], 'CheckReason')
                story.append(Spacer(1, 6))
    if has_claim:
        story.extend(image_preview_flowables(report, styles))
        ocr_findings()
        media_details()
    if not media:
        text_model_details()
    decorate = report_page_decoration(report)
    SimpleDocTemplate(buffer, pagesize=(595, 842), leftMargin=45, rightMargin=45,
                      topMargin=62, bottomMargin=52).build(story, onFirstPage=decorate, onLaterPages=decorate)
    return buffer.getvalue()
