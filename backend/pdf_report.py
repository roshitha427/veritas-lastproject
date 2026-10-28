from io import BytesIO
from html import escape
import re
from urllib.parse import urlparse
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from backend.report_contract import build_report_contract
from backend.report_metrics import model_metrics, format_percentage
from backend.report_links import manual_check_links
from backend.pdf_media import image_preview_flowables
from backend.report_style import report_styles, report_page_decoration
from backend.report_summary import general_knowledge_explanation


def _model_note(value):
    """Keep benchmark caveats without adding evaluation metrics to the report."""
    text = re.sub(r'\bscored \d+(?:\.\d+)?% on\b', 'was evaluated on', str(value))
    return re.sub(r'(?<=requested )\d+(?:\.\d+)?%\s+(?=held-out accuracy target\b)', '', text)


def report_pdf(report, detailed=True):
    recorded_contract = (report.get('result') or {}).get('report_contract') or {}
    if not detailed and isinstance(recorded_contract, dict) and recorded_contract.get('schema_version', 0) >= 1:
        from backend.compact_pdf import compact_report_pdf
        return compact_report_pdf(report)
    contract = build_report_contract(report)
    summary = (report.get('result') or {}).get('report_summary') or {}
    origin = (report.get('result') or {}).get('media_origin') or {}
    inconclusive = report.get('type') in ('image', 'video') and origin.get('verdict') == 'inconclusive'
    unverified = summary.get('mode') != 'media_only' and summary.get('verdict') == 'unverified'
    if not detailed and (inconclusive or unverified):
        from backend.concise_pdf import concise_report_pdf
        return concise_report_pdf(report, Table)
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=(595, 842), rightMargin=45, leftMargin=45, topMargin=62, bottomMargin=52)
    styles = report_styles()
    story = [Paragraph('Verification report', styles['Title']),
             Spacer(1, 8)]
    def p(text, style='Normal'):
        story.append(Paragraph(escape(str(text)), styles[style]))
        story.append(Spacer(1, 8))
    def p_with_citations(text, sources, style='Normal'):
        source_urls = {source['id']: source['url'] for source in contract['sources']
                       if source['accepted'] is True and source['id'] not in contract['verification']['unresolved_source_ids']}
        parts, cursor = [], 0
        for match in re.finditer(r'\[(\d+)\]', str(text)):
            parts.append(escape(str(text)[cursor:match.start()]))
            source_id = int(match.group(1))
            url = source_urls.get(source_id)
            label = escape(match.group(0))
            parts.append(f'<link href="{escape(url, quote=True)}" color="#11594e">{label}</link>' if url else label)
            cursor = match.end()
        parts.append(escape(str(text)[cursor:]))
        story.append(Paragraph(''.join(parts), styles[style]))
        story.append(Spacer(1, 8))
    p('Report ID: ' + report['id'], 'Small')
    p('Created: ' + report['created_at'], 'Small')
    p('Content type: ' + report['type'], 'Small')
    p('Report snapshot: ' + contract['revision'], 'Small')
    original = report.get('content', '')
    if report.get('type') in ('image', 'video'):
        if report.get('filename'):
            p('Uploaded media file: ' + report['filename'])
        digest = (report.get('result') or {}).get('sha256')
        if digest:
            p('Uploaded media SHA-256: ' + digest)
        if original:
            p('Submitted claim / caption', 'Heading2')
            p(original)
    elif original and not original.isascii():
        p('Original content includes Unicode text. Read the full original in the web report.', 'Heading2')
    else:
        p(report.get('title', ''), 'Heading2')
    result = report.get('result') or {}
    media_scores_available = (result.get('media_origin') or {}).get('verdict') in (
        'ai_generated', 'likely_real', 'likely_not_ai_generated')
    media_only = (result.get('report_summary') or {}).get('mode') == 'media_only'
    inspection = result.get('media_inspection') or {}
    def ocr_findings():
        ocr = result.get('ocr_review') or {}
        original_ocr = contract['claim']['original_ocr_text']
        if not original_ocr and not ocr.get('user_reviewed'):
            return
        p('OCR findings (transcription, separate from factual verification)', 'Heading2')
        if ocr.get('user_reviewed'):
            p('User-reviewed text used for verification')
            p(contract['claim']['reviewed_ocr_text'] or 'No OCR text selected for verification.')
            if ocr.get('changed'):
                p('Original automatic transcription: ' + original_ocr)
        else:
            p('Automatic transcription: ' + original_ocr)
        if inspection.get('ocr_uncertain'):
            p('Some original image text was uncertain; compare the transcription with the uploaded image.')
        for span in inspection.get('uncertain_spans') or []:
            p(str(span.get('text') or '') + ': ' + str(span.get('reason') or ''))
    verification = result.get('verification', {})
    def evidence_summary():
        if not summary:
            return
        p('Visual media assessment' if media_only else 'Claim verification', 'Heading2')
        if contract['claim']['checked_text']:
            p('Claim checked: ' + contract['claim']['checked_text'])
        if not media_only:
            p('Verdict: ' + str(summary.get('verdict', 'unverified')).replace('_', ' ').upper(), 'Heading3')
        if summary.get('headline'):
            p(summary['headline'], 'Heading3')
        summary_sources = summary.get('sources') or verification.get('evidence', [])
        p_with_citations(summary.get('summary', 'No source-grounded summary is available.'), summary_sources)
        if summary.get('reason'):
            p_with_citations('Reason: ' + str(summary['reason']), summary_sources)
        if summary.get('verification_timestamp'):
            p('Checked: ' + str(summary['verification_timestamp']))
        if summary.get('summary_source_ids') or summary.get('source_ids'):
            ids = summary.get('summary_source_ids') or summary.get('source_ids')
            p('Sources used for the summary: ' + ', '.join(f'[{number}]' for number in ids))
        for point in summary.get('key_points', []):
            ids = ', '.join(f'[{number}]' for number in point.get('source_ids', []))
            p_with_citations(f"- {point.get('text', '')}" + (f' Sources: {ids}' if ids else ''), summary_sources)
        for claim in summary.get('claims', []):
            p('Claim assessment: ' + str(claim.get('verdict', 'unverified')).upper(), 'Heading3')
            p(claim.get('text', ''))
            p_with_citations(claim.get('reason', ''), summary_sources)
            if claim.get('supporting_evidence'):
                p('Supporting source IDs: ' + ', '.join(f'[{number}]' for number in claim['supporting_evidence']))
            if claim.get('contradicting_evidence'):
                p('Contradicting source IDs: ' + ', '.join(f'[{number}]' for number in claim['contradicting_evidence']))
            if claim.get('missing_evidence'):
                p('Evidence still needed: ' + '; '.join(claim['missing_evidence']))
        review = summary.get('verification_review') or {}
        if not media_only and summary.get('verdict') == 'unverified' and review:
            reasons = list(dict.fromkeys(review.get('insufficiency_reasons') or []))
            needed = list(dict.fromkeys(review.get('additional_evidence_needed') or []))
            if reasons:
                p('Why the evidence is insufficient', 'Heading3')
                for reason in reasons:
                    p('- ' + str(reason))
            if needed:
                p('Evidence needed to settle the claim', 'Heading3')
                for requirement in needed:
                    p('- ' + str(requirement))
        if report.get('type') == 'text':
            p(general_knowledge_explanation(summary))
        retrieval = summary.get('rag_analysis') or {}
        if retrieval:
            p('Retrieval summary', 'Heading3')
            p(f"Retrieved excerpts: {retrieval.get('retrieved_count', 0)}")
            p(retrieval.get('scope', ''))
        for limitation in summary.get('limitations', []):
            p('Limitation: ' + _model_note(limitation))
        if not media_only:
            p('The verdict is limited to the retrieved sources listed in this report. An unverified result means the available evidence did not settle the claim.')
        for warning in contract['warnings']:
            p(warning)
    claim_media = report.get('type') in ('image', 'video') and not media_only and bool(
        original or result.get('claim_text') or summary.get('claims')
        or summary.get('verdict') in ('true', 'false'))
    if claim_media or report.get('type') == 'text':
        evidence_summary()
    if claim_media:
        story.extend(image_preview_flowables(report, styles))
        ocr_findings()
    elif report.get('type') in ('image', 'video'):
        story.extend(image_preview_flowables(report, styles))
    p('Model metrics', 'Heading2')
    rows = [['Measure', 'Result']] + [[label, format_percentage(value)] for label, value in model_metrics(report)]
    table = Table(rows, colWidths=[320, 175])
    table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#11594e')),
                              ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                              ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                              ('TOPPADDING', (0, 0), (-1, -1), 12),
                              ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.HexColor('#d9e5e1'))]))
    story.extend([table, Spacer(1, 18)])
    if report.get('type') == 'text':
        p('Real news pattern score and Fake news pattern score measure learned news-writing patterns. '
          'Text model confidence is the strongest class score; it is not measured accuracy or verified factual certainty.')
        ai = result.get('ai_assessment') or {}
        if ai.get('status') == 'too_short':
            p('The AI-writing detector needs input containing English words.')
        elif ai.get('status') == 'unsupported_language':
            p('The AI-writing detector currently supports English only.')
        elif result.get('ai_percentage') is not None:
            p(f"The AI-writing detector reviews up to {ai.get('max_words', 800)} words.")
            if ai.get('experimental_input'):
                p('This short-text AI-writing estimate is experimental and outside the validated article-length range.')
    inspection = result.get('media_inspection') or {}
    if report.get('type') in ('image', 'video'):
        labels = result.get('media_labels') or {}
        if labels.get('origin_label'):
            p(labels['origin_label'], 'Heading2')
            p('The supplied reference label is separate from model scores.')
        if labels.get('style'):
            p('Visual style: ' + labels['style'])
        if labels.get('manipulation'):
            p(labels['manipulation'])
        p('Scene description and visible features', 'Heading2')
        p(inspection.get('description') or 'Scene description unavailable for this analysis.')
        for feature in inspection.get('visible_features') or []:
            p('- ' + feature)
        for action in inspection.get('observed_actions') or []:
            p('Observed action: ' + action)
        origin = result.get('media_origin') or {}
        p('Origin assessment: ' + str(origin.get('verdict', 'not assessed')))
        p(origin.get('reason', ''))
        credentials = result.get('content_credentials') or {}
        if credentials.get('ai_declared'):
            p('Embedded Content Credentials', 'Heading2')
            p(credentials.get('message', ''))
            p('Declaration scope: ' + str(credentials.get('scope', 'unknown')))
            if credentials.get('validation_codes'):
                p('Credential validation issues: ' + ', '.join(credentials['validation_codes']))
        temporal = result.get('temporal_assessment') or {}
        if temporal.get('status') == 'ready':
            p('Temporal manipulation analysis', 'Heading2')
            p('Status: ready')
            if temporal.get('sampled_frames') is not None:
                p(f"Sampled frames: {temporal['sampled_frames']}")
            p(temporal.get('scope', ''))
        media_text = result.get('media_text_analysis') or {}
        if media_text:
            p('Separate analysis of extracted text', 'Heading2')
            p(media_text.get('content', ''))
            news = media_text.get('text_model_assessment') or {}
            p('Text classifier status: ' + str(news.get('status', 'not assessed')))
            if news.get('status') != 'assessed':
                p(news.get('reason', ''))
            p('Extracted text is checked separately against sources. Its writing patterns do not establish factual truth.')
    text_model = result.get('text_model_assessment') or {}
    if report.get('type') == 'text' and text_model:
        p('Local text classifier output', 'Heading2')
        p('Status: ' + str(text_model.get('status', 'not assessed')))
        if text_model.get('status') != 'assessed':
            p(text_model.get('reason', 'The text classifier did not produce a prediction.'))
        elif text_model.get('experimental_input'):
            p('This short input is outside the article benchmark; its model scores are experimental.')
        evaluation = text_model.get('evaluation') or {}
        if evaluation.get('accuracy') is not None:
            p('The classifier was evaluated on a held-out dataset. That evaluation does not establish claim-verification accuracy.')
    if not claim_media and report.get('type') != 'text':
        evidence_summary()
    ai = result.get('ai_assessment', {})
    if ai.get('status') == 'experimental':
        p('Experimental AI-generation estimate', 'Heading2')
        if ai.get('scope') == 'text':
            p('This local classifier was trained on historical human and ChatGPT answers. Its benchmark does not establish accuracy on current generators, news writing or other languages, and it cannot prove authorship or factual truth.')
        else:
            p('This is an experimental image-model score. Calibration, when available, applies to its validation dataset. It cannot prove AI authorship or factual truth.')
        if ai.get('scope') == 'sampled_video_frames':
            p(f"The score averages {ai.get('sample_count')} sampled video frames. Audio and generated motion are not assessed by this model.")
    if not media_only:
        p('What was checked', 'Heading2')
        if verification.get('attribution_url'):
            p('Data attribution: ' + verification['attribution_url'])
        if verification.get('search_query'):
            p('Evidence search terms: ' + verification['search_query'])
        p('Retrieved source count: ' + str(len(verification.get('evidence', []))))
        p('Status: ' + verification.get('status', 'unverified'))
        p(verification.get('explanation', ''))
        p('Checked at: ' + verification.get('checked_at', 'Not checked'))
        if summary.get('verdict') == 'unverified':
            suggestions = manual_check_links(summary)
            if suggestions:
                p('Where to check this claim', 'Heading2')
                p('Suggested places to check; these links have not verified this claim.')
                for item in suggestions:
                    story.append(Paragraph(f'<link href="{escape(item["url"], quote=True)}" color="#4a358c"><u>{escape(item["name"])}</u></link>', styles['CheckLink']))
                    story.append(Spacer(1, 6))
                    p(item['description'], 'CheckReason')
    if (result.get('media_origin') or {}).get('basis') == 'content_credentials':
        p('Classifier metrics are raw pattern signals, separate from the embedded content declaration and its validation status.')
    elif report.get('type') in ('image', 'video') and not media_scores_available:
        p('Origin remains unresolved. The real and AI pattern scores are experimental visual signals, not reliable origin probabilities. Model confidence is the larger class score, not measured accuracy.')
    elif result.get('metric_note'):
        p(result['metric_note'], 'Heading3')
    if result.get('confidence_note') and (report.get('type') == 'text' or media_scores_available):
        p(result['confidence_note'])
    if result.get('observations'):
        p('Content observations', 'Heading2')
        for observation in result['observations']:
            p(f"{observation.get('label', 'Observation')}: {observation.get('value', '')}")
    visual = result.get('visual_model_assessment') or {}
    video = result.get('video_assessment') or {}
    if video.get('status') == 'experimental':
        p('Experimental video model', 'Heading2')
        p('Status: experimental')
        p(video.get('note', ''))
        p(video.get('scope', ''))
        if video.get('training_example'):
            p('This video is a training example. Its assessment describes training fit, not independent accuracy.')
    if visual.get('status') == 'ready':
        p('Experimental manipulation analysis', 'Heading2')
        p('Visual classifier status: ready')
        p(visual.get('scope', ''))
    if result.get('frames'):
        p('Sampled video frames', 'Heading2')
        for frame in result['frames']:
            p(f"Frame reviewed at {frame.get('seconds', '?')} seconds.")
    for item in contract['sources']:
        source_number = item['id']
        if item['accepted'] is False:
            p(f"Excluded from evidence: {item.get('title', '')} [{source_number}]", 'Heading3')
            p(item['acceptance_reason'])
            continue
        p(f"Source [{source_number}]: {item.get('title', '')}", 'Heading3')
        if item['accepted'] is None:
            p('Unreviewed legacy source: ' + item['acceptance_reason'])
        elif not item.get('citation_usable', True):
            p('This reference is ambiguous or unavailable as a unique verified citation.')
        else:
            claims = contract['verification']['claims']
            uses = []
            if any(source_number in row['supporting_source_ids'] for row in claims):
                uses.append('supporting evidence')
            if any(source_number in row['contradicting_source_ids'] for row in claims):
                uses.append('contradicting evidence')
            if not uses:
                uses.append('related context; does not settle the claim')
            p('Use in this assessment: ' + '; '.join(uses) + '.')
        fields = [('source_claim', 'Claim reviewed by this source'), ('publisher', 'Publisher'),
                  ('reliability', 'Source category'), ('rating', 'Published fact-check rating'),
                  ('published_at', 'Publication date'), ('reviewed_at', 'Review date'),
                  ('retrieved_at', 'Retrieved'), ('excerpt', 'Original source passage'),
                  ('relationship', 'Source context')]
        for key, label in fields:
            value = item.get(key)
            if value not in (None, ''):
                p(f'{label}: ' + (str(value).replace('_', ' ') if key == 'reliability' else str(value)))
        source_url = item.get('url', '')
        if (source_url and urlparse(source_url).scheme == 'https' and urlparse(source_url).hostname
                and (summary.get('verdict') != 'unverified' or (item['accepted'] is True and item.get('citation_usable', True)))):
            safe_url = escape(source_url, quote=True)
            story.append(Paragraph(f'<link href="{safe_url}" color="#4a358c"><u>Open original source: {escape(source_url)}</u></link>', styles['CheckLink']))
            story.append(Spacer(1, 8))
    if result.get('speech'):
        p('Video speech transcription', 'Heading2')
        p('Status: ' + result['speech'].get('status', 'unavailable'))
        transcript = result.get('transcript', '')
        if transcript and transcript.isascii():
            p(transcript)
        elif transcript:
            p('The transcript includes Unicode text. Read the complete transcript in the web report.')
        assessment = result.get('speech_text_assessment', {})
        if assessment.get('model'):
            p('Spoken-text classifier status: ' + str(assessment['model'].get('status', 'not assessed')))
        p('Automatic transcripts may contain errors. Review the words against the recording.')
    p('Model and interpretation', 'Heading2')
    p(result.get('scope', 'No model assessment available.'))
    for item in result.get('limitations', []):
        p(_model_note(item))
    for item in result.get('models', []):
        p(f"{item.get('name')} | version {item.get('version')} | {item.get('status')}")
        if item.get('source_url'):
            p('Model source: ' + item['source_url'])
        for benchmark, external in item.get('external_evaluations', {}).items():
            p(f"External benchmark: {benchmark}. Historical article evaluation does not establish claim-verification accuracy.")
        evaluation = item.get('evaluation')
        if evaluation:
            p('Held-out evaluation applies only to the recorded test dataset. It does not establish accuracy on arbitrary submissions.')
    if report.get('type') == 'text' and original and original.isascii():
        p('Original submitted text', 'Heading2')
        p(original)
    if report.get('type') in ('image', 'video'):
        p('Visual-origin scores are experimental estimates. They do not establish whether an event or caption is true. Validation calibration does not guarantee accuracy on other image types.')
        inspection = result.get('media_inspection') or {}
        origin = result.get('media_origin') or {}
        verdict_labels = {'ai_content_declared': 'AI-generated content declared; see credential validation',
                          'ai_generated': 'Likely AI-generated', 'likely_real': 'Real image' if report.get('type') == 'image' else 'Likely real / not flagged as AI-generated',
                          'inconclusive': 'Origin inconclusive',
                          'classification_unavailable': 'Origin classification unavailable'}
        if origin.get('verdict'):
            p('AI-origin classification: ' + verdict_labels.get(origin['verdict'], 'Classification unavailable'))
            if origin.get('reason'):
                p(origin['reason'])
        if inspection.get('description'):
            p('Content description: ' + inspection['description'])
        if inspection.get('visual_artifacts'):
            p('Visible details to review: ' + inspection['visual_artifacts'])
        if not claim_media:
            ocr_findings()
    else:
        p('Confidence is a model probability, not measured accuracy or a guarantee of truth. Fake and real are complementary model classes; AI generation is a separate assessment.')
    if report.get('language') in ('hi', 'te'):
        p('This PDF uses an English summary. See the web report for the original Unicode content.')
    decorate = report_page_decoration(report)
    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
    return buffer.getvalue()
