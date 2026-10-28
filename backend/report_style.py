"""Consistent typography, page chrome and spacing for report exports."""
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def report_styles():
    styles = getSampleStyleSheet()
    for name in ('Normal', 'BodyText'):
        styles[name].fontName = 'Helvetica'
        styles[name].fontSize = 9.5
        styles[name].leading = 14
        styles[name].textColor = colors.HexColor('#4b4b61')
    styles['Title'].fontName = 'Helvetica-Bold'
    styles['Title'].fontSize = 23
    styles['Title'].leading = 28
    styles['Title'].alignment = 0
    styles['Title'].textColor = colors.HexColor('#252341')
    for name, size in (('Heading1', 16), ('Heading2', 12), ('Heading3', 10)):
        styles[name].fontName = 'Helvetica-Bold'
        styles[name].fontSize = size
        styles[name].leading = size + 5
        styles[name].textColor = colors.HexColor('#53428b')
        styles[name].spaceBefore = 12
        styles[name].spaceAfter = 6
        styles[name].keepWithNext = True
    styles.add(ParagraphStyle('Small', parent=styles['Normal'], fontSize=8, leading=11,
                              textColor=colors.HexColor('#858297')))
    styles.add(ParagraphStyle('CheckLink', parent=styles['Normal'], fontName='Helvetica-Bold',
                              fontSize=12, leading=17, textColor=colors.HexColor('#4a358c'),
                              backColor=colors.HexColor('#f0ecff'), borderColor=colors.HexColor('#d4c8f3'),
                              borderWidth=.7, borderPadding=8, spaceBefore=10, spaceAfter=10))
    styles.add(ParagraphStyle('CheckReason', parent=styles['Normal'], fontSize=10.5, leading=15,
                              textColor=colors.HexColor('#48445c'), spaceAfter=8))
    return styles


def report_page_decoration(report):
    identifier = str(report.get('id') or '')[:10]

    def decorate(canvas, document):
        width, height = document.pagesize
        canvas.saveState()
        canvas.setFillColor(colors.HexColor('#7054da'))
        canvas.roundRect(45, height - 41, width - 90, 18, 3, fill=1, stroke=0)
        canvas.setFillColor(colors.white)
        canvas.setFont('Helvetica-Bold', 8)
        canvas.drawString(54, height - 35, 'AI-POWERED NEWS VERIFICATION')
        canvas.setStrokeColor(colors.HexColor('#e6e1f0'))
        canvas.line(45, 38, width - 45, 38)
        canvas.setFillColor(colors.HexColor('#858297'))
        canvas.setFont('Helvetica', 7)
        canvas.drawString(45, 26, 'Private report | ' + identifier + ' | Evidence and model signals are separate')
        canvas.drawRightString(width - 45, 26, 'Page ' + str(document.page))
        canvas.restoreState()

    return decorate
