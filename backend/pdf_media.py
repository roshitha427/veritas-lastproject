"""Shared image-preview rendering; video files/posters are never embedded."""
from io import BytesIO

from reportlab.platypus import Image, Paragraph, Spacer

from backend.report_media import preview_bytes


def image_preview_flowables(report, styles):
    if report.get('type') != 'image':
        return []
    raw = preview_bytes(report)
    if raw is None:
        return []
    image = Image(BytesIO(raw))
    scale = min(1, 324 / image.imageWidth, 252 / image.imageHeight)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    image.hAlign = 'LEFT'
    return [Paragraph('Uploaded image preview', styles['Heading3']), image, Spacer(1, 12)]
