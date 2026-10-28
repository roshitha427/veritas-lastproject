"""The three recorded classifier metrics displayed in reports and exports."""
import math


METRIC_LABELS = {
    'text': ('Real news pattern score', 'Fake news pattern score', 'Text model confidence'),
    'image': ('Real image pattern score', 'AI image pattern score', 'Image model confidence'),
    'video': ('Real video frame pattern score', 'AI video frame pattern score', 'Video frame model confidence'),
}


def _score(value):
    return value if (isinstance(value, (int, float)) and not isinstance(value, bool)
                     and math.isfinite(value) and 0 <= value <= 100) else None


def model_metrics(report):
    """Keep model scores separate from evidence, OCR and writing-origin scores."""
    result = report.get('result') or {}
    fake = result.get('fake_percentage')
    if fake is None and report.get('type') in ('image', 'video'):
        fake = result.get('ai_percentage')
    labels = METRIC_LABELS.get(report.get('type'), METRIC_LABELS['text'])
    return list(zip(labels, (_score(result.get('real_percentage')), _score(fake),
                             _score(result.get('confidence')))))


def format_percentage(value):
    value = _score(value)
    return 'Not assessed' if value is None else f'{value:.1f}%'
