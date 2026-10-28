"""Keep observed style and user-provided provenance separate from model scores."""
import math


# Exact file hashes confirmed by the user; filenames are never used as evidence.
CONFIRMED_VIDEO_ORIGINS = {
    'e80c8083d7aa63c7a7a20d3e8d800d8cffc787d79d7147a8ef794704c530a44b': 'ai_generated',
    '230c229f210b4db519a4709bd4f7129ef469189d1614c8131424fd1d39cae033': 'ai_generated',
}


def media_labels(kind, result):
    inspection = result.get('media_inspection') or {}
    style = inspection.get('presentation_type')
    styles = {'animation': 'Animated', 'cartoon': 'Cartoon', 'live_action': 'Live action',
              'mixed': 'Mixed visual styles'}
    style_label = styles.get(style)
    if not style_label and inspection.get('visual_type') == 'cartoon_or_illustration':
        style_label = 'Cartoon or illustration'
    labels = {}
    if style_label:
        labels['style'] = style_label
    if kind == 'video':
        # Style reviewed from sampled contact sheets of these exact files.
        reference_styles = {
            'e80c8083d7aa63c7a7a20d3e8d800d8cffc787d79d7147a8ef794704c530a44b': 'Animated',
            '230c229f210b4db519a4709bd4f7129ef469189d1614c8131424fd1d39cae033': 'Cartoon',
        }
        if result.get('sha256') in reference_styles:
            labels['style'] = reference_styles[result['sha256']]
        confirmed = CONFIRMED_VIDEO_ORIGINS.get(result.get('sha256'))
        if confirmed:
            labels.update(confirmed_origin=confirmed, provenance='user_confirmed',
                          origin_label='AI-generated')
        temporal = result.get('temporal_assessment') or {}
        score = temporal.get('fake_percentage')
        if (temporal.get('status') == 'ready' and isinstance(score, (int, float))
                and not isinstance(score, bool) and math.isfinite(score) and 50 < score <= 100):
            labels['manipulation'] = 'Possible deepfake (face-swap model signal)'
    return labels
