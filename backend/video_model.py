"""Video-specific experimental signal, distinct from image and face-swap scores."""
import json
import math
from pathlib import Path

from ml.temporal import TemporalModel

ROOT = Path(__file__).resolve().parents[1]


def video_detection_status(config, detector):
    """Explain installed research candidates without enabling or loading them."""
    candidates = []
    selected = config.get('VIDEO_MODEL_PATH')
    paths = list((ROOT / 'models').glob('video-*/model-card.json'))
    if selected:
        paths.append(Path(selected).with_name('model-card.json'))
    reasons = {
        'not_activated_failed_generalization': 'Held-out generalization checks failed; fitting supplied training clips did not establish reliable detection.',
        'not_activated_no_heldout_improvement': 'The candidate did not improve held-out performance and remains inactive.',
    }
    seen = set()
    for path in sorted(paths):
        if path.resolve() in seen:
            continue
        seen.add(path.resolve())
        try:
            card = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if card.get('purpose') != 'video_synthetic_or_manipulated':
            continue
        candidates.append({'name': card.get('name', 'Video research candidate'), 'version': card.get('version'),
            'prediction_enabled': card.get('prediction_enabled') is True,
            'locally_validated': card.get('locally_validated') is True,
            'deployment_ready': card.get('deployment_ready') is True,
            'reason': reasons.get(card.get('activation_decision'), 'No approved independent deployment validation is recorded.'),
            'dataset_diagnostics': {name: {'accuracy': value.get('accuracy'), 'test_count': value.get('test_count')}
                for name, value in (card.get('evaluation_by_dataset') or {}).items()
                if isinstance(value, dict) and isinstance(value.get('accuracy'), (int, float))}})
    enabled = bool(detector.ready())
    return {'configured': bool(selected), 'inference_enabled': enabled,
            'status': 'experimental_signal' if enabled else 'validation_required' if candidates else 'not_configured',
            'candidates': candidates,
            'reason': ('An optional video signal is enabled; it is separate from factual verification and does not prove full AI generation.' if enabled else
                       'No qualified synthetic-video detector is enabled. Review the installed candidate checks; frame-consistency analysis is a separate module.')}


class VideoModel(TemporalModel):
    def ready(self):
        if not super().ready():
            return False
        try:
            card = json.loads(self.path.with_name('model-card.json').read_text(encoding='utf-8'))
            return (card.get('purpose') == 'video_synthetic_or_manipulated'
                    and card.get('prediction_enabled') is True)
        except (OSError, ValueError):
            return False

    def assess(self, images, sha256=None):
        if not self.ready():
            return {'status': 'not_configured'}
        if len(images) < 3:
            return {'status': 'insufficient_frames', 'reason': 'At least three readable frames are required.'}
        probability, card = self.predict(images)
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError('Invalid video prediction')
        training_example = any(row.get('sha256') == sha256 for row in card.get('personal_training_fit', []))
        return {'status': 'experimental', 'model_version': card['version'],
                'synthetic_or_manipulated_percentage': round(probability * 100, 1),
                'real_class_percentage': round((1 - probability) * 100, 1),
                'sampled_frames': len(images), 'training_example': training_example,
                'locally_validated': card.get('locally_validated') is True,
                'scope': card['scope'],
                'note': 'Experimental video-level signal, separate from still-image scores. '
                        'It cannot distinguish full AI generation from face swaps or prove origin. '
                        + ('This clip was used for training; this result measures training fit.' if training_example else
                           'Modern generated-video accuracy has not been independently established.')}
