"""Separate, local AI-image scoring; never used as a factual-truth classifier."""
import json
import math
from pathlib import Path
import re
import threading

from PIL import ImageOps


from ml.origin_features import generation_views


def retire_obsolete_origin(result, active_directory):
    """Old uploads were deleted; retain the old diagnostic rather than invent a rescore."""
    assessment = result.get('ai_assessment') or {}
    if assessment.get('status') == 'requires_reanalysis':
        return False
    # This checkpoint repeatedly misclassified the supplied real photographs.
    if 'Ateeqq/ai-vs-human-image-detector'.casefold() not in str(assessment.get('source_url', '')).casefold():
        return False
    try:
        active = json.loads((Path(active_directory) / 'model-card.json').read_text(encoding='utf-8'))
    except (OSError, TypeError, ValueError):
        return False
    if active.get('prediction_enabled') is not True or assessment.get('model_version') == active.get('version'):
        return False
    result['retired_ai_assessment'] = {'assessment': assessment, 'ai_percentage': result.get('ai_percentage'),
                                     'confidence': result.get('confidence')}
    result['ai_assessment'] = {'status': 'requires_reanalysis', 'replaced_by': active.get('version')}
    for key in ('ai_percentage', 'fake_percentage', 'real_percentage', 'confidence'):
        result[key] = None
    for frame in result.get('frames') or []:
        frame.pop('ai_percentage', None)
    summary = result.get('report_summary')
    if isinstance(summary, dict):
        summary['model_signal'] = 'The previous image-origin estimate was retired. Upload this file again for a new prediction.'
    return True


def _origin_thresholds(assessment, visual_type):
    """Only a reviewed model card can authorize binary origin decisions.

    Thresholds are percentages selected on a separate validation collection,
    identified by its fingerprint. Prediction enablement permits inference;
    it does not establish validation, deployment readiness, or media scope.
    """
    if not (assessment.get('locally_validated') is True and assessment.get('deployment_ready') is True):
        return None
    policy = assessment.get('decision_policy')
    if not isinstance(policy, dict) or policy.get('validation_status') != 'validated':
        return None
    fingerprint = policy.get('validation_dataset_sha256')
    if not isinstance(fingerprint, str) or not re.fullmatch(r'[0-9a-fA-F]{64}', fingerprint):
        return None
    scopes = policy.get('visual_types')
    if not isinstance(scopes, list) or visual_type not in scopes:
        return None
    lower, upper = policy.get('real_max_percentage'), policy.get('ai_min_percentage')
    if any(not isinstance(value, (int, float)) or isinstance(value, bool)
           or not math.isfinite(value) for value in (lower, upper)):
        return None
    if not 0 <= lower < upper <= 100:
        return None
    return lower, upper


def media_origin_assessment(score, assessment, visual_type='unclear', credentials=None, *, kind='image', video_assessment=None):
    """An unvalidated classifier score cannot establish an image's origin."""
    assessed = (isinstance(score, (int, float)) and not isinstance(score, bool)
                and math.isfinite(score) and 0 <= score <= 100)
    thresholds = _origin_thresholds(assessment, visual_type)
    verdict = 'inconclusive' if assessed else 'classification_unavailable'
    reason = ('The detector has not been locally validated with reviewed decision thresholds for this media type. '
              'Its experimental score cannot establish whether this image is real or AI-generated.'
              if assessed else 'The AI-origin detector did not return a valid score.')
    if assessed and thresholds:
        lower, upper = thresholds
        verdict = 'likely_real' if score <= lower else 'ai_generated' if score >= upper else 'inconclusive'
        reason = ('The score falls within the validation-defined inconclusive range.' if verdict == 'inconclusive'
                  else 'The score meets the reviewed decision threshold for this media type. A model prediction is not proof of origin.')
    if assessment.get('status') == 'requires_reanalysis':
        reason = 'The image-origin detector has been replaced. Upload this file again to get a new prediction.'
    if visual_type == 'cartoon_or_illustration' and assessed:
        verdict = 'inconclusive'
        reason = ('This is a cartoon or illustration. The current detector has not been validated '
                  'to distinguish human-drawn from AI-generated artwork. Its raw class scores '
                  'cannot establish the origin of this illustration.')
    publish_scores = verdict in ('ai_generated', 'likely_real')
    if credentials and credentials.get('ai_declared') and credentials.get('binding_valid'):
        verified = credentials.get('status') == 'verified'
        return {'policy_version': 5, 'status': credentials['status'], 'basis': 'content_credentials',
                'verdict': 'ai_content_declared', 'score': None, 'scores_publishable': False,
                'visual_type': visual_type, 'reason': credentials['message'],
                'headline': 'AI-generated content declared' if not verified else 'AI-generated content — Content Credentials',
                'credential_scope': credentials.get('scope'), 'credential_verified': verified}
    if kind == 'video':
        if video_assessment and video_assessment.get('status') == 'experimental':
            signal = video_assessment.get('synthetic_or_manipulated_percentage')
            if isinstance(signal, (int, float)) and not isinstance(signal, bool) and 0 <= signal <= 100:
                return {'policy_version': 7, 'status': 'experimental_video_model',
                        'basis': 'video_consistency_features', 'verdict': 'inconclusive',
                        'headline': ('AI/manipulation-like video signals (experimental)' if signal >= 70 else
                                     'Real-like video signals (experimental)' if signal <= 30 else
                                     'Video model signals are mixed'),
                        'score': None, 'scores_publishable': False, 'visual_type': visual_type,
                        'reason': video_assessment['note']}
        # Image benchmarks and face-swap consistency do not validate synthetic
        # video detection. Preserve the frame scores only as diagnostics.
        return {'policy_version': 6, 'status': 'video_model_not_validated',
                'basis': 'sampled_image_frames', 'verdict': 'inconclusive',
                'headline': 'Video origin inconclusive', 'score': None,
                'scores_publishable': False, 'visual_type': visual_type,
                'reason': 'The installed detector scores still frames, not generated motion. '
                          'It has not been validated to distinguish AI-generated videos from real recordings. '
                          'A low frame score or face-swap score does not establish that this video is real.'}
    return {'policy_version': 8, 'status': assessment.get('status', 'not_assessed'),
            'verdict': verdict, 'score': score if publish_scores else None,
            'scores_publishable': publish_scores, 'visual_type': visual_type, 'reason': reason}


class GenerationDetector:
    def __init__(self, directory):
        self.directory = Path(directory) if directory else None
        self.lock = threading.Lock()
        self.model = None
        self.processor = None
        self.card = None
        self.ai_index = None
        self.classifier = None
        self.visual_origin = None

    def ready(self):
        if self.directory and all((self.directory / name).is_file() for name in ('model.joblib','backbone.pt','model-card.json')):
            try:
                card = json.loads((self.directory / 'model-card.json').read_text(encoding='utf-8'))
                return card.get('architecture') == 'cnn_origin' and card.get('purpose') == 'ai_generation'
            except (OSError, ValueError):
                return False
        if self.directory and (self.directory / 'classifier.joblib').is_file() and (self.directory / 'model-card.json').is_file():
            try:
                card = json.loads((self.directory / 'model-card.json').read_text(encoding='utf-8'))
                root = Path(__file__).resolve().parents[1]
                base = (root / card['base_model_path']).resolve()
                return (card.get('architecture') == 'frozen_vit_probe' and base.is_relative_to(root)
                        and all((base / name).is_file() for name in
                                ('config.json', 'preprocessor_config.json', 'model.safetensors')))
            except (OSError, ValueError, KeyError, TypeError):
                return False
        return bool(self.directory and all((self.directory / name).is_file() for name in (
            'config.json', 'preprocessor_config.json', 'model.safetensors', 'model-card.json')))

    def _load(self):
        if self.model is not None:
            return
        directory = str(self.directory)
        card = json.loads((self.directory / 'model-card.json').read_text(encoding='utf8'))
        if card.get('architecture') == 'cnn_origin':
            from ml.vision import VisualModel
            self.visual_origin = VisualModel(self.directory / 'model.joblib')
            self.model = self.visual_origin
            self.card = card
            return
        if card.get('architecture') == 'frozen_vit_probe':
            import joblib
            from ml.origin_features import OriginEncoder
            root = Path(__file__).resolve().parents[1]
            base = (root / card['base_model_path']).resolve()
            if not base.is_relative_to(root):
                raise ValueError('The origin backbone must be a trusted project artifact.')
            model = OriginEncoder(base)
            classifier = joblib.load(self.directory / 'classifier.joblib')
            self.model, self.classifier = model, classifier
            self.card = card
            return
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        processor = AutoImageProcessor.from_pretrained(directory, local_files_only=True, trust_remote_code=False, use_fast=False)
        model = AutoModelForImageClassification.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False, use_safetensors=True).eval()
        labels = {int(index): label for index, label in model.config.id2label.items()}
        matches = [index for index, label in labels.items() if label == card.get('ai_label')]
        if len(matches) != 1 or len(labels) != 2:
            raise ValueError('The AI-image model must have two classes and an explicit, matching ai_label.')
        self.processor, self.model = processor, model
        self.card, self.ai_index = card, matches[0]

    def predict(self, images, batch_size=2):
        """Return uncalibrated AI-class scores, in input order, in bounded batches."""
        with self.lock:
            self._load()
            scores = []
            if self.visual_origin is not None:
                for offset in range(0,len(images),batch_size):
                    batch, _ = self.visual_origin.predict(images[offset:offset+batch_size])
                    scores.extend(batch)
                return scores, dict(self.card)
            if self.classifier is not None:
                import numpy as np
                # Four originals yield twelve views: three full encoder batches.
                probe_batch_size = max(4, batch_size)
                for offset in range(0,len(images),probe_batch_size):
                    features = self.model.extract(images[offset:offset+probe_batch_size])
                    logits = self.classifier['classifier'].decision_function(features) / self.classifier['temperature']
                    scores.extend((1/(1+np.exp(-np.clip(logits,-30,30)))).tolist())
                return scores, dict(self.card)
            import torch
            for offset in range(0, len(images), batch_size):
                batch = [ImageOps.exif_transpose(image).convert('RGB') for image in images[offset:offset + batch_size]]
                inputs = self.processor(images=batch, return_tensors='pt')
                with torch.inference_mode():
                    probabilities = self.model(**inputs).logits.softmax(dim=-1)
                scores.extend(probabilities[:, self.ai_index].tolist())
            return scores, dict(self.card)
