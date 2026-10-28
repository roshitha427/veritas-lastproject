import hashlib
import importlib.util
import json
import logging
import math
import re
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from backend.verification import verify_submission
from ml.vision import VisualModel
from ml.text_cleaning import clean_text, content_language
from backend.speech import SpeechTranscriber
from backend.generation import GenerationDetector, media_origin_assessment
from backend.report_summary import filter_relevant_evidence, summarize_report
from backend.local_evidence import evidence_ready
from backend.media_inspection import inspect_media, select_automatic_ocr_claim
from ml.temporal import TemporalModel
from backend.video_model import VideoModel
from ml.ai_text import AITextModel
from backend.evidence_nli import EvidenceVerifier
from backend.result_cache import ResultCache
from backend.evidence_metrics import evidence_metrics
from backend.source_recommendations import with_source_recommendations
from backend.verification_review import annotate_verification_review

Image.MAX_IMAGE_PIXELS = 30_000_000


def runtime_available(*modules):
    """Check optional inference packages without importing their heavy runtimes."""
    try:
        return all(importlib.util.find_spec(name) is not None for name in modules)
    except (ImportError, ValueError):
        return False


def _is_visual_caption_only(text):
    """Treat short labels such as 'Shinchan cartoon picture' as descriptions."""
    words = re.findall(r"[\w'-]+", (text or '').lower(), flags=re.UNICODE)
    visual_terms = {'picture', 'photo', 'image', 'cartoon', 'drawing', 'illustration',
                    'character', 'screenshot', 'poster', 'meme', 'scene', 'person',
                    'man', 'woman', 'child', 'dog', 'cat', 'landscape', 'building'}
    claim_terms = {'won', 'lost', 'died', 'killed', 'elected', 'arrested', 'announced',
                   'approved', 'banned', 'launched', 'caused', 'increased', 'decreased',
                   'confirmed', 'reported', 'claimed', 'reveals', 'proves'}
    has_number = any(char.isdigit() for char in text or '')
    visual_caption = len(words) <= 10 and bool(set(words) & visual_terms)
    return visual_caption and not bool(set(words) & claim_terms) and not has_number


def _usable_media_claim(text):
    words = re.findall(r"[\w'-]+", (text or ''), flags=re.UNICODE)
    return len((text or '').strip()) >= 10 and len(words) >= 2


def transcript_claim_text(text):
    """Keep speech visible without searching greetings and entertainment dialogue.

    This conservative routing rule is not a factual-truth classifier. Explicit
    user claims remain searchable independently of automatic speech extraction.
    """
    claims = []
    for sentence in re.split(r'(?<=[.!?\u0964])\s+|\n+', (text or '').strip()):
        if not _usable_media_claim(sentence):
            continue
        if re.search(r'\b(welcome(?:\s+back)?|thank\s+you|thanks|hello|hi\s+there|good\s+(?:morning|evening)|'
                     r'ladies\s+and\s+gentlemen|subscribe|follow\s+me)\b', sentence, re.I):
            continue
        if content_language(sentence) != 'en' or re.search(
                r'\b(is|are|was|were|has|have|had|will|won|lost|died|killed|elected|arrested|'
                r'announced|approved|banned|launched|caused|increased|decreased|confirmed|'
                r'reported|claimed|opened|closed|passed|signed|costs?|contains?|causes?|meets?|meet)\b',
                sentence, re.I):
            claims.append(sentence)
    return '\n'.join(claims)


def detected_face_counts(images):
    """Face presence gates a face-swap diagnostic; it never determines origin."""
    import cv2
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    counts = []
    for image in images:
        bounded = image.copy()
        bounded.thumbnail((640, 640))
        gray = cv2.cvtColor(np.asarray(bounded), cv2.COLOR_RGB2GRAY)
        counts.append(len(cascade.detectMultiScale(gray, 1.1, 5)))
    return counts


def media_claim_text(text):
    """Keep OCR content searchable while excluding camera attribution overlays."""
    lines = []
    for line in (text or '').splitlines():
        words = re.findall(r"[\w'-]+", line, flags=re.UNICODE)
        camera_mark = re.match(r'^\s*(?:redmi(?:\s+(?:on\b|note\b|\d)|\s*$)|shot\s+(?:on|by)\b|(?:taken|captured)\s+(?:on|by)\b|watermark\b)', line, re.I)
        if camera_mark and len(words) <= 12:
            continue
        lines.append(line.strip())
    cleaned = '\n'.join(line for line in lines if line)
    return cleaned if _usable_media_claim(cleaned) else ''


def text_input_metrics(text, source_count=0):
    return {'word_count': len(clean_text(text).split()), 'character_count': len(text),
            'sentence_count': len([part for part in re.split(r'[.!?\u0964]+(?:\s+|$)', text) if part.strip()]),
            'source_count': source_count}


class Analyzer:
    def __init__(self, config):
        self.config = config
        self.visual = VisualModel(config['IMAGE_MODEL_PATH'])
        self.text_pipe = None
        self.text_lock = threading.Lock()
        self.speech = SpeechTranscriber(config.get('WHISPER_MODEL'))
        self.generation = GenerationDetector(config.get('AI_IMAGE_MODEL_PATH'))
        self.target_accuracy = config.get('MODEL_TARGET_ACCURACY', .90)
        self.temporal = TemporalModel(config.get('TEMPORAL_MODEL_PATH'))
        self.video = VideoModel(config.get('VIDEO_MODEL_PATH'))
        self.ai_text = AITextModel(config['AI_TEXT_MODEL_PATH']) if config.get('AI_TEXT_MODEL_PATH') else None
        self.evidence_verifier = EvidenceVerifier(config.get('EVIDENCE_NLI_MODEL_PATH'))
        self.result_cache = ResultCache()

    def readiness(self):
        text_path = Path(self.config['TEXT_MODEL_PATH'])
        provider = self.config.get('EVIDENCE_PROVIDER', 'local')
        vision_runtime = runtime_available('torch', 'torchvision')
        text_runtime = runtime_available('torch', 'transformers')
        generation_runtime = vision_runtime and text_runtime
        generation_directory = getattr(self.generation, 'directory', None)
        if generation_directory:
            try:
                card = json.loads((Path(generation_directory) / 'model-card.json').read_text(encoding='utf8'))
                if card.get('architecture') == 'cnn_origin':
                    generation_runtime = vision_runtime
            except (OSError, ValueError):
                pass
        visual_ready = vision_runtime and self.visual.ready()
        temporal_ready = vision_runtime and self.temporal.ready()
        video_ready = vision_runtime and self.video.ready()
        return {'image': visual_ready, 'video': visual_ready or temporal_ready or video_ready,
                'video_synthetic_detection': video_ready,
                'temporal': temporal_ready,
                'text': text_runtime and (text_path / 'config.json').is_file() and (text_path / 'model-card.json').is_file()
                    and any((text_path / filename).is_file() for filename in ('model.safetensors', 'pytorch_model.bin')),
                'speech': text_runtime and self.speech.ready(),
                'ai_generation': generation_runtime and self.generation.ready(),
                'ai_writing': bool(self.ai_text and self.ai_text.ready),
                'gemini_report_summary': bool(self.config.get('GEMINI_API_KEY')),
                'tavily_search': bool(self.config.get('TAVILY_API_KEY')),
                'local_evidence_comparison': text_runtime and self.evidence_verifier.ready(),
                'supplied_article_search': Path(self.config.get('DATASET_ARTICLES_PATH') or 'missing-dataset').is_file(),
                'evidence_search': (provider == 'local' and evidence_ready(self.config.get('EVIDENCE_MODEL_PATH'))) or provider in ('public', 'rss', 'gdelt', 'auto') or (provider == 'tavily' and bool(self.config.get('TAVILY_API_KEY')))}

    def text(self, text):
        if not self.readiness()['text']:
            return None, {'name': 'Transformer text classifier', 'version': 'not installed', 'status': 'not_configured'}
        import json
        directory = Path(self.config['TEXT_MODEL_PATH'])
        card = json.loads((directory / 'model-card.json').read_text(encoding='utf8'))
        if content_language(text) not in card.get('languages', ['en']):
            return None, {'name': card.get('name', 'BERT'), 'version': card.get('version', '1'),
                          'status': 'unsupported_language', 'evaluation': card.get('evaluation')}
        word_count = len(clean_text(text).split())
        short_input = word_count < 20 or (text.strip().endswith('?') and word_count < 80)
        if card.get('fake_label') is None:
            raise ValueError('The text model must declare its fake_label in model-card.json.')
        with self.text_lock:
            if self.text_pipe is None:
                # Transformers resolves pipeline lazily. Concurrent first imports
                # can observe a partially initialized module, so serialize import
                # together with model initialization and inference.
                from transformers.pipelines import pipeline
                self.text_pipe = pipeline('text-classification', model=str(directory), tokenizer=str(directory), device=-1)
            scores = self.text_pipe(clean_text(text), truncation=True, max_length=card.get('max_length', 512), top_k=None)
        fake = next((x['score'] for x in scores if x['label'] == card['fake_label']), None)
        if fake is not None:
            odds = math.log(max(1e-7, fake) / max(1e-7, 1 - fake))
            fake = 1 / (1 + math.exp(-odds / card.get('temperature', 1.)))
        evaluation = card.get('evaluation')
        return fake, {'name': card.get('name', 'BERT text classifier'), 'version': card.get('version', '1'),
                      'status': 'ready', 'evaluation': evaluation,
                      'input_scope': 'short_text' if short_input else 'news_article',
                      'experimental_input': short_input,
                      'requested_target_accuracy': self.target_accuracy,
                      'requested_target_met': bool(evaluation and evaluation.get('accuracy', 0) >= self.target_accuracy),
                      'external_evaluations': card.get('external_evaluations', {})}

    def analyze(self, kind, text, filepath, language, sources, reviewed_ocr=None):
        """Reuse identical inputs briefly, keeping one evidence/model snapshot."""
        digest = None
        if filepath:
            with open(filepath, 'rb') as media:
                digest = hashlib.file_digest(media, 'sha256').hexdigest()
        # Settings, publisher permissions and artifact revisions invalidate reuse.
        revisions = []
        for name, value in sorted(self.config.items()):
            if name.endswith('_PATH') and value:
                path = Path(value)
                candidates = ([path / filename for filename in (
                    'model-card.json', 'model.joblib', 'model.safetensors', 'pytorch_model.bin',
                    'backbone.pt', 'classifier.joblib', 'config.json')] if path.is_dir() else
                    [path, path.with_name('model-card.json'), path.with_name('backbone.pt')])
                for candidate in candidates:
                    if candidate.is_file():
                        stat = candidate.stat()
                        revisions.append((str(candidate), stat.st_size, stat.st_mtime_ns))
        source_records = sorted(json.dumps(item, sort_keys=True, default=str) for item in sources)
        fingerprint = json.dumps([kind, text, digest, language, source_records, reviewed_ocr,
                                  dict(self.config), revisions,
                                  datetime.now(timezone.utc).date().isoformat()],
                                 sort_keys=True, default=str)
        key = hashlib.sha256(fingerprint.encode('utf-8')).hexdigest()

        def compute():
            result = (self._analyze(kind, text, filepath, language, sources, reviewed_ocr=reviewed_ocr)
                      if reviewed_ocr is not None else self._analyze(kind, text, filepath, language, sources))
            stamp = datetime.now(timezone.utc)
            result['analysis_snapshot'] = {
                'created_at': stamp.isoformat(),
                'expires_at': (stamp + timedelta(seconds=self.result_cache.ttl)).isoformat(),
                'reuse_window_seconds': self.result_cache.ttl,
            }
            return result

        return self.result_cache.get_or_compute(key, compute, self._reusable_result)

    @staticmethod
    def _reusable_result(result):
        """Do not preserve a transient outage or rejected external response."""
        summary = result.get('report_summary') or {}
        generation = (summary.get('gemini_analysis') or {}).get('status')
        if generation not in (None, 'completed', 'not_configured', 'not_called_no_evidence'):
            return False
        checks = [result.get(name) or {} for name in (
            'verification', 'media_inspection', 'speech', 'ai_assessment',
            'text_model_assessment', 'video_assessment', 'temporal_assessment')]
        failures = {'unavailable', 'timeout', 'network_error', 'quota_or_auth_unavailable',
                    'quota_or_rate_limit', 'server_error', 'authentication_error', 'permission_denied'}
        return not any(row.get('status') in failures or row.get('provider_status') in failures
                       or any(provider.get('status') in failures for provider in row.get('providers', []))
                       for row in checks)

    def _analyze(self, kind, text, filepath, language, sources, reviewed_ocr=None):
        supplied_claim = (text or '').strip()
        result = {'fake_percentage': None, 'real_percentage': None, 'ai_percentage': None, 'confidence': None,
                  'models': [], 'observations': [], 'limitations': [
                    'AI generation is separate from factual accuracy. A generated illustration does not establish whether the surrounding news claim is true.',
                    'Model confidence is not measured accuracy and does not verify a news claim.'],
                  'scope': 'No model prediction is available for this content.'}
        if filepath:
            with open(filepath, 'rb') as media:
                result['sha256'] = hashlib.file_digest(media, 'sha256').hexdigest()
        probability = None
        text_model_card = None
        if kind == 'text':
            self._assess_ai_writing(text, result)
            try:
                probability, card = self.text(text)
            except Exception:
                logging.getLogger(__name__).exception('Text classifier failed; continuing with evidence analysis.')
                probability, card = None, {'name': 'Transformer text classifier', 'version': 'unknown',
                                           'status': 'unavailable'}
            text_model_card = card
            result['models'].append(card)
            self._text_domain_warning(result, card)
            result['scope'] = 'Text fake/real classification based on the installed training corpus. Independent evidence is required.'
            if card.get('experimental_input'):
                result['scope'] = 'Experimental news-pattern scoring of a short claim or question; factual truth is assessed separately using sources.'
                result['metric_note'] = ('Short-text scores are experimental: this classifier was trained on news articles. '
                    'Model confidence is its strongest class score, not measured accuracy or evidence that the claim is true.')
                result['limitations'].append('Short claims and questions are outside the article benchmark. Their classifier scores do not verify current facts.')
            if card['status'] == 'unsupported_language':
                result['limitations'].append('This text model was evaluated on English only. It abstains for Hindi, Telugu, and other unsupported scripts.')
                result['metric_note'] = 'The text classifier was evaluated on English only, so it did not calculate fake/real scores or confidence for this language.'
            elif card['status'] == 'claim_out_of_domain':
                result['scope'] = 'The news-article classifier abstained on this short claim or question. Review the concise evidence assessment and source details in the downloaded PDF.'
                result['limitations'].append('The text classifier was trained on full news articles and performs poorly on short political claims. It abstains here; use the retrieved evidence details in the downloaded PDF to assess the claim.')
                metric_note = (
                    'The text classifier abstained because this is a short claim outside its full-article training domain. '
                    'Fake/real scores and confidence were not computed.'
                )
                ai_status = (result.get('ai_assessment') or {}).get('status')
                if ai_status == 'too_short':
                    metric_note += f" The separate AI-writing detector needs at least {result['ai_assessment'].get('min_words', 80)} English words."
                elif ai_status == 'unsupported_language':
                    metric_note += ' The separate AI-writing detector has not been evaluated for this language.'
                elif not self.ai_text or not self.ai_text.ready:
                    metric_note += ' The separate AI-writing detector is not configured.'
                result['metric_note'] = metric_note
            elif probability is None:
                result['limitations'].append('Text model not trained: supply a labeled text dataset to enable transformer classification.')
                result['metric_note'] = 'The text classifier is not configured; fake/real scores and confidence were not computed.'
        else:
            images = []
            timestamps = []
            from backend.content_credentials import inspect_content_credentials
            result['content_credentials'] = inspect_content_credentials(filepath)
            if kind == 'image':
                with Image.open(filepath) as image:
                    image.load()
                    img = ImageOps.exif_transpose(image).convert('RGB')
                    result['observations'].append({'label': 'Dimensions', 'value': f'{img.width} × {img.height} px'})
                    images = [img.copy()]
                    result['media_details'] = {'width': img.width, 'height': img.height,
                                              'format': image.format, 'file_bytes': Path(filepath).stat().st_size}
            else:
                import cv2
                cap = cv2.VideoCapture(str(filepath))
                try:
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    if not cap.isOpened() or not math.isfinite(fps) or fps <= 0 or frames <= 0:
                        raise ValueError('This video could not be decoded. Use an MP4 or WebM file.')
                    duration = frames / fps
                    if not math.isfinite(duration) or duration > 120:
                        raise ValueError('Please use a video of two minutes or less.')
                    samples = (min(12, max(1, int(frames))) if self.temporal.ready() or self.video.ready()
                               else min(12, max(1, int(duration))))
                    for index in np.rint(np.linspace(0, frames-1, samples)).astype(int):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                        ok, frame = cap.read()
                        if ok:
                            if frame.shape[0] * frame.shape[1] > 30_000_000:
                                raise ValueError('Video frame dimensions exceed the supported limit.')
                            images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                            timestamps.append(round(float(index / fps), 2))
                    if not images:
                        raise ValueError('No readable frames were found in the video.')
                    result['observations'].extend([{'label': 'Duration', 'value': f'{duration:.1f} seconds'},
                                                   {'label': 'Sampled frames', 'value': str(len(images))}])
                    result['media_details'] = {'width': images[0].width, 'height': images[0].height,
                                              'duration_seconds': round(duration, 2), 'sampled_frames': len(images),
                                              'file_bytes': Path(filepath).stat().st_size}
                finally:
                    cap.release()
                if not self.temporal.ready():
                    result['limitations'].append('The AI-origin estimate uses sampled video frames; motion and audio authenticity are not assessed by that detector.')
            frame_probabilities, frame_model_version = None, None
            # A short-lived signed preview is bound to the authenticated user
            # and image bytes by the API. Reuse its exact OCR instead of asking
            # vision again and silently changing the text the user just reviewed.
            media_inspection = (dict(reviewed_ocr['inspection'])
                if kind == 'image' and reviewed_ocr and isinstance(reviewed_ocr.get('inspection'), dict)
                else inspect_media(images, self.config.get('GEMINI_API_KEY'),
                                   self.config.get('GEMINI_MODEL', 'gemini-3.5-flash-lite')))
            result['media_inspection'] = media_inspection
            face_counts = detected_face_counts(images)
            result['observations'].append({'label': 'Faces detected (maximum per frame)', 'value': str(max(face_counts, default=0))})
            try:
                if self.visual.ready():
                    probabilities, card = self.visual.predict(images)
                    frame_probabilities, frame_model_version = probabilities, card.get('version')
                    probability = float(np.mean(probabilities))
                    result['visual_model_assessment'] = {
                        'status': 'ready', 'fake_percentage': round(probability * 100, 1),
                        'real_percentage': round((1 - probability) * 100, 1),
                        'scope': 'Dataset-specific manipulation signal, separate from AI origin and factual truth.',
                    }
                    result['models'].append(dict(card, status='ready'))
                    if card.get('evaluation'):
                        result['models'][-1].update(requested_target_accuracy=self.target_accuracy,
                            requested_target_met=card['evaluation']['accuracy'] >= self.target_accuracy)
                    result['limitations'].extend(card.get('limitations', []))
                    result['limitations'].append('The visual classifier was trained on face-image crops. Performance on general news photos or non-face video frames is not established.')
                    if card.get('evaluation', {}).get('accuracy', 0) < self.target_accuracy:
                        result['limitations'].append(f'The visual classifier did not meet the requested {self.target_accuracy * 100:.0f}% held-out accuracy target. Treat its scores as experimental.')
                    if kind == 'video':
                        result['frames'] = [{'seconds': sec, 'visual_fake_percentage': round(score * 100, 1)}
                                            for sec, score in zip(timestamps, probabilities)]
                    result['scope'] = card.get('scope', 'Dataset-specific fake/real visual classification. Does not determine whether the surrounding news is true.')
                else:
                    result['models'].append({'name': 'Shared visual classifier', 'version': 'not trained', 'status': 'not_configured'})
                    result['limitations'].append('No trained visual artifact is installed. Media was inspected but authenticity was not assessed.')
                    if kind != 'video' or not self.temporal.ready():
                        result['metric_note'] = 'The visual classifier is not configured, so fake/real scores and model confidence were not computed.'
            except (OSError, ValueError, RuntimeError, ImportError):
                logging.getLogger(__name__).exception('Visual classifier failed; retaining OCR and AI-origin analysis.')
                result['visual_model_assessment'] = {'status': 'unavailable'}
                result['limitations'].append('The face-manipulation classifier could not run; other media analysis was retained.')
            if kind == 'video' and self.temporal.ready():
                if not any(face_counts):
                    result['temporal_assessment'] = {'status': 'out_of_scope',
                        'reason': 'No human face was detected in the sampled frames. The installed face-swap model '
                                  'cannot assess this clip or determine whether it is AI-generated.'}
                elif len(images) < 3:
                    result['limitations'].append('Too few readable frames for the learned temporal model. The video score uses available visual frames only.')
                else:
                    try:
                        probability, temporal_card = self.temporal.predict(images, frame_probabilities, frame_model_version)
                        temporal_card.update(status='ready', requested_target_accuracy=self.target_accuracy,
                            requested_target_met=temporal_card['evaluation']['accuracy'] >= self.target_accuracy)
                        result['models'].append(temporal_card)
                        result['temporal_assessment'] = {
                            'status': 'ready', 'sampled_frames': len(images), 'primary_score': False,
                            'fake_percentage': round(probability * 100, 1),
                            'real_percentage': round((1 - probability) * 100, 1),
                            'scope': 'Temporal manipulation signal, separate from the AI-origin estimate.',
                        }
                        result['scope'] = temporal_card['scope']
                        result['limitations'].extend(temporal_card.get('limitations', []))
                        result['limitations'].append('A separate learned consistency model also checked sampled frames. The AI-origin estimate remains separate; audio authenticity is not assessed.')
                    except (OSError, ValueError, RuntimeError, ImportError):
                        result['temporal_assessment'] = {'status': 'unavailable'}
                        result['limitations'].append('The learned video model could not run. Available frame-based analysis is retained.')
            visual_type = media_inspection.get('visual_type')
            if kind == 'video' and self.video.ready():
                try:
                    result['video_assessment'] = self.video.assess(images, result.get('sha256'))
                except (OSError, ValueError, RuntimeError, ImportError):
                    logging.getLogger(__name__).exception('Video-level model failed; retaining other analysis.')
                    result['video_assessment'] = {'status': 'unavailable'}
            # Run the AI-origin detector on every visual type. The prior skip for
            # cartoons and screenshots left those uploads without the requested
            # simple result; the report still marks these scores experimental.
            if self.generation.ready():
                try:
                    ai_scores, ai_card = self.generation.predict(images)
                    if (len(ai_scores) != len(images) or any(
                            isinstance(score, (bool, np.bool_)) or not np.isfinite(float(score))
                            or not 0 <= float(score) <= 1 for score in ai_scores)):
                        raise ValueError('The image-origin model returned invalid class scores.')
                    result['ai_percentage'] = round(float(np.mean(ai_scores)) * 100, 1)
                    result['ai_assessment'] = {'status': 'experimental', 'calibrated': bool(ai_card.get('validation_calibration')),
                        'locally_validated': ai_card.get('locally_validated') is True,
                        'scope': 'sampled_video_frames' if kind == 'video' else 'image',
                        'sample_count': len(ai_scores), 'source_url': ai_card['source_url']}
                    if ai_card.get('prediction_enabled'):
                        result['ai_assessment'].update(prediction_enabled=True, model_version=ai_card.get('version'))
                    for policy_key in ('deployment_ready', 'decision_policy'):
                        if policy_key in ai_card:
                            result['ai_assessment'][policy_key] = ai_card[policy_key]
                    result['models'].append(dict(ai_card, status='ready'))
                    result['limitations'].extend(ai_card.get('limitations', [])[:2])
                    if kind == 'video':
                        result['limitations'].extend(ai_card.get('limitations', [])[2:])
                    if kind == 'video':
                        frame_results = result.get('frames') or [{'seconds': sec} for sec in timestamps]
                        for frame_result, score in zip(frame_results, ai_scores):
                            frame_result['ai_percentage'] = round(score * 100, 1)
                        result['frames'] = frame_results
                    if visual_type in ('cartoon_or_illustration', 'screenshot_or_document'):
                        result['limitations'].append(
                            f'The AI-origin estimate for {visual_type.replace("_", " ")} is experimental and has not been independently validated for this media type.')
                except (OSError, RuntimeError, ValueError, TypeError, ImportError):
                    result['ai_assessment'] = {'status': 'unavailable'}
                    result['limitations'].append('The AI-image detector could not run. Its percentage was left unassessed; the other analysis results are still available.')
            else:
                result['ai_assessment'] = {'status': 'not_configured', 'calibrated': False,
                    'scope': 'sampled_video_frames' if kind == 'video' else 'image'}
                result['limitations'].append('No dedicated AI-image detector is installed. The AI-generated percentage is not assessed for this media.')
            if kind == 'video' and self.readiness()['speech']:
                try:
                    speech = self.speech.transcribe(filepath)
                    transcript = speech.pop('text')
                    result['speech'] = speech
                    result['models'].append({key: speech[key] for key in ('name', 'version', 'status')})
                    if transcript.strip():
                        result['transcript'] = transcript
                        text = '\n'.join([text, transcript]).strip()
                        speech_claim = transcript_claim_text(transcript)
                        if speech_claim:
                            speech_probability, speech_card = self.text(speech_claim)
                            self._text_domain_warning(result, speech_card)
                            result['speech_text_assessment'] = {'fake_percentage': None if speech_probability is None else round(speech_probability * 100, 1), 'model': speech_card}
                        result['limitations'].append('Speech is an automatic transcript that may contain errors. Its text assessment is separate from frame scores; review the words against the recording.')
                    elif speech['status'] not in ('no_audio', 'silence', 'no_speech_detected'):
                        result['limitations'].append('Video audio could not be transcribed; visual analysis and any supplied claim were still assessed.')
                except (OSError, RuntimeError, ValueError):
                    result['speech'] = {'name': 'Whisper tiny speech transcription', 'status': 'unavailable'}
                    result['limitations'].append('Speech transcription was configured but could not run. Check local Whisper files and FFmpeg installation.')
        if kind == 'text' and probability is not None:
            result.update(fake_percentage=round(probability * 100, 1), real_percentage=round((1 - probability) * 100, 1),
                          confidence=round(max(probability, 1 - probability) * 100, 1))
        if kind == 'text':
            result['text_model_assessment'] = self._text_model_assessment(probability, text_model_card or {})
        else:
            # Media confidence is the strongest class probability after averaging
            # frame scores. Calibration is dataset-specific and separate from factual truth.
            ai_score = result.get('ai_percentage')
            result['fake_percentage'] = ai_score
            result['real_percentage'] = None if ai_score is None else round(100 - ai_score, 1)
            result['confidence'] = None if ai_score is None else round(max(ai_score, 100 - ai_score), 1)
            result['confidence_note'] = ('Model confidence is the strongest AI-origin class score. '
                + ('It was calibrated on a bounded validation set and is not measured accuracy or factual certainty.'
                   if (result.get('ai_assessment') or {}).get('calibrated') else
                   'It is uncalibrated and is not measured accuracy or factual certainty.')
                + (' Video confidence uses the average sampled-frame prediction.' if kind == 'video' else ''))
            result['metric_note'] = ('The model estimates AI generation from visual patterns. This score is experimental, not calibrated accuracy or proof of origin.'
                                     if ai_score is not None else
                                     'The AI-origin model could not assess this upload, so no real/AI estimate is shown.')

            inspection = result.get('media_inspection') or {}
            original_text = str(inspection.get('visible_text') or '')
            extracted_text = original_text
            if kind == 'image' and reviewed_ocr is not None:
                extracted_text = reviewed_ocr['text']
                result['ocr_review'] = {'user_reviewed': True, 'original_text': original_text,
                    'corrected_text': extracted_text, 'changed': extracted_text != original_text}
                inspection.update(user_reviewed=True, corrected_text=extracted_text)
                if inspection.get('ocr_uncertain'):
                    result['limitations'].append('The automatic OCR contained uncertain text. The user-reviewed wording was used for source verification; it is not independent evidence.')
            elif inspection.get('ocr_uncertain'):
                result['limitations'].append('Some image text was not read reliably. The original transcription is preserved; uncertainty may limit claim extraction and source matching.')
            result['ocr_text'] = extracted_text
            claim_parts = []
            if supplied_claim and not _is_visual_caption_only(supplied_claim) and _usable_media_claim(supplied_claim):
                claim_parts.append(supplied_claim)
            # Keep the exact OCR visible in the report, but only search text
            # long enough to carry a useful proposition or query.
            selected_ocr = extracted_text
            if reviewed_ocr is None:
                # Headlines and factual assertions are selected separately from
                # publisher logos, banners and footers, while raw OCR stays visible.
                selected_ocr = select_automatic_ocr_claim(inspection)
            ocr_claim = media_claim_text(selected_ocr)
            if inspection.get('text_role') == 'watermark_or_label' and reviewed_ocr is None:
                ocr_claim = ''
            if ocr_claim:
                claim_parts.append(ocr_claim)
            transcript = str(result.get('transcript') or '').strip()
            speech_claim = transcript_claim_text(transcript)
            if speech_claim:
                claim_parts.append(speech_claim)
            claim = '\n'.join(dict.fromkeys(part for part in claim_parts if part))[:12000]

            visual_type = inspection.get('visual_type', 'unclear')
            result['media_origin'] = media_origin_assessment(ai_score, result.get('ai_assessment') or {}, visual_type,
                                                            result.get('content_credentials'), kind=kind,
                                                            video_assessment=result.get('video_assessment'))
            from backend.media_labels import media_labels
            result['media_labels'] = media_labels(kind, result)
            media_verdict = result['media_origin']['verdict']
            result['media_content_mode'] = ('claim_review' if claim else
                                            'text_detected' if extracted_text or transcript else 'visual_only')
            result['claim_text'] = claim
            if claim:
                # OCR and transcripts use the same writing models as text submissions.
                try:
                    text_probability, text_card = self.text(claim)
                except Exception:
                    logging.getLogger(__name__).exception('Media text classifier failed; retaining visual analysis.')
                    text_probability, text_card = None, {'status': 'unavailable', 'name': 'News text classifier'}
                writing = {'limitations': [], 'models': []}
                self._assess_ai_writing(claim, writing)
                result['media_text_analysis'] = {
                    'content': claim, 'text_model_assessment': self._text_model_assessment(text_probability, text_card),
                    'ai_percentage': writing.get('ai_percentage'), 'ai_assessment': writing.get('ai_assessment', {}),
                    'text_metrics': text_input_metrics(claim), 'limitations': writing['limitations']}
                result['models'].append(dict(text_card, purpose='media_text_classification'))
                result['models'].extend(writing['models'])
            if not claim:
                description = str(inspection.get('description') or '').strip()
                visual_artifacts = str(inspection.get('visual_artifacts') or '').strip()
                if media_verdict == 'ai_content_declared':
                    origin_label = result['media_origin']['reason']
                elif media_verdict == 'inconclusive':
                    origin_label = f'{kind.capitalize()} origin is inconclusive. ' + result['media_origin']['reason']
                elif ai_score is None:
                    origin_label = f'{kind.capitalize()} origin could not be classified.'
                elif ai_score >= 50:
                    origin_label = 'The model classifies this as likely AI-generated.'
                else:
                    origin_label = 'The model classifies this as a likely real image.'
                summary_parts = [origin_label]
                if description:
                    summary_parts.append(description)
                if visual_artifacts:
                    summary_parts.append('Visible details to review: ' + visual_artifacts)
                if extracted_text:
                    summary_parts.append('OCR text: ' + extracted_text[:1000])
                if transcript:
                    summary_parts.append('Speech transcript: ' + transcript[:1000])
                if (extracted_text or transcript) and not claim:
                    summary_parts.append('Extracted text is shown separately; no substantive factual claim was selected for a source search.')
                result['verification'] = {'status': 'not_applicable', 'provider_status': 'not_run', 'evidence': []}
                result['report_summary'] = {
                    'schema_version': 5, 'mode': 'media_only',
                    'headline': (result['media_origin'].get('headline') or ('Origin inconclusive' if media_verdict == 'inconclusive' else
                                 'Likely AI-generated' if ai_score is not None and ai_score >= 50 else
                                 'Likely real image' if ai_score is not None else 'Origin classification unavailable')),
                    'summary': ' '.join(summary_parts),
                    'reason': result['metric_note'],
                }

        claim = result.get('claim_text') if kind in ('image', 'video') else text
        if kind == 'text' or claim:
            detected = content_language(claim)
            try:
                verification = verify_submission(claim, detected if detected in ('en', 'hi', 'te') else language,
                                  sources,
                                  provider=self.config.get('EVIDENCE_PROVIDER', 'local'),
                                  evidence_path=self.config.get('EVIDENCE_MODEL_PATH'),
                                  dataset_path=self.config.get('DATASET_ARTICLES_PATH'),
                                  gemini_key=self.config.get('GEMINI_API_KEY'),
                                  gemini_model=self.config.get('GEMINI_MODEL', 'gemini-3.5-flash-lite'),
                                  grounded_search=self.config.get('GEMINI_GROUNDED_SEARCH', False),
                                  tavily_key=self.config.get('TAVILY_API_KEY'))
            except Exception:
                logging.getLogger(__name__).exception('Evidence retrieval failed; retaining model analysis.')
                verification = {'status': 'unverified', 'provider_status': 'unavailable', 'evidence': [],
                                'explanation': 'Source retrieval failed. Model scores remain available; the claim is unverified.'}
            checked_at = verification.get('checked_at')
            for item in verification.get('evidence') or []:
                if not isinstance(item, dict):
                    continue
                item.setdefault('retrieved_at', checked_at)
                item.setdefault('source_type', item.get('kind', 'source'))
            verification = filter_relevant_evidence(claim, verification)
            result['verification'] = verification
            local_assessment = self.evidence_verifier.assess(claim, verification)
            result['evidence_assessment'] = local_assessment
            result['report_summary'] = summarize_report(
                claim, kind, result, verification,
                api_key=self.config.get('GEMINI_API_KEY'),
                model=self.config.get('GEMINI_MODEL', 'gemini-3.5-flash-lite'),
                local_assessment=local_assessment,
            )
            annotate_verification_review(result, claim)
            result['report_summary'] = with_source_recommendations(result['report_summary'], claim, sources)
        if kind == 'text':
            result['text_metrics'] = text_input_metrics(text, len((result.get('verification') or {}).get('evidence') or []))
            result['evidence_metrics'] = evidence_metrics(result.get('report_summary'))
        elif result.get('media_text_analysis'):
            result['media_text_analysis']['text_metrics']['source_count'] = len((result.get('verification') or {}).get('evidence') or [])
        return result

    @staticmethod
    def _text_domain_warning(result, card):
        diagnostic = card.get('external_evaluations', {}).get('liar_strict_binary')
        if diagnostic and card.get('status') == 'ready':
            result['text_domain_warning'] = {'benchmark': 'LIAR short political claims',
                'accuracy': diagnostic['accuracy'], 'test_count': diagnostic['test_count'],
                'article_accuracy': (card.get('evaluation') or {}).get('accuracy'),
                'source_url': diagnostic['source_url']}
            note = (f"External claim benchmark: this article-trained model scored {diagnostic['accuracy'] * 100:.1f}% on "
                    f"{diagnostic['test_count']} unambiguous LIAR political claims. Its historical article accuracy does not establish claim verification accuracy.")
            if note not in result['limitations']:
                result['limitations'].append(note)

    @staticmethod
    def _text_model_assessment(probability, card):
        status = card.get('status', 'not_configured')
        model = card.get('name', 'Transformer text classifier')
        if probability is None:
            reasons = {
                'not_configured': 'No trained text classifier is installed. Gemini evidence comparison can still assess the claim when its API is configured.',
                'claim_out_of_domain': 'The classifier abstained because this short claim or question is outside its full-article training domain.',
                'unsupported_language': 'The classifier abstained because this language is outside its evaluated languages.',
                'unavailable': 'The text classifier could not complete this run. The Gemini evidence assessment is shown separately.',
            }
            return {'status': status, 'model': model, 'version': card.get('version'),
                    'verdict': 'not_assessed', 'reason': reasons.get(status,
                        'The text classifier did not produce a prediction for this input.'),
                    'evaluation': card.get('evaluation')}

        fake = round(probability * 100, 1)
        real = round((1 - probability) * 100, 1)
        confidence = round(max(probability, 1 - probability) * 100, 1)
        return {
            'status': 'assessed', 'model': model, 'version': card.get('version'),
            'verdict': 'fake' if probability >= 0.5 else 'authentic',
            'fake_percentage': fake, 'real_percentage': real, 'confidence': confidence,
            'experimental_input': bool(card.get('experimental_input')),
            'reason': (f'The news-pattern score is {fake:.1f}% with {confidence:.1f}% model confidence. '
                       'This is a learned text-pattern signal, not a fact-check of the claim.'
                       + (' This short input is outside the article benchmark; its score is experimental.'
                          if card.get('experimental_input') else '')),
            'evaluation': card.get('evaluation'),
        }

    def _assess_ai_writing(self, text, result):
        if not self.ai_text or not self.ai_text.ready:
            result['limitations'].append('No validated AI-writing detector is installed. The AI-generated percentage is not assessed for text.')
            return
        try:
            probability, card = self.ai_text.predict(text)
            if probability is None:
                result['ai_assessment'] = {'status': card['status'], 'scope': 'text', 'min_words': card.get('min_words',80),
                    'observed_words': card.get('observed_words'), 'max_words': card.get('max_words',800)}
                if card['status'] == 'too_short':
                    result['limitations'].append(f"The AI-writing model requires at least {card.get('min_words',80)} English words. Short-text authorship is not assessed.")
                elif card['status'] == 'unsupported_language':
                    result['limitations'].append('The AI-writing model has not been evaluated for this language. Authorship is not assessed.')
                return
            result['ai_percentage'] = round(probability*100,1)
            short_input = bool(card.get('experimental_input'))
            result['ai_assessment'] = {'status': 'experimental', 'scope': 'text', 'calibrated': not short_input,
                'experimental_input': short_input,
                'min_words': card.get('min_words',80), 'max_words': card.get('max_words',800),
                'observed_words': card.get('observed_words'),
                'sample_count': 1, 'source_url': card.get('dataset',{}).get('source_url', 'https://huggingface.co/datasets/Hello-SimpleAI/HC3'),
                'benchmark_accuracy': card.get('evaluation',{}).get('accuracy'),
                'domain_accuracy': card.get('domain_evaluation',{}).get('wiki_domain_holdout',{}).get('accuracy')}
            result['models'].append(dict(card, status='ready', purpose='ai_generation', score_scope='text',
                source_url=result['ai_assessment']['source_url']))
            result['limitations'].extend(card.get('limitations', []))
            if short_input:
                result['limitations'].append('AI-writing estimates for short text are experimental and have not been validated below the model\'s article-length benchmark.')
        except (OSError, ValueError, RuntimeError, ImportError, KeyError):
            result['ai_assessment'] = {'status': 'unavailable', 'scope': 'text'}
            result['limitations'].append('The AI-writing detector could not run. Its percentage was left unassessed; news classification and evidence retrieval are separate.')
