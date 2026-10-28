"""Local, experimental AI-writing inference, separate from news truth."""
import html
from pathlib import Path
import re
import threading

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from ml.text_cleaning import content_language


def clean_ai_text(text, max_words=800):
    """Normalize tokenization artifacts identically during training/inference."""
    text = html.unescape(str(text))
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'https?://\S+|\bURL_\d+\b', ' ', text, flags=re.I)
    text = text.replace('\u2019', "'").replace('\u2018', "'")
    text = re.sub(r"\b(\w+)\s+n\s*'\s*t\b", r"\1n't", text, flags=re.I)
    text = re.sub(r"\b(\w+)\s+'\s*(s|m|re|ve|ll|d)\b", r"\1'\2", text, flags=re.I)
    text = re.sub(r'\s+([,.;:!?%])', r'\1', text)
    text = re.sub(r'([([{])\s+', r'\1', text)
    text = re.sub(r'\s+([)\]}])', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return ' '.join(text.split()[:max_words])


def word_count(text):
    return len(re.findall(r"\b[A-Za-z][A-Za-z'-]*\b", text))


class AITextModel:
    def __init__(self, path=None):
        self.path = Path(path or 'models/ai-text')
        if self.path.suffix != '.joblib':
            self.path = self.path / 'model.joblib'
        self._artifact = None
        self._load_lock = threading.Lock()

    @property
    def ready(self):
        return self.path.is_file()

    def predict(self, text):
        if not self.ready:
            return None, {'status': 'unavailable', 'languages': ['en'], 'min_words': 80}
        if self._artifact is None:
            # Only load project-owned training artifacts, never user uploads.
            with self._load_lock:
                if self._artifact is None:
                    self._artifact = joblib.load(self.path)
        artifact = self._artifact
        card = dict(artifact['card'])
        card['model_status'] = card.get('status', 'experimental')
        if content_language(str(text)) != 'en':
            card['status'] = 'unsupported_language'
            return None, card
        cleaned = clean_ai_text(text, card.get('max_words', 800))
        count = word_count(cleaned)
        card['observed_words'] = count
        if count == 0:
            card['status'] = 'too_short'
            return None, card
        card['experimental_input'] = count < card.get('min_words', 80)
        with threadpool_limits(limits=3):
            features = artifact['vectorizer'].transform([cleaned])
            logit = float(artifact['classifier'].decision_function(features)[0])
        probability = float(1 / (1 + np.exp(-np.clip(logit / artifact['temperature'], -30, 30))))
        card['status'] = 'ready'
        return probability, card
