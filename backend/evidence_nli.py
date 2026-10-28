"""Local passage/claim comparison with source, freshness and abstention checks."""
from datetime import datetime, timezone
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import threading

from backend.evidence_assessment import assess_submission, claim_tokens
from ml.text_cleaning import content_language
from backend.claim_context import source_can_settle_current_claim


SETTLEMENT_POLICY_VERSION = 1
SETTLEMENT_SCOPE = 'english_claim_passage_comparison'


class EvidenceVerifier:
    def __init__(self, directory, threshold=.92):
        self.directory = Path(directory) if directory else None
        self.threshold = threshold
        self.lock = threading.Lock()
        self.tokenizer = self.model = None
        self._qualification_cache = None

    def ready(self):
        return bool(self.directory and all((self.directory / name).is_file()
                    for name in ('config.json', 'model.safetensors', 'tokenizer_config.json')))

    def settlement_qualification(self):
        """Installation readiness does not authorize factual verdict settlement.

        Future releases require a locally reviewed model-card contract:
        status=validated, locally_validated/deployment_ready=true, and an approved
        runtime_validation record binding this scope, threshold, model version,
        model/tokenizer SHA-256s and a held-out validation report. Threshold
        selection must use calibration data. Merely setting prediction_enabled,
        adding a benchmark, or obtaining a large NLI softmax score is insufficient.
        """
        result = {'policy_version': SETTLEMENT_POLICY_VERSION, 'qualified': False,
                  'status': 'experimental_unvalidated', 'scope': SETTLEMENT_SCOPE,
                  'threshold': self.threshold}

        def refuse(reason):
            return dict(result, reason=reason)

        if not self.directory:
            return refuse('No local evidence-model validation metadata is installed.')
        try:
            card_bytes = (self.directory / 'model-card.json').read_bytes()
            card = json.loads(card_bytes)
            if (card.get('status') != 'validated' or card.get('locally_validated') is not True
                    or card.get('deployment_ready') is not True):
                return refuse('The installed evidence model has no approved local deployment validation; comparisons remain experimental.')
            validation = card.get('runtime_validation') or {}
            if (validation.get('policy_version') != SETTLEMENT_POLICY_VERSION
                    or validation.get('review_status') != 'approved'
                    or validation.get('scope') != SETTLEMENT_SCOPE
                    or validation.get('threshold') != self.threshold
                    or not card.get('version') or validation.get('model_version') != card['version']
                    or validation.get('threshold_selection_split') != 'calibration'
                    or validation.get('evaluation_split') != 'held_out_test'
                    or not isinstance(validation.get('reviewer'), str) or not validation['reviewer'].strip()
                    or not re.fullmatch(r'[a-f0-9]{64}', str(validation.get('evaluation_manifest_sha256') or ''))):
                return refuse('The deployment review does not bind the required model, scope, threshold and held-out evaluation policy.')
            reviewed = datetime.fromisoformat(str(validation.get('reviewed_at') or '').replace('Z', '+00:00'))
            if reviewed.tzinfo is None or reviewed > datetime.now(timezone.utc):
                return refuse('The deployment review timestamp is missing, ambiguous or in the future.')
            required = ('config.json', 'model.safetensors', 'tokenizer_config.json', 'tokenizer.json')
            expected = validation.get('model_files_sha256') or {}
            if any(not re.fullmatch(r'[a-f0-9]{64}', str(expected.get(name) or '')) for name in required):
                return refuse('The deployment review is missing required model/tokenizer fingerprints.')
            report = validation.get('evaluation_report') or {}
            relative_report = Path(str(report.get('path') or ''))
            report_path = (self.directory / relative_report).resolve()
            if (relative_report.is_absolute() or not report_path.is_relative_to(self.directory.resolve())
                    or not report_path.is_file()
                    or not re.fullmatch(r'[a-f0-9]{64}', str(report.get('sha256') or ''))):
                return refuse('The deployment review has no valid local held-out evaluation report.')
            files = [(self.directory / name, expected[name]) for name in required]
            files.append((report_path, report['sha256']))
            signature = (self.threshold, hashlib.sha256(card_bytes).hexdigest(),
                         tuple((str(path), path.stat().st_size, path.stat().st_mtime_ns) for path, _ in files))
            if self._qualification_cache and self._qualification_cache[0] == signature:
                return dict(self._qualification_cache[1])
            for path, expected_hash in files:
                with path.open('rb') as handle:
                    if hashlib.file_digest(handle, 'sha256').hexdigest() != expected_hash:
                        return refuse('An installed model, tokenizer or evaluation report differs from the reviewed fingerprint.')
            qualified = dict(result, qualified=True, status='qualified', model_version=card['version'],
                             reviewer=validation['reviewer'], reviewed_at=validation['reviewed_at'],
                             evaluation_manifest_sha256=validation['evaluation_manifest_sha256'],
                             reason='A reviewed deployment policy matches the installed artifacts, threshold and English passage-comparison scope.')
            self._qualification_cache = (signature, qualified)
            return dict(qualified)
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            return refuse('Evidence-model deployment validation metadata is absent, invalid or unreadable.')

    @staticmethod
    def _passages(passage, claim):
        if len(passage) <= 800:
            return [passage]
        sentences = [part.strip() for part in re.split(r'(?<=[.!?])\s+|(?<=[.!?])(?=[A-Z])|\n+', passage) if part.strip()]
        terms = claim_tokens(claim)
        ranked = []
        for index, sentence in enumerate(sentences):
            overlap = len(terms & claim_tokens(sentence))
            if not overlap:
                continue
            decisive = bool(re.search(r'\b(false|incorrect|misleading|actually|not|no|fabricated|altered)\b', sentence, re.I))
            # Preserve source wording and nearby context; do not paraphrase a
            # passage into a fact the source never asserted.
            window = ' '.join(sentences[max(0, index - 1):index + 2])[:1200]
            ranked.append((overlap + 3 * decisive, -index, window))
            if decisive:
                ranked.append((overlap + 4, -index, sentence[:1200]))
        selected = []
        for _, _, window in sorted(ranked, reverse=True):
            if window not in selected:
                selected.append(window)
            if len(selected) == 3:
                break
        return selected or [passage[:1200]]

    @staticmethod
    def _hypothesis(claim, passage):
        # Expand only abbreviations whose full form is present in the evidence.
        # This does not supply names, dates, answers, or factual world knowledge.
        for abbreviation, full in [('ED', 'Enforcement Directorate'), ('CM', 'Chief Minister'), ('PM', 'Prime Minister')]:
            if full.casefold() in passage.casefold():
                claim = re.sub(r'\b' + abbreviation + r'\b', full, claim, flags=re.I)
        return claim

    def predict(self, pairs):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        if not pairs:
            return []
        with self.lock:
            if self.model is None:
                self.tokenizer = AutoTokenizer.from_pretrained(str(self.directory), local_files_only=True, trust_remote_code=False)
                self.model = AutoModelForSequenceClassification.from_pretrained(str(self.directory),
                    local_files_only=True, trust_remote_code=False, use_safetensors=True).eval()
                labels = {int(index): label.lower() for index, label in self.model.config.id2label.items()}
                if set(labels.values()) != {'contradiction', 'entailment', 'neutral'}:
                    raise ValueError('Evidence model label mapping is not explicit.')
            outputs = []
            labels = {int(index): label.lower() for index, label in self.model.config.id2label.items()}
            for start in range(0, len(pairs), 4):
                batch = pairs[start:start + 4]
                inputs = self.tokenizer([pair[0] for pair in batch], [pair[1] for pair in batch],
                    padding=True, truncation=True, max_length=384, return_tensors='pt')
                with torch.inference_mode():
                    probabilities = self.model(**inputs).logits.softmax(-1).tolist()
                outputs.extend({labels[index]: score for index, score in enumerate(row)} for row in probabilities)
            return outputs

    @staticmethod
    def _fresh(claim, source):
        if not source_can_settle_current_claim(claim, source):
            return False
        if not re.search(r'\b(current|currently|today|now|latest)\b', claim, re.I):
            return True
        # Import/collection dates never stand in for publication/current evidence.
        value = (source.get('retrieved_at') if source.get('retrieval_status') == 'article_excerpt_retrieved'
                 else source.get('reviewed_at') or source.get('published_at'))
        if not value:
            return False
        try:
            stamp = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
            if stamp.tzinfo is None:
                stamp = stamp.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - stamp.astimezone(timezone.utc)).total_seconds()
            return 0 <= age <= (86400 if source.get('retrieval_status') == 'article_excerpt_retrieved' else 90 * 86400)
        except (ValueError, TypeError):
            return False

    def assess(self, text, verification):
        from backend.report_summary import _source_relevance, _claim_terms, _tokens
        from backend.current_affairs import _office_context
        from backend.verification import source_reliability
        evidence = verification.get('evidence') or []
        assessment = assess_submission(text, evidence)
        assessment['evidence_model'] = {'status': 'not_configured', 'method': 'local_passage_comparison'}
        if not self.ready() or content_language(text) != 'en':
            return assessment
        qualification = self.settlement_qualification()
        settlement_enabled = qualification['qualified']
        assessment['evidence_model'].update(
            status='ready' if settlement_enabled else 'experimental',
            settlement_enabled=settlement_enabled, qualification=qualification,
            base_verdict=assessment['verdict'], candidate_verdict=assessment['verdict'], comparisons=[])
        tasks, references = [], []
        for claim_index, claim in enumerate(assessment['claims']):
            if claim['verdict'] != 'unverified' or claim.get('conflict') or len(claim['text']) > 1800:
                continue
            eligible_sources = 0
            for source_id, source in enumerate(evidence, start=1):
                if source_reliability(source) not in ('official_primary_source', 'published_fact_check_with_rating',
                        'fact_check_publisher_page_unrated', 'established_news_reporting', 'academic_source'):
                    continue
                if not self._fresh(claim['text'], source) or not _source_relevance(claim['text'], source)['relevant']:
                    continue
                passage = str(source.get('snippet') or '').strip()
                if len(passage) < 25 or content_language(passage) != 'en' or source.get('retrieval_status') == 'metadata_only':
                    continue
                # Compare the evidence text itself. Article titles and spreadsheet
                # classification labels are not substituted for source reasoning.
                terms = _claim_terms(self._hypothesis(claim['text'], passage))
                title_terms = _tokens(self._hypothesis(str(source.get('title') or ''), passage))
                title_topic = len(terms & title_terms) >= min(3, len(terms)) and len(terms & title_terms) / max(1, len(terms)) >= .4
                for window in self._passages(passage, claim['text']):
                    words = _tokens(self._hypothesis(window, passage))
                    overlap = len(terms & words)
                    office = _office_context(claim['text'])
                    office_topic = bool(office and office[1] and office[0] in window.casefold()
                                        and all(place in words for place in office[1]))
                    conclusion = bool(re.search(r'\b(false|incorrect|misleading|actually|fabricated|altered|does not|did not|not real)\b', window, re.I))
                    if not (overlap >= min(2, len(terms)) and overlap / max(1, len(terms)) >= .55
                            or office_topic or title_topic and overlap >= 1 and conclusion):
                        continue
                    tasks.append((window, self._hypothesis(claim['text'], passage)))
                    references.append((claim_index, source_id))
                eligible_sources += 1
                if eligible_sources >= 4:
                    break
        if not tasks:
            return assessment
        try:
            scores = self.predict(tasks)
        except (OSError, ValueError, RuntimeError, ImportError, KeyError):
            assessment['evidence_model']['status'] = 'unavailable'
            return assessment
        votes = {index: {'true': [], 'false': []} for index in range(len(assessment['claims']))}
        comparisons = []
        for (claim_index, source_id), pair, probabilities in zip(references, tasks, scores):
            label = max(probabilities, key=probabilities.get)
            strength = probabilities[label]
            # Numerical support requires the asserted values in the passage.
            numbers = {token for token in _tokens(pair[1]) if token.isdigit()}
            values_present = numbers.issubset(_tokens(pair[0]))
            verdict = ('true' if label == 'entailment' and values_present else
                       'false' if label == 'contradiction' else 'unverified')
            quoted_claim = bool(re.search(r'\b(purporting|being shared|users claim|viral claim|post claiming|claimed on social media)\b', pair[0], re.I))
            source_conclusion = bool(re.search(r'\b(but|however|actually|false|incorrect|misleading|fabricated|altered|does not|did not|not real)\b', pair[0], re.I))
            if verdict == 'true' and quoted_claim and not source_conclusion:
                verdict = 'unverified'
            if strength < self.threshold:
                verdict = 'unverified'
            if verdict != 'unverified':
                votes[claim_index][verdict].append(source_id)
            comparisons.append({'claim_index': claim_index, 'source_id': source_id, 'relationship': label,
                                'model_score': round(strength, 4), 'candidate_verdict': verdict,
                                'accepted_verdict': verdict if settlement_enabled else 'unverified',
                                'evidence_quote': pair[0]})
        candidate_claims = deepcopy(assessment['claims'])
        for index, claim in enumerate(candidate_claims):
            if claim['verdict'] != 'unverified':
                continue
            support, contradiction = sorted(set(votes[index]['true'])), sorted(set(votes[index]['false']))
            if support and contradiction:
                claim.update(conflict=True, reason='Retrieved sources produced conflicting evidence comparisons; the claim remains unverified.')
            elif support or contradiction:
                claim.update(verdict='true' if support else 'false', supporting_evidence=support,
                    contradicting_evidence=contradiction, missing_evidence=[],
                    reason=('The local evidence-comparison model found that the retrieved source passage(s) '
                            + ('support' if support else 'contradict') + ' this claim. '
                            'This is an automated comparison of cited evidence, not a guarantee.'))
        verdicts = {claim['verdict'] for claim in candidate_claims}
        candidate_verdict = next(iter(verdicts)) if len(verdicts) == 1 and not assessment.get('unassessed_claim_count') else 'unverified'
        assessment['evidence_model'].update(candidate_verdict=candidate_verdict, candidate_claims=candidate_claims,
                                             comparisons=comparisons)
        if not settlement_enabled:
            # Preserve the deterministic published-review verdict and reasoning.
            # Suggestions cannot enter supporting/contradicting evidence lists.
            assessment['method'] = 'experimental_passage_comparison'
            return assessment
        assessment['claims'] = candidate_claims
        assessment['verdict'] = candidate_verdict
        assessment['supporting_evidence'] = sorted({i for c in assessment['claims'] for i in c['supporting_evidence']})
        assessment['contradicting_evidence'] = sorted({i for c in assessment['claims'] for i in c['contradicting_evidence']})
        assessment['reason'] = ' '.join(claim['reason'] for claim in assessment['claims'])[:1500]
        assessment['method'] = 'local_evidence_comparison'
        return assessment
