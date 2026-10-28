"""Train a local HC3 AI-writing model with question-disjoint evaluation.

Only author-published JSON data are parsed; no remote dataset code is executed.
The Wiki domain is held out completely from training and model selection.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '3')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '3')
os.environ.setdefault('MKL_NUM_THREADS', '3')

import argparse
import hashlib
import json
from pathlib import Path
import re
import time

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from threadpoolctl import threadpool_limits

from ml.ai_text import AITextModel, clean_ai_text, word_count


SELF_REFERENCE = re.compile(r'\b(as an? (?:ai|artificial intelligence)|(?:ai|large) language model|(?:i am|i\'m) (?:an? )?(?:ai|chatbot)|chatgpt|openai)\b', re.I)


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def key(text, limit=None):
    tokens = re.findall(r'\w+', text.casefold())
    if limit is not None:
        tokens = tokens[:limit]
    return hashlib.sha256(' '.join(tokens).encode()).hexdigest()


def prepare(data_dir, min_words, max_words):
    groups = []
    counts = {}
    for domain in ['reddit_eli5', 'wiki_csai']:
        path = data_dir / (domain + '.jsonl')
        raw_count = 0
        eligible_count = 0
        for line in path.open(encoding='utf8'):
            item = json.loads(line)
            raw_count += 1
            pair = []
            for field, label in [('human_answers', 0), ('chatgpt_answers', 1)]:
                candidates = item.get(field, [])
                selected = None
                for answer in candidates:
                    if not isinstance(answer, str) or SELF_REFERENCE.search(answer):
                        continue
                    cleaned = clean_ai_text(answer, max_words)
                    if word_count(cleaned) >= min_words:
                        selected = cleaned
                        break
                if selected is not None:
                    pair.append({'text': selected, 'label': label})
            if len(pair) != 2:
                continue
            qkey = key(str(item['question']))
            groups.append({'question_key': qkey, 'domain': domain, 'pair': pair,
                           'answer_keys': [key(row['text'], 80) for row in pair]})
            eligible_count += 1
        counts[domain] = {'raw_questions': raw_count, 'eligible_balanced_pairs': eligible_count,
                          'file_sha256': digest(path)}
    # Group questions sharing an answer prefix, including cross-domain duplicates.
    # Remove all implicated groups instead of allowing a shared prefix into splits.
    q_counts = {}
    a_counts = {}
    for group in groups:
        q_counts[group['question_key']] = q_counts.get(group['question_key'], 0) + 1
        for answer_key in group['answer_keys']:
            a_counts[answer_key] = a_counts.get(answer_key, 0) + 1
    unique = [group for group in groups if q_counts[group['question_key']] == 1
              and all(a_counts[value] == 1 for value in group['answer_keys'])]
    counts['removed_duplicate_question_or_answer_prefix_pairs'] = len(groups) - len(unique)
    reddit = [group for group in unique if group['domain'] == 'reddit_eli5']
    wiki = [group for group in unique if group['domain'] == 'wiki_csai']
    train, rest = train_test_split(reddit, test_size=.3, random_state=42)
    validation, test = train_test_split(rest, test_size=.5, random_state=42)
    val_select, val_cal = train_test_split(validation, test_size=.5, random_state=43)
    splits = {'train': train, 'validation_selection': val_select, 'validation_calibration': val_cal,
              'test': test, 'wiki_domain_holdout': wiki}
    frames = {}
    seen_questions = set()
    seen_prefixes = set()
    for split, split_groups in splits.items():
        question_keys = {group['question_key'] for group in split_groups}
        answer_keys = {value for group in split_groups for value in group['answer_keys']}
        if seen_questions.intersection(question_keys) or seen_prefixes.intersection(answer_keys):
            raise AssertionError('Question or answer-prefix leakage across prepared splits.')
        seen_questions.update(question_keys)
        seen_prefixes.update(answer_keys)
        frames[split] = pd.DataFrame([dict(row, question_key=group['question_key'], domain=group['domain'])
                                      for group in split_groups for row in group['pair']])
        frames[split].to_csv(data_dir / (split + '.csv'), index=False)
    counts['prepared_counts'] = {split: {'questions': len(frame) // 2, 'answers': len(frame),
                                        'labels': {str(label): int(count) for label, count in frame['label'].value_counts().items()}}
                                 for split, frame in frames.items()}
    return frames, counts


def metric_report(labels, probabilities):
    from ml.train_image import metrics
    result = metrics(np.asarray(labels), probabilities)
    result['labels'] = ['human', 'ai']
    matrix = result['confusion_matrix']
    result['human_recall'] = matrix[0][0] / sum(matrix[0])
    result['ai_recall'] = matrix[1][1] / sum(matrix[1])
    return result


def main(args):
    started = time.monotonic()
    data_dir = Path(args.data)
    source = json.loads((data_dir / 'source-manifest.json').read_text(encoding='utf8'))
    for filename in ['reddit_eli5.jsonl', 'wiki_csai.jsonl']:
        if digest(data_dir / filename) != source['files'][filename]['sha256']:
            raise ValueError('Source file fingerprint does not match pinned download manifest.')
    frames, counts = prepare(data_dir, args.min_words, args.max_words)
    print(json.dumps({'preparation': counts}), flush=True)
    if len(frames['train']) < 1000 or len(frames['wiki_domain_holdout']) < 50:
        raise ValueError('Insufficient balanced training/domain-holdout examples.')
    vectorizer = FeatureUnion([
        ('word', TfidfVectorizer(ngram_range=(1, 2), max_features=35_000, min_df=3,
                                strip_accents='unicode', sublinear_tf=True, dtype=np.float32)),
        ('char', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=50_000,
                                min_df=3, sublinear_tf=True, dtype=np.float32))])
    with threadpool_limits(limits=3):
        train_x = vectorizer.fit_transform(frames['train']['text'])
        select_x = vectorizer.transform(frames['validation_selection']['text'])
        candidates = []
        selected = None
        for c in [.5, 2., 8.]:
            classifier = LogisticRegression(C=c, max_iter=500, solver='liblinear', random_state=42)
            classifier.fit(train_x, frames['train']['label'])
            predictions = classifier.predict(select_x)
            score = float(balanced_accuracy_score(frames['validation_selection']['label'], predictions))
            candidates.append({'C': c, 'validation_balanced_accuracy': score,
                               'validation_f1': float(f1_score(frames['validation_selection']['label'], predictions))})
            if selected is None or score > selected[0]:
                selected = (score, c, classifier)
            print(json.dumps(candidates[-1]), flush=True)
        classifier = selected[2]
        cal_x = vectorizer.transform(frames['validation_calibration']['text'])
        cal_logits = classifier.decision_function(cal_x)
        sigmoid = lambda logits: 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
        temperature = float(minimize_scalar(lambda value: log_loss(frames['validation_calibration']['label'], sigmoid(cal_logits / value), labels=[0, 1]),
                                             bounds=(.1, 10), method='bounded').x)
        evaluations = {}
        for split in ['test', 'wiki_domain_holdout']:
            frame = frames[split]
            probabilities = sigmoid(classifier.decision_function(vectorizer.transform(frame['text'])) / temperature)
            evaluations[split] = metric_report(frame['label'], probabilities)
            predictions = frame[['question_key', 'domain', 'label']].copy()
            predictions['ai_probability'] = probabilities
            predictions.to_csv(data_dir / (split + '-predictions.csv'), index=False)
    card = {'name': 'HC3 local AI-writing classifier', 'version': '1.0-' + source['revision'][:12],
            'task': 'ai_written_text', 'status': 'experimental', 'languages': ['en'],
            'labels': {'0': 'human', '1': 'ai'}, 'min_words': args.min_words, 'max_words': args.max_words,
            'architecture': 'Word and character TF-IDF with logistic regression', 'selected_C': selected[1],
            'temperature': temperature, 'threshold': .5, 'target_accuracy': .9,
            'target_met': bool(evaluations['test']['accuracy'] >= .9 and evaluations['wiki_domain_holdout']['accuracy'] >= .9),
            'target_scope': '90% required separately on HC3 Reddit question-held-out and Wiki domain-held-out benchmarks.',
            'evaluation': evaluations['test'], 'domain_evaluation': evaluations,
            'validation_candidates': candidates, 'seed': 42,
            'calibration': 'Temperature fitted on a separate validation half; model C selected on other validation half. Tests untouched until final evaluation.',
            'training_counts': counts,
            'preprocessing': 'One eligible human and one AI answer per question. At least 80 English words. Up to 800 whitespace tokens. AI self-reference examples excluded. Tokenization punctuation/contractions normalized. URLs and URL_N placeholders removed. Questions and shared normalized first 80-word answer prefixes removed across the entire corpus before splitting. Vocabulary fitted on training answers only.',
            'dataset': source,
            'license': 'HC3 author dataset card: CC BY-SA 4.0. Reddit ELI5 upstream BSD; Wiki CSAI upstream CC BY-SA. Source attribution retained in source-manifest and downloaded README.',
            'limitations': ['Historical 2023 ChatGPT-versus-human explanatory answers; not a benchmark for contemporary LLMs or news articles.',
                            'Style, collection and source artifacts may inflate detection accuracy despite normalization and question-group separation.',
                            'Does not detect factual truth, image/video/audio generation, paraphrasing or mixed human/AI editing.',
                            'No Hindi/Telugu performance measured; unsupported scripts and texts below 80 words abstain.',
                            'Normalized prefix deduplication does not prove semantic or near-duplicate independence.',
                            'Balanced benchmark calibration does not establish real-world AI prevalence or universal probabilities.'],
            'runtime_seconds': time.monotonic() - started}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    artifact_tmp = output / 'model.joblib.tmp'
    card_tmp = output / 'model-card.json.tmp'
    joblib.dump({'vectorizer': vectorizer, 'classifier': classifier, 'temperature': temperature, 'card': card}, artifact_tmp, compress=3)
    card_tmp.write_text(json.dumps(card, indent=2), encoding='utf8')
    artifact_tmp.replace(output / 'model.joblib')
    card_tmp.replace(output / 'model-card.json')
    adapter = AITextModel(output)
    expected_frame = frames['test'].iloc[0]
    probability, inference_card = adapter.predict(expected_frame['text'])
    expected = float(sigmoid(classifier.decision_function(vectorizer.transform([expected_frame['text']])) / temperature)[0])
    if probability is None or abs(probability - expected) > 1e-8:
        raise AssertionError('Saved adapter does not reproduce trained pipeline inference.')
    if adapter.predict('Too short.')[0] is not None or adapter.predict('\u092f\u0939 \u090f\u0915 \u0938\u092e\u093e\u091a\u093e\u0930 \u0939\u0948')[0] is not None:
        raise AssertionError('Expected short/unsupported-language abstention.')
    print(json.dumps({'evaluation': evaluations, 'target_met': card['target_met'],
                      'artifact': str(output / 'model.joblib'), 'adapter_status': inference_card['status'],
                      'runtime_seconds': card['runtime_seconds']}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/hc3')
    parser.add_argument('--output', default='models/ai-text')
    parser.add_argument('--min-words', type=int, default=80)
    parser.add_argument('--max-words', type=int, default=800)
    main(parser.parse_args())
