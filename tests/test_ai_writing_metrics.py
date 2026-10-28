from pathlib import Path
from types import SimpleNamespace

import pytest

from ml.ai_text import AITextModel


def test_ai_writing_likelihood_is_independent_of_real_news_score(app):
    analyzer = app.extensions['analyzer']
    analyzer.text = lambda text: (.2, {'name': 'BERT fixture', 'status': 'ready', 'version': 'test'})
    analyzer.ai_text = SimpleNamespace(ready=True, predict=lambda text: (.84, {
        'name': 'AI-writing fixture', 'status': 'ready', 'version': 'test',
        'min_words': 80, 'max_words': 800, 'observed_words': 120}))
    result = analyzer.analyze('text', 'Example blog article. ' * 40, None, 'en', [])
    assert result['ai_percentage'] == 84.
    assert result['real_percentage'] == 80.
    assert result['fake_percentage'] == 20.
    assert result['ai_assessment']['observed_words'] == 120
    assert result['ai_assessment']['max_words'] == 800


@pytest.mark.skipif(not Path('models/ai-text/model.joblib').is_file(), reason='Local AI-writing model not installed')
def test_installed_ai_writing_model_scores_article_and_explains_short_input():
    model = AITextModel('models/ai-text')
    short_score, short_card = model.predict('This short example is not a complete blog article.')
    assert short_score is not None and 0 <= short_score <= 1
    assert short_card['status'] == 'ready'
    assert short_card['experimental_input'] is True
    empty_score, empty_card = model.predict('12345 !!!')
    assert empty_score is None and empty_card['status'] == 'too_short'
    article = (
        'A local community library has expanded its weekend opening hours after residents asked for more time '
        'to use its reading rooms and public computers. The revised schedule gives students a quiet place '
        'to prepare for examinations and lets working adults visit outside normal office hours. '
        'Library staff will review attendance figures over the next several months before recommending '
        'whether the arrangement should continue. Volunteers have offered to help visitors find books and '
        'learn basic computer skills. Residents can suggest improvements through a feedback form at the '
        'front desk. The project focuses on access to shared resources, reading opportunities, and practical '
        'support for people who do not have a suitable study space at home.'
    )
    score, card = model.predict(article)
    assert score is not None and 0 <= score <= 1
    assert card['status'] == 'ready'
    assert card['observed_words'] >= card['min_words']
