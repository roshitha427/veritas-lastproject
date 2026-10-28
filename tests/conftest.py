import os
import re

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import make_url
from backend.app import create_app


@pytest.fixture(scope='session')
def mysql_config():
    url = os.getenv('TEST_DATABASE_URL')
    if not url:
        pytest.skip('Set TEST_DATABASE_URL to a disposable MySQL database ending in _test.')
    parsed = make_url(url)
    if parsed.get_backend_name() != 'mysql' or not re.fullmatch(r'[A-Za-z0-9_]+_test', parsed.database or ''):
        pytest.fail('TEST_DATABASE_URL must identify a disposable MySQL database ending in _test.')
    return {'DATABASE_BACKEND': 'mysql', 'DATABASE_URL': url}


@pytest.fixture
def clean_mysql(mysql_config):
    engine = create_engine(mysql_config['DATABASE_URL'], pool_pre_ping=True)
    def clear():
        existing = set(inspect(engine).get_table_names())
        with engine.begin() as connection:
            for name in ('password_resets', 'reports', 'datasets', 'sources', 'users'):
                if name in existing:
                    connection.execute(text('DELETE FROM ' + name))
    clear()
    yield mysql_config
    clear()
    engine.dispose()


@pytest.fixture
def app(tmp_path, clean_mysql):
    app = create_app({**clean_mysql, 'TESTING': True, 'APP_ENV': 'development', 'SECRET_KEY': 'test-only-key',
        'DATA_DIR': str(tmp_path / 'data'), 'SESSION_COOKIE_SECURE': False,
        'IMAGE_MODEL_PATH': str(tmp_path / 'missing.joblib'),
        'TEXT_MODEL_PATH': str(tmp_path / 'text'), 'AI_IMAGE_MODEL_PATH': str(tmp_path / 'ai-image'),
        'TEMPORAL_MODEL_PATH': str(tmp_path / 'temporal.joblib'),
        'VIDEO_MODEL_PATH': None,
        'AI_TEXT_MODEL_PATH': str(tmp_path / 'ai-text.joblib'),
        'EVIDENCE_NLI_MODEL_PATH': str(tmp_path / 'missing-nli'),
        'DATASET_ARTICLES_PATH': str(tmp_path / 'missing-dataset.sqlite'), 'GEMINI_GROUNDED_SEARCH': False,
        'GEMINI_API_KEY': None,
        'TAVILY_API_KEY': None, 'WHISPER_MODEL': None, 'EVIDENCE_PROVIDER':'none',
        'MAIL_HOST':None, 'MAIL_FROM':None, 'MAIL_USERNAME':None, 'MAIL_PASSWORD':None})
    yield app
    app.extensions['analysis_executor'].shutdown(wait=True)
    app.extensions['mail_executor'].shutdown(wait=True)
    app.extensions['store'].close()


@pytest.fixture
def client(app):
    return app.test_client()


def token(client):
    return client.get('/api/session').json['csrf_token']


def register(client, email='reader@example.com'):
    result = client.post('/api/auth/register', json={'name':'Reader', 'email':email, 'password':'long-test-password-123'},
                         headers={'X-CSRF-Token':token(client)})
    assert result.status_code == 200
    return result.json
