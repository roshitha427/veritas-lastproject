"""Serve browser integration tests on loopback using a disposable MySQL database."""
import argparse
import os
from pathlib import Path
import re

from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError
from werkzeug.security import check_password_hash, generate_password_hash

ROOT = Path(__file__).resolve().parents[1]
QA_EMAIL = 'qa-admin@example.invalid'
QA_PASSWORD = 'local-browser-qa-only-password'


class QAConfigurationError(ValueError):
    """A fixed, credential-free diagnostic safe to print during QA startup."""


def qa_settings(environ=None, *, live_evidence=False):
    """Require explicit test storage; never inherit the application's database."""
    environ = os.environ if environ is None else environ
    value = environ.get('TEST_DATABASE_URL')
    try:
        url = make_url(value) if value else None
        if (url is None or url.drivername not in ('mysql', 'mysql+pymysql')
                or not url.host or not url.username
                or not re.fullmatch(r'[A-Za-z0-9_]+_test', url.database or '')):
            raise ValueError
    except (ArgumentError, TypeError, ValueError):
        raise QAConfigurationError('Set TEST_DATABASE_URL to an explicit disposable MySQL database whose name ends in _test. Browser tests write test accounts and reports there.') from None
    try:
        port = int(environ.get('QA_PORT', '5001'))
        if not 1 <= port <= 65535:
            raise ValueError
    except (TypeError, ValueError):
        raise QAConfigurationError('QA_PORT must be an integer between 1 and 65535.') from None
    models = ROOT / 'models'
    config = {
        'APP_ENV': 'development', 'TESTING': True,
        'SECRET_KEY': 'browser-qa-only-secret-never-use-in-production',
        'DATABASE_BACKEND': 'mysql',
        'DATABASE_URL': url.set(drivername='mysql+pymysql').render_as_string(hide_password=False),
        'MYSQL_SSL_CA': None, 'MYSQL_POOL_SIZE': 5,
        'SESSION_COOKIE_SECURE': False,
        'PUBLIC_BASE_URL': f'http://127.0.0.1:{port}',
        'DATA_DIR': str(ROOT / '.runtime' / 'browser-data'),
        'ANALYSIS_WORKERS': 1, 'ANALYSIS_QUEUE_SIZE': 3,
        'RECOVER_INTERRUPTED_JOBS': True,
        'EVIDENCE_PROVIDER': 'public' if live_evidence else 'none',
        'GEMINI_API_KEY': None, 'TAVILY_API_KEY': None, 'GEMINI_GROUNDED_SEARCH': False,
        'MAIL_HOST': None, 'MAIL_FROM': None, 'MAIL_USERNAME': None, 'MAIL_PASSWORD': None,
        'MAIL_USE_TLS': False, 'MAIL_USE_SSL': False,
        'TEXT_MODEL_PATH': str(models / 'text'),
        'IMAGE_MODEL_PATH': str(models / 'image-v2' / 'model.joblib'),
        'AI_IMAGE_MODEL_PATH': str(models / 'ai-image'),
        'AI_TEXT_MODEL_PATH': str(models / 'ai-text' / 'model.joblib'),
        'TEMPORAL_MODEL_PATH': str(models / 'video-temporal' / 'model.joblib'),
        'VIDEO_MODEL_PATH': None, 'WHISPER_MODEL': None,
        'EVIDENCE_NLI_MODEL_PATH': str(ROOT / '.runtime' / 'browser-models-disabled' / 'nli'),
        'EVIDENCE_MODEL_PATH': str(ROOT / '.runtime' / 'browser-models-disabled' / 'evidence.joblib'),
        'DATASET_ARTICLES_PATH': str(ROOT / '.runtime' / 'browser-models-disabled' / 'articles.sqlite'),
    }
    return config, port


def seed_qa_admin(store):
    """Create a known test-only account; refuse to overwrite a different account."""
    existing = store.one('users', {'email': QA_EMAIL})
    if existing:
        if (existing.get('role') != 'admin' or existing.get('disabled')
                or not check_password_hash(existing.get('password_hash') or '', QA_PASSWORD)):
            raise QAConfigurationError('The disposable database contains a different QA account. Choose a fresh test database.')
        return
    store.save('users', {'name': 'Browser QA Administrator', 'email': QA_EMAIL,
        'password_hash': generate_password_hash(QA_PASSWORD), 'role': 'admin', 'disabled': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live-evidence', action='store_true',
                        help='Enable the public keyless evidence provider for explicitly selected live browser tests.')
    args = parser.parse_args()
    app = None
    try:
        config, port = qa_settings(live_evidence=args.live_evidence)
        if not (ROOT / 'frontend' / 'dist' / 'index.html').is_file():
            raise QAConfigurationError('Build the frontend before starting browser QA: npm --prefix frontend run build.')
        from backend.app import create_app
        from waitress import serve
        app = create_app(config)
        seed_qa_admin(app.extensions['store'])
        print(f'Browser QA: http://127.0.0.1:{port}', flush=True)
        print('Test-only administrator enabled in the selected disposable MySQL database. Do not expose this server publicly.', flush=True)
        serve(app, host='127.0.0.1', port=port, threads=6,
              max_request_body_size=50 * 1024 * 1024, expose_tracebacks=False,
              clear_untrusted_proxy_headers=True)
        return 0
    except KeyboardInterrupt:
        return 0
    except QAConfigurationError as error:
        print(str(error))
        return 1
    except Exception as error:
        print('Browser QA startup failed (' + type(error).__name__ + '). Check the disposable MySQL server and installed dependencies. Connection settings are not displayed.')
        return 1
    finally:
        if app is not None:
            from scripts.check_system import close_checked_app
            close_checked_app(app)


if __name__ == '__main__':
    raise SystemExit(main())
