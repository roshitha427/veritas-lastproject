"""Check local setup without connecting to services or loading model weights."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
CORE_MODULES = ('flask', 'sqlalchemy', 'pymysql', 'dotenv', 'requests', 'PIL',
                'reportlab', 'numpy', 'sklearn', 'joblib', 'cv2', 'waitress')
INFERENCE_MODULES = ('torch', 'torchvision', 'transformers', 'imageio_ffmpeg')


def module_available(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def read_environment(root=ROOT):
    values = {}
    if module_available('dotenv'):
        from dotenv import dotenv_values
        values.update(dotenv_values(Path(root) / 'backend' / '.env'))
    values.update(os.environ)
    return values


def inspect_setup(root=ROOT, config=None, available=module_available):
    """Return only fixed diagnostics; never include supplied configuration values."""
    root = Path(root)
    config = read_environment(root) if config is None else config
    checks = []

    def add(name, status, detail):
        checks.append({'name': name, 'status': status, 'detail': detail})

    add('python', 'ok' if sys.version_info >= (3, 12) else 'error', 'Use Python 3.12 or newer.')
    missing = [name for name in CORE_MODULES if not available(name)]
    add('runtime_dependencies', 'error' if missing else 'ok',
        'Install requirements.txt; missing modules: ' + ', '.join(missing) if missing else 'Runtime dependencies found.')
    missing = [name for name in INFERENCE_MODULES if not available(name)]
    add('inference_dependencies', 'warning' if missing else 'ok',
        'Install requirements-inference.txt for neural models; missing modules: ' + ', '.join(missing)
        if missing else 'Inference dependencies found; run model smoke checks to verify compatibility.')
    add('database_backend', 'ok' if config.get('DATABASE_BACKEND', 'mysql') == 'mysql' else 'error',
        'This project supports MySQL only.')
    database_url = config.get('DATABASE_URL')
    if database_url:
        try:
            parsed = urlsplit(database_url)
            configured = parsed.scheme in ('mysql', 'mysql+pymysql') and bool(parsed.hostname and parsed.path.strip('/'))
            parsed.port
        except (ValueError, TypeError):
            configured = False
    else:
        configured = bool(config.get('MYSQL_DATABASE') and config.get('MYSQL_USER')
                          and config.get('MYSQL_PASSWORD') is not None)
    add('database_configuration', 'ok' if configured else 'error',
        'Set DATABASE_URL or MYSQL_DATABASE, MYSQL_USER and MYSQL_PASSWORD. Run python -m scripts.check_database to test the connection.')
    production = config.get('APP_ENV') == 'production'
    if production:
        add('session_secret', 'ok' if len(config.get('SECRET_KEY') or '') >= 32 else 'error',
            'Production requires a random SECRET_KEY of at least 32 characters.')
        add('secure_cookie', 'ok' if str(config.get('COOKIE_SECURE', '')).lower() == 'true' else 'error',
            'Production requires COOKIE_SECURE=true.')
        try:
            parsed = urlsplit(config.get('PUBLIC_BASE_URL') or '')
            secure_url = (parsed.scheme == 'https' and bool(parsed.hostname) and not parsed.username
                          and not parsed.password and not parsed.query and not parsed.fragment)
        except (ValueError, TypeError):
            secure_url = False
        add('public_url', 'ok' if secure_url else 'error', 'Production requires an HTTPS PUBLIC_BASE_URL.')
    built = (root / 'frontend' / 'dist' / 'index.html').is_file()
    add('frontend_build', 'ok' if built else ('error' if production else 'warning'),
        'Frontend build found.' if built else 'Run npm --prefix frontend ci and npm --prefix frontend run build.')
    models = (
        ('text_model', config.get('TEXT_MODEL_PATH') or 'models/text', 'config.json'),
        ('image_model', config.get('IMAGE_MODEL_PATH') or 'models/image-v2/model.joblib', None),
        ('ai_writing_model', config.get('AI_TEXT_MODEL_PATH') or 'models/ai-text/model.joblib', None),
    )
    for name, path, filename in models:
        target = root / path
        if filename:
            target /= filename
        add(name, 'ok' if target.is_file() else 'warning',
            'Artifact found; this does not validate model accuracy.' if target.is_file()
            else 'Artifact not installed. Install an exported model bundle or train this model separately.')
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', action='store_true', help='Print machine-readable diagnostics.')
    args = parser.parse_args()
    checks = inspect_setup()
    if args.json:
        print(json.dumps({'checks': checks}, indent=2))
    else:
        for check in checks:
            print(f"[{check['status']}] {check['name']}: {check['detail']}")
    return int(any(check['status'] == 'error' for check in checks))


if __name__ == '__main__':
    raise SystemExit(main())
