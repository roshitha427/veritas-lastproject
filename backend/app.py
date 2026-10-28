import atexit
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from functools import wraps
import hashlib
import hmac
import json
import logging
import os
from pathlib import Path
import re
import secrets
import tempfile
import threading
import time
from urllib.parse import urlparse

from dotenv import load_dotenv
from flask import Flask, abort, g, jsonify, request, send_file, send_from_directory, session
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from sqlalchemy.exc import DBAPIError, IntegrityError
from werkzeug.exceptions import HTTPException
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
from backend.analysis import Analyzer
from backend.admin import register_admin
from backend.mailer import ready as mail_ready, send_password_reset_email
from backend.pdf_report import report_pdf
from backend.store import Store, now
from backend.verification import DEFAULT_SOURCES, verify_submission
from backend.source_recommendations import with_source_recommendations
from backend.verification_review import annotate_verification_review
from backend.report_contract import attach_report_contract

ROOT = Path(__file__).resolve().parents[1]


def create_app(overrides=None):
    load_dotenv(ROOT / 'backend' / '.env')
    app = Flask(__name__, static_folder=None)
    config = dict(APP_ENV=os.getenv('APP_ENV', 'development'), SECRET_KEY=os.getenv('SECRET_KEY'),
                  DATABASE_BACKEND=os.getenv('DATABASE_BACKEND', 'mysql'),
                  DATABASE_URL=os.getenv('DATABASE_URL'),
                  MYSQL_HOST=os.getenv('MYSQL_HOST', '127.0.0.1'),
                  MYSQL_PORT=int(os.getenv('MYSQL_PORT', '3306')),
                  MYSQL_DATABASE=os.getenv('MYSQL_DATABASE', 'veritas'),
                  MYSQL_USER=os.getenv('MYSQL_USER', 'veritas'), MYSQL_PASSWORD=os.getenv('MYSQL_PASSWORD'),
                  MYSQL_SSL_CA=os.getenv('MYSQL_SSL_CA'), MYSQL_POOL_SIZE=int(os.getenv('MYSQL_POOL_SIZE', '5')),
                  DATA_DIR=os.getenv('DATA_DIR', str(ROOT / 'data')),
                  ANALYSIS_WORKERS=int(os.getenv('ANALYSIS_WORKERS', '1')),
                  ANALYSIS_QUEUE_SIZE=int(os.getenv('ANALYSIS_QUEUE_SIZE', '3')),
                  IMAGE_MODEL_PATH=str(ROOT / os.getenv('IMAGE_MODEL_PATH', 'models/image-v2/model.joblib')),
                  AI_IMAGE_MODEL_PATH=str(ROOT / os.getenv('AI_IMAGE_MODEL_PATH', 'models/ai-image')),
                  TEXT_MODEL_PATH=str(ROOT / os.getenv('TEXT_MODEL_PATH', 'models/text')),
                  TAVILY_API_KEY=os.getenv('TAVILY_API_KEY'),
                  GEMINI_API_KEY=os.getenv('GEMINI_API_KEY'),
                  GEMINI_MODEL=os.getenv('GEMINI_MODEL', 'gemini-3.5-flash-lite'),
                  WHISPER_MODEL=str(ROOT / os.getenv('WHISPER_MODEL', 'models/whisper-tiny'))
                      if os.getenv('WHISPER_MODEL', 'models/whisper-tiny') else None,
                  EVIDENCE_PROVIDER=os.getenv('EVIDENCE_PROVIDER', 'public'),
                  EVIDENCE_MODEL_PATH=str(ROOT / os.getenv('EVIDENCE_MODEL_PATH', 'models/evidence/index.joblib')),
                  PUBLIC_BASE_URL=os.getenv('PUBLIC_BASE_URL', 'http://127.0.0.1:5000'),
                  PASSWORD_RESET_TTL_SECONDS=int(os.getenv('PASSWORD_RESET_TTL_SECONDS', '3600')),
                  MAIL_HOST=os.getenv('MAIL_HOST'), MAIL_PORT=int(os.getenv('MAIL_PORT', '587')),
                  MAIL_USERNAME=os.getenv('MAIL_USERNAME'), MAIL_PASSWORD=os.getenv('MAIL_PASSWORD'),
                  MAIL_FROM=os.getenv('MAIL_FROM'), MAIL_USE_TLS=os.getenv('MAIL_USE_TLS', 'true').lower() == 'true',
                  MAIL_USE_SSL=os.getenv('MAIL_USE_SSL', 'false').lower() == 'true',
                  MAIL_TIMEOUT_SECONDS=float(os.getenv('MAIL_TIMEOUT_SECONDS', '12')),
                  MODEL_TARGET_ACCURACY=float(os.getenv('MODEL_TARGET_ACCURACY', '.90')),
                  TEMPORAL_MODEL_PATH=str(ROOT / os.getenv('TEMPORAL_MODEL_PATH', 'models/video-temporal/model.joblib')),
                  VIDEO_MODEL_PATH=str(ROOT / os.environ['VIDEO_MODEL_PATH']) if os.getenv('VIDEO_MODEL_PATH') else None,
                  AI_TEXT_MODEL_PATH=str(ROOT / os.getenv('AI_TEXT_MODEL_PATH', 'models/ai-text/model.joblib')),
                  EVIDENCE_NLI_MODEL_PATH=str(ROOT / os.getenv('EVIDENCE_NLI_MODEL_PATH', 'models/evidence-nli')),
                  DATASET_ARTICLES_PATH=str(ROOT / os.getenv('DATASET_ARTICLES_PATH', 'models/dataset-articles-v2/evidence.sqlite')),
                  GEMINI_GROUNDED_SEARCH=os.getenv('GEMINI_GROUNDED_SEARCH', 'false').lower() == 'true',
                  MAX_CONTENT_LENGTH=50 * 1024 * 1024, SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE='Strict',
                  SESSION_COOKIE_SECURE=os.getenv('COOKIE_SECURE', 'false').lower() == 'true',
                  PERMANENT_SESSION_LIFETIME=timedelta(hours=8), TESTING=False, RECOVER_INTERRUPTED_JOBS=False)
    config.update(overrides or {})
    if config['DATABASE_BACKEND'] != 'mysql':
        raise RuntimeError('Production requires MySQL; DATABASE_BACKEND must be mysql in every environment.')
    if not 1 <= config['ANALYSIS_WORKERS'] <= 8 or not config['ANALYSIS_WORKERS'] <= config['ANALYSIS_QUEUE_SIZE'] <= 32:
        raise RuntimeError('ANALYSIS_WORKERS must be 1-8 and ANALYSIS_QUEUE_SIZE must be between workers and 32.')
    config['DATA_DIR'] = str(Path(config['DATA_DIR']).resolve())
    config.setdefault('REPORT_VIDEO_DIR', str(Path(config['DATA_DIR']) / 'report-videos'))
    if not .5 < config['MODEL_TARGET_ACCURACY'] < 1:
        raise RuntimeError('MODEL_TARGET_ACCURACY must be between .5 and 1.')
    if config['EVIDENCE_PROVIDER'] not in ('local', 'public', 'rss', 'gdelt', 'tavily', 'auto', 'none'):
        raise RuntimeError('EVIDENCE_PROVIDER must be local, public, rss, gdelt, tavily, auto, or none.')
    if not config['SECRET_KEY']:
        if config['APP_ENV'] == 'production':
            raise RuntimeError('Set a long random SECRET_KEY in production.')
        keyfile = Path(config['DATA_DIR']) / 'session.key'
        keyfile.parent.mkdir(parents=True, exist_ok=True)
        if not keyfile.exists():
            keyfile.write_text(secrets.token_hex(32))
        config['SECRET_KEY'] = keyfile.read_text().strip()
    if config['APP_ENV'] == 'production' and not config['SESSION_COOKIE_SECURE']:
        raise RuntimeError('Production requires MySQL and COOKIE_SECURE=true behind HTTPS.')
    if config['APP_ENV'] == 'production':
        if not isinstance(config['SECRET_KEY'], (str, bytes)) or len(config['SECRET_KEY']) < 32:
            raise RuntimeError('Production requires a random SECRET_KEY with at least 32 characters.')
    if not 300 <= config['PASSWORD_RESET_TTL_SECONDS'] <= 86400:
        raise RuntimeError('PASSWORD_RESET_TTL_SECONDS must be between 300 and 86400.')
    if not 1 <= config['MAIL_PORT'] <= 65535 or config['MAIL_TIMEOUT_SECONDS'] <= 0:
        raise RuntimeError('MAIL_PORT and MAIL_TIMEOUT_SECONDS must be positive valid values.')
    public_url = urlparse(config['PUBLIC_BASE_URL'])
    if (public_url.scheme not in ('http', 'https') or not public_url.hostname or public_url.username
            or public_url.password or public_url.query or public_url.fragment):
        raise RuntimeError('PUBLIC_BASE_URL must be an absolute HTTP(S) URL without credentials, query, or fragment.')
    if config['APP_ENV'] == 'production' and public_url.scheme != 'https':
        raise RuntimeError('Production requires HTTPS PUBLIC_BASE_URL.')
    if config['MAIL_USE_TLS'] and config['MAIL_USE_SSL']:
        raise RuntimeError('Choose either MAIL_USE_TLS or MAIL_USE_SSL, not both.')
    app.config.update(config)
    ocr_signer = URLSafeTimedSerializer(app.config['SECRET_KEY'], salt='image-ocr-preview-v1')
    store = Store(app.config)
    atexit.register(store.close)
    analyzer = Analyzer(app.config)
    executor = ThreadPoolExecutor(max_workers=config['ANALYSIS_WORKERS'], thread_name_prefix='veritas-analysis')
    slots = threading.BoundedSemaphore(config['ANALYSIS_QUEUE_SIZE'])
    mail_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='veritas-mail')
    mail_slots = threading.BoundedSemaphore(8)
    app.extensions.update(store=store, analyzer=analyzer, analysis_executor=executor,
                          mail_executor=mail_executor, mail_slots=mail_slots)
    atexit.register(executor.shutdown, wait=False)
    atexit.register(mail_executor.shutdown, wait=False)
    rate_lock = threading.Lock()
    rates = defaultdict(deque)

    if not store.find('sources', limit=1):
        for name, url, description in DEFAULT_SOURCES:
            store.save('sources', dict(name=name, url=url, domain=urlparse(url).hostname.removeprefix('www.'),
                                       description=description, enabled=True))
    # Enable newly imported source families for existing databases on restart.
    # Never reset an administrator's saved enabled/disabled setting.
    existing_domains = {s.get('domain') for s in store.find('sources')}
    for name, url, description in DEFAULT_SOURCES:
        domain = urlparse(url).hostname.removeprefix('www.')
        if domain not in existing_domains:
            store.save('sources', dict(name=name, url=url, domain=domain, description=description, enabled=True))
    # The local executor cannot resume work after a process restart.
    for state in (('queued', 'analyzing') if app.config['RECOVER_INTERRUPTED_JOBS'] else ()):
        for report in store.find('reports', {'status': state}, limit=10000):
            report.update(status='failed', error='Analysis was interrupted by a server restart. Please submit again.')
            store.save('reports', report)

    def public_user(user):
        return {k: user[k] for k in ('id', 'name', 'email', 'role', 'created_at', 'disabled')}

    def json_object():
        # All JSON endpoints accept objects. A valid JSON array or scalar should
        # produce a useful client error instead of an AttributeError and a 500.
        if request.content_length and request.content_length > 64 * 1024:
            abort(413, description='JSON requests must be smaller than 64 KB.')
        body = request.get_json()
        if not isinstance(body, dict):
            abort(400, description='Send a JSON object with the required fields.')
        return body

    def text_field(body, key, default=''):
        value = body.get(key, default)
        if not isinstance(value, str):
            abort(400, description=f'{key} must be text.')
        return value

    def unique_record(collection, document, message):
        try:
            return store.save(collection, document)
        except Exception as exc:
            if isinstance(exc, IntegrityError):
                abort(409, description=message)
            raise

    def release_analysis_resources(filepath):
        try:
            if filepath:
                Path(filepath).unlink(missing_ok=True)
        except OSError:
            app.logger.exception('Could not remove a temporary analysis upload')
        finally:
            # A locked/deleted temporary file must never consume queue capacity.
            slots.release()

    def protect(admin=False):
        def decorate(fn):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                if not g.user:
                    abort(401, description='Please sign in to continue.')
                if admin and g.user['role'] != 'admin':
                    abort(403, description='Administrator access is required.')
                return fn(*args, **kwargs)
            return wrapper
        return decorate

    def throttle(key, limit, period=60):
        if app.config['TESTING']:
            return
        stamp = time.monotonic()
        with rate_lock:
            # Bound IP entries as well as timestamps.
            if len(rates) > 5000:
                rates.clear()
            bucket = rates[key]
            while bucket and bucket[0] < stamp - period:
                bucket.popleft()
            if len(bucket) >= limit:
                abort(429, description='Too many requests. Please wait a minute and try again.')
            bucket.append(stamp)

    @app.before_request
    def before():
        g.user = None
        # Health must report database outages consistently even when the browser
        # has a signed-in session; resolving that user can itself fail offline.
        if session.get('user_id') and request.path != '/api/health':
            user = store.one('users', {'id': session['user_id']})
            if (user and not user.get('disabled')
                    and session.get('session_version', 0) == user.get('session_version', 0)):
                g.user = user
            else:
                session.clear()
        if request.path.startswith('/api/') and request.method in ('POST', 'PATCH', 'DELETE', 'PUT'):
            token = request.headers.get('X-CSRF-Token', '')
            if not token or not hmac.compare_digest(token.encode('utf8'), session.get('csrf', '').encode('utf8')):
                abort(403, description='Your session has expired. Refresh the page and try again.')

    @app.after_request
    def headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        response.headers['Content-Security-Policy'] = "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; font-src 'self'; media-src 'self' blob:; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
        if request.path.startswith('/api/'):
            response.headers['Cache-Control'] = 'no-store'
        return response

    @app.errorhandler(Exception)
    def error(exc):
        if isinstance(exc, HTTPException):
            return jsonify(error=exc.description), exc.code
        if isinstance(exc, DBAPIError):
            app.logger.warning('Database connection unavailable (%s)', type(exc).__name__)
            return jsonify(error='The database is temporarily unavailable. Please try again after the connection is restored.',
                           code='database_unavailable'), 503
        app.logger.exception('Unhandled request failure')
        return jsonify(error='The request could not be completed. Please try again.'), 500

    @app.get('/api/health')
    def health():
        try:
            store.ping()
            details = {'storage': store.sync_status()} if hasattr(store, 'sync_status') else {}
            return jsonify(status='ok', database=store.kind, database_connected=True,
                           password_reset_email=mail_ready(app.config), models=analyzer.readiness(), **details)
        except Exception:
            return jsonify(status='degraded', database=store.kind, database_connected=False,
                           password_reset_email=mail_ready(app.config)), 503

    @app.get('/api/session')
    def get_session():
        session.setdefault('csrf', secrets.token_urlsafe(32))
        return jsonify(user=public_user(g.user) if g.user else None, csrf_token=session['csrf'])

    def auth_response(user):
        session.clear()
        session.update(user_id=user['id'], session_version=user.get('session_version', 0), csrf=secrets.token_urlsafe(32))
        session.permanent = True
        return jsonify(user=public_user(user), csrf_token=session['csrf'])

    @app.post('/api/auth/register')
    def register():
        throttle(('auth', request.remote_addr), 10)
        body = json_object()
        name, email, password = text_field(body, 'name').strip(), text_field(body, 'email').strip().lower(), text_field(body, 'password')
        if not 2 <= len(name) <= 80 or not re.fullmatch(r'[^\s@]{1,64}@[^\s@.]+\.[^\s@]{2,}', email) or len(email) > 254:
            abort(400, description='Enter your name and a valid email address.')
        if not 12 <= len(password) <= 128:
            abort(400, description='Use a password between 12 and 128 characters.')
        if store.one('users', {'email': email}):
            abort(409, description='This email is already registered. Please sign in.')
        user = unique_record('users', dict(name=name, email=email, password_hash=generate_password_hash(password), role='user', disabled=False, session_version=0),
                             'This email is already registered. Please sign in.')
        return auth_response(user)

    @app.post('/api/auth/login')
    def login():
        throttle(('auth', request.remote_addr), 10)
        body = json_object()
        email = text_field(body, 'email').strip().lower()
        password = text_field(body, 'password')
        user = store.one('users', {'email': email})
        # Constant-cost dummy check when an email does not exist.
        valid = check_password_hash(user['password_hash'] if user else DUMMY_HASH, password[:129])
        if not user or not valid or user.get('disabled'):
            abort(401, description='Email or password is incorrect, or the account is disabled.')
        return auth_response(user)

    @app.post('/api/auth/forgot-password')
    def forgot_password():
        throttle(('password-reset-ip', request.remote_addr), 10, period=3600)
        body = json_object()
        email = text_field(body, 'email').strip().lower()
        if not re.fullmatch(r'[^\s@]{1,64}@[^\s@.]+\.[^\s@]{2,}', email) or len(email) > 254:
            abort(400, description='Enter a valid email address.')
        if not mail_ready(app.config):
            abort(503, description='Password reset email is not configured. Contact the site administrator.')
        throttle(('password-reset-email', email), 5, period=3600)
        user = store.one('users', {'email': email})
        if user and not user.get('disabled'):
            # Replace earlier links for this account and store only a digest of
            # the new opaque token. Expired tokens are rejected on consumption.
            for old in store.find('password_resets', {'user_id': user['id']}, limit=100):
                store.delete('password_resets', old['id'])
            token = secrets.token_urlsafe(32)
            token_hash = hashlib.sha256(token.encode('utf8')).hexdigest()
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=app.config['PASSWORD_RESET_TTL_SECONDS'])
            store.save('password_resets', {'id': token_hash, 'user_id': user['id'], 'expires_at': expires_at})
            reset_url = app.config['PUBLIC_BASE_URL'].rstrip('/') + '/#reset/' + token

            mail_slots = app.extensions['mail_slots']
            if not mail_slots.acquire(blocking=False):
                store.delete('password_resets', token_hash)
                app.logger.warning('Password reset email queue is full for account %s', user['id'])
            else:
                def deliver_reset():
                    try:
                        send_password_reset_email(app.config, email, user.get('name', ''), reset_url)
                    except Exception:
                        store.delete('password_resets', token_hash)
                        app.logger.exception('Password reset email delivery failed for account %s', user['id'])
                    finally:
                        mail_slots.release()

                try:
                    app.extensions['mail_executor'].submit(deliver_reset)
                except RuntimeError:
                    mail_slots.release()
                    store.delete('password_resets', token_hash)
                    app.logger.exception('Password reset email could not be queued for account %s', user['id'])
        return jsonify(message='If an account uses that email, a reset link will be sent shortly.'), 200

    @app.post('/api/auth/reset-password')
    def reset_password():
        throttle(('password-reset-ip', request.remote_addr), 20, period=3600)
        body = json_object()
        token = text_field(body, 'token')
        password = text_field(body, 'password')
        if len(token) > 200 or not 12 <= len(password) <= 128:
            abort(400, description='The reset link or new password is invalid.')
        token_hash = hashlib.sha256(token.encode('utf8')).hexdigest()
        reset = store.take('password_resets', token_hash)
        if not reset:
            abort(400, description='The reset link is invalid or expired. Request a new link.')
        expires_at = reset.get('expires_at')
        if isinstance(expires_at, str):
            try:
                expires_at = datetime.fromisoformat(expires_at)
            except ValueError:
                expires_at = None
        if expires_at and expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if not isinstance(expires_at, datetime) or expires_at <= datetime.now(timezone.utc):
            abort(400, description='The reset link is invalid or expired. Request a new link.')
        user = store.one('users', {'id': reset.get('user_id')})
        if not user or user.get('disabled'):
            abort(400, description='The reset link is invalid or expired. Request a new link.')
        user['password_hash'] = generate_password_hash(password)
        user['session_version'] = int(user.get('session_version', 0)) + 1
        store.save('users', user)
        return jsonify(ok=True)

    @app.post('/api/auth/logout')
    def logout():
        session.clear()
        session['csrf'] = secrets.token_urlsafe(32)
        return jsonify(ok=True, csrf_token=session['csrf'])

    @app.patch('/api/profile')
    @protect()
    def update_profile():
        throttle(('profile', g.user['id']), 10)
        body = json_object()
        allowed = {'name', 'email', 'current_password', 'password', 'confirm_password'}
        if set(body) - allowed or not set(body).intersection({'name', 'email', 'password'}):
            abort(400, description='Update your name, email address, or password.')

        user = dict(g.user)
        if 'name' in body:
            name = text_field(body, 'name').strip()
            if not 2 <= len(name) <= 80:
                abort(400, description='Name must be between 2 and 80 characters.')
            user['name'] = name

        email_changed = False
        if 'email' in body:
            email = text_field(body, 'email').strip().lower()
            if not re.fullmatch(r'[^\s@]{1,64}@[^\s@.]+\.[^\s@]{2,}', email) or len(email) > 254:
                abort(400, description='Enter a valid email address.')
            email_changed = email != user['email']
            user['email'] = email

        password_changed = 'password' in body
        if password_changed:
            password = text_field(body, 'password')
            confirmation = text_field(body, 'confirm_password')
            if not 12 <= len(password) <= 128:
                abort(400, description='Use a password between 12 and 128 characters.')
            if password != confirmation:
                abort(400, description='The new passwords do not match.')

        if email_changed or password_changed:
            current_password = text_field(body, 'current_password')
            if not check_password_hash(g.user['password_hash'], current_password[:129]):
                abort(401, description='Enter your current password to change your email or password.')
        if password_changed:
            user['password_hash'] = generate_password_hash(password)

        updated = unique_record('users', user, 'This email is already registered.')
        return jsonify(user=public_user(updated))

    @app.get('/api/sources')
    def sources():
        return jsonify(items=store.find('sources', {'enabled': True}))

    @app.get('/api/reports')
    @protect()
    def reports():
        items = store.find('reports', {'owner_id': g.user['id']}, limit=200)
        from backend.report_media import public_report
        for item in items:
            if attach_report_contract(item):
                store.save('reports', item)
        return jsonify(items=[{k: v for k, v in public_report(item, app.config['REPORT_VIDEO_DIR']).items() if k not in ('content',)} for item in items])

    def owned_report(report_id):
        report = store.one('reports', {'id': report_id, 'owner_id': g.user['id']})
        if not report:
            abort(404, description='Report not found.')
        return report

    def ensure_report_summary(report):
        """Upgrade older completed reports when they are opened or downloaded."""
        if report.get('status') != 'complete':
            return report
        result = report.get('result') or {}
        metrics_changed = False
        if report.get('type') == 'text' and result.get('report_summary'):
            from backend.report_summary import apply_knowledge_policy
            previous = result['report_summary']
            corrected = apply_knowledge_policy(previous, report.get('content') or result.get('claim_text') or '')
            if corrected != previous:
                result['report_summary'] = corrected
                metrics_changed = True
        if report.get('type') in ('image', 'video'):
            from backend.generation import media_origin_assessment, retire_obsolete_origin
            from backend.analysis import media_claim_text, transcript_claim_text
            retired = retire_obsolete_origin(result, getattr(analyzer.generation, 'directory', None))
            metrics_changed = metrics_changed or retired
            origin = media_origin_assessment(
                result.get('ai_percentage'), result.get('ai_assessment') or {},
                (result.get('media_inspection') or {}).get('visual_type', 'unclear'), result.get('content_credentials'),
                kind=report['type'], video_assessment=result.get('video_assessment'))
            origin_changed = result.get('media_origin') != origin
            result['media_origin'] = origin
            metrics_changed = metrics_changed or origin_changed
            summary = result.get('report_summary') or {}
            if origin_changed and origin['verdict'] == 'ai_content_declared' and summary.get('mode') == 'media_only':
                summary['headline'] = origin['headline']
                summary['summary'] = origin['reason'] + ' ' + str((result.get('media_inspection') or {}).get('description') or '')
                summary['model_signal'] = 'Raw classifier signals are separate from the embedded content declaration.'
            if summary and origin['verdict'] in ('inconclusive', 'classification_unavailable'):
                signal = f"{report['type'].capitalize()} origin could not be determined reliably. Raw experimental class signals are not reliable origin probabilities."
                if summary.get('model_signal') != signal:
                    summary['model_signal'] = signal
                    metrics_changed = True
            if retired and summary.get('mode') == 'media_only':
                summary['headline'] = 'Reanalyze with the updated detector'
                summary['summary'] = origin['reason']
            inspection = result.get('media_inspection') or {}
            ocr = str(inspection.get('visible_text') or result.get('ocr_text') or '')
            is_video = report['type'] == 'video'
            if is_video and any(item.get('label') == 'Faces detected (maximum per frame)' and str(item.get('value')) == '0'
                                for item in result.get('observations') or []):
                temporal = result.get('temporal_assessment') or {}
                if temporal.get('status') == 'ready':
                    temporal.update(status='out_of_scope', reason='No human face was detected in the sampled frames. '
                        'The face-swap model cannot determine whether this video is AI-generated.')
                    metrics_changed = True
            from backend.media_labels import media_labels
            labels = media_labels(report['type'], result)
            if result.get('media_labels') != labels:
                result['media_labels'] = labels
                metrics_changed = True
            # Repair old watermark-only reports without dropping genuine claims or transcripts.
            watermark_only = (bool(ocr.strip()) and (not media_claim_text(ocr) or inspection.get('text_role') == 'watermark_or_label')
                              and not str(report.get('content') or '').strip()
                              and not str(result.get('transcript') or '').strip())
            nonclaim_video = (is_video and not str(report.get('content') or '').strip()
                              and not transcript_claim_text(result.get('transcript'))
                              and (not media_claim_text(ocr) or inspection.get('text_role') == 'watermark_or_label'))
            watermark_repaired = False
            if (watermark_only or nonclaim_video) and summary.get('mode') != 'media_only':
                watermark_repaired = True
                result['claim_text'] = ''
                result['media_content_mode'] = 'text_detected'
                result['verification'] = {'status': 'not_applicable', 'provider_status': 'not_run', 'evidence': []}
                result.pop('media_text_analysis', None)
                result.pop('speech_text_assessment', None)
                summary = {'schema_version': 5, 'mode': 'media_only',
                           'headline': 'Origin inconclusive' if origin['verdict'] == 'inconclusive' else 'Media analyzed',
                           'summary': inspection.get('description') or 'Media inspected.',
                           'model_signal': summary.get('model_signal', '') if origin.get('scores_publishable') else 'Image origin could not be determined reliably. Raw experimental class signals are not reliable origin probabilities.',
                           'reason': ('No substantive factual claim was selected from the extracted text. Speech and on-screen text are shown separately.'
                                      if is_video else 'The visible text is a camera watermark or attribution, not a factual claim.')}
                result['report_summary'] = summary
                metrics_changed = True
            if (origin_changed or watermark_repaired) and summary.get('mode') == 'media_only' and origin['verdict'] == 'inconclusive':
                summary['headline'] = origin.get('headline') or 'Origin inconclusive'
                description = str(summary.get('summary') or '')
                for old_label in ('The model classifies this as likely AI-generated.',
                                  'The model classifies this as a likely real image.',
                                  'The model did not flag this as AI-generated.'):
                    description = description.replace(old_label, '').strip()
                summary['summary'] = f"{report['type'].capitalize()} origin is inconclusive. " + result['media_origin']['reason'] + ' ' + description
                metrics_changed = True
        from backend.evidence_policy import repair_unqualified_evidence
        evidence_claim = str(result.get('claim_text') or report.get('content')
                             or (result.get('media_text_analysis') or {}).get('content') or '')
        if repair_unqualified_evidence(result, evidence_claim, report.get('type', 'text'),
                                       analyzer.evidence_verifier):
            metrics_changed = True
        if report.get('type') == 'text':
            content = str(report.get('content') or '')
            from backend.analysis import text_input_metrics
            counts = text_input_metrics(content, len((result.get('verification') or {}).get('evidence') or []))
            if result.get('text_metrics') != counts:
                result['text_metrics'] = counts
                metrics_changed = True
            if (result.get('text_model_assessment') or {}).get('status') == 'claim_out_of_domain':
                try:
                    probability, card = analyzer.text(content)
                    if probability is not None:
                        assessment = analyzer._text_model_assessment(probability, card)
                        result.update(fake_percentage=assessment['fake_percentage'],
                                      real_percentage=assessment['real_percentage'], confidence=assessment['confidence'],
                                      text_model_assessment=assessment,
                                      metric_note='Short-text scores are experimental news-pattern signals, not fact-check verdicts.',
                                      scope='Experimental short-text news-pattern classification; factual evidence is assessed separately.')
                        models = [item for item in result.get('models', []) if item.get('status') != 'claim_out_of_domain']
                        result['models'] = models + [card]
                        for container in (result, result.get('report_summary') or {}):
                            if 'limitations' in container:
                                container['limitations'] = [note for note in container['limitations']
                                    if not ('abstain' in note.lower() and 'classifier' in note.lower())]
                                container['limitations'].append(result['metric_note'])
                        if result.get('report_summary'):
                            from backend.report_summary import _model_signal
                            result['report_summary']['model_signal'] = _model_signal(result, 'text')
                        metrics_changed = True
                except Exception:
                    app.logger.warning('Saved short-text metrics could not be refreshed; source verdict preserved.')
            if (result.get('ai_assessment') or {}).get('status') == 'too_short' and analyzer.ai_text and analyzer.ai_text.ready:
                refreshed = {'limitations': [], 'models': []}
                analyzer._assess_ai_writing(content, refreshed)
                if refreshed.get('ai_percentage') is not None:
                    result.update(ai_percentage=refreshed['ai_percentage'], ai_assessment=refreshed['ai_assessment'])
                    result['models'] = result.get('models', []) + refreshed['models']
                    result['limitations'] = [note for note in result.get('limitations', [])
                        if not ('AI-writing model requires at least' in note or 'Short-text authorship is not assessed' in note)]
                    result['limitations'].extend(refreshed['limitations'])
                    metrics_changed = True
        if result.get('report_summary'):
            previous_summary = result['report_summary']
            recommendation_claim = (result.get('claim_text') or report.get('content')
                                    or (result.get('media_text_analysis') or {}).get('content')
                                    or result.get('transcript') or '')
            previous_verification = result.get('verification')
            annotate_verification_review(result, recommendation_claim)
            if result.get('report_summary') != previous_summary or result.get('verification') != previous_verification:
                metrics_changed = True
            previous_summary = result['report_summary']
            corrected_summary = with_source_recommendations(previous_summary, recommendation_claim, store.find('sources'))
            if corrected_summary != previous_summary:
                result['report_summary'] = corrected_summary
                metrics_changed = True
        report['result'] = result
        if attach_report_contract(report):
            metrics_changed = True
        if metrics_changed:
            report.update(result=result, metrics_updated_at=now())
            store.save('reports', report)
        if (result.get('report_summary') or {}).get('schema_version', 0) >= 4:
            return report

        from backend.report_summary import filter_relevant_evidence, summarize_report

        kind = report.get('type', 'text')
        claim = str(report.get('content') or result.get('transcript') or '').strip()
        transcript = str(result.get('transcript') or '').strip()
        if transcript and transcript not in claim:
            claim = '\n'.join(part for part in (claim, transcript) if part)
        verification = result.get('verification') or {'evidence': [], 'provider_status': 'unavailable'}
        for item in verification.get('evidence') or []:
            if isinstance(item, dict):
                item.setdefault('retrieved_at', verification.get('checked_at'))
                item.setdefault('source_type', item.get('kind', 'source'))
        verification = filter_relevant_evidence(claim, verification)

        # Older records may not have saved evidence, or may contain only
        # unrelated matches. Retry retrieval once before settling on Unverified.
        if claim and not verification.get('evidence'):
            try:
                fresh = verify_submission(
                    claim,
                    report.get('language', 'en'),
                    store.find('sources'),
                    provider=app.config.get('EVIDENCE_PROVIDER', 'local'),
                    evidence_path=app.config.get('EVIDENCE_MODEL_PATH'),
                    dataset_path=app.config.get('DATASET_ARTICLES_PATH'),
                    tavily_key=app.config.get('TAVILY_API_KEY'),
                    gemini_key=app.config.get('GEMINI_API_KEY'),
                    gemini_model=app.config.get('GEMINI_MODEL', 'gemini-3.5-flash-lite'),
                    grounded_search=app.config.get('GEMINI_GROUNDED_SEARCH', False),
                )
                fresh = filter_relevant_evidence(claim, fresh)
                if fresh.get('evidence') or not verification.get('evidence'):
                    verification = fresh
            except Exception:
                app.logger.warning('Evidence refresh failed while upgrading a saved report.')

        result['verification'] = verification
        if kind == 'text' and result.get('text_metrics'):
            result['text_metrics']['source_count'] = len(verification.get('evidence') or [])
        result['report_summary'] = summarize_report(
            claim, kind, result, verification,
            api_key=app.config.get('GEMINI_API_KEY'),
            model=app.config.get('GEMINI_MODEL', 'gemini-3.5-flash-lite'),
        )
        annotate_verification_review(result, claim)
        result['report_summary'] = with_source_recommendations(result['report_summary'], claim, store.find('sources'))
        report['result'] = result
        attach_report_contract(report)
        store.save('reports', report)
        return report

    @app.get('/api/reports/<report_id>')
    @protect()
    def get_report(report_id):
        from backend.report_media import public_report
        return jsonify(public_report(ensure_report_summary(owned_report(report_id)), app.config['REPORT_VIDEO_DIR']))

    @app.post('/api/reports/<report_id>/ask')
    @protect()
    def ask_report(report_id):
        throttle(('report-question', g.user['id']), 20, period=3600)
        report = owned_report(report_id)
        if report.get('status') != 'complete':
            abort(409, description='Wait for the report to finish before asking about its sources.')
        question = text_field(json_object(), 'question').strip()
        if not 3 <= len(question) <= 1000:
            abort(400, description='Enter a question between 3 and 1,000 characters.')
        report = ensure_report_summary(report)
        from backend.rag import answer_from_report
        return jsonify(answer_from_report(question, report,
            api_key=app.config.get('GEMINI_API_KEY'), model=app.config.get('GEMINI_MODEL', 'gemini-3.5-flash-lite')))

    @app.delete('/api/reports/<report_id>')
    @protect()
    def delete_report(report_id):
        report = owned_report(report_id)
        if report['status'] in ('queued', 'analyzing'):
            abort(409, description='Wait for analysis to finish before deleting this report.')
        store.delete('reports', report_id)
        from backend.report_media import remove_report_media
        remove_report_media(report, app.config['REPORT_VIDEO_DIR'])
        return jsonify(ok=True)

    @app.get('/api/reports/<report_id>/preview')
    @protect()
    def report_preview(report_id):
        from io import BytesIO
        from backend.report_media import preview_bytes
        image = preview_bytes(owned_report(report_id))
        if image is None:
            abort(404, description='No image preview was saved for this report.')
        response = send_file(BytesIO(image), mimetype='image/jpeg', etag=False)
        response.headers['Cache-Control'] = 'private, no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

    @app.get('/api/reports/<report_id>/poster')
    @protect()
    def report_video_poster(report_id):
        from io import BytesIO
        from backend.report_media import video_poster_bytes
        image = video_poster_bytes(owned_report(report_id))
        if image is None:
            abort(404, description='No video poster was saved for this report.')
        response = send_file(BytesIO(image), mimetype='image/jpeg', etag=False)
        response.headers['Cache-Control'] = 'private, no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

    @app.get('/api/reports/<report_id>/video')
    @protect()
    def report_video(report_id):
        from backend.report_media import public_report, video_preview_path
        report = owned_report(report_id)
        path = video_preview_path(report, app.config['REPORT_VIDEO_DIR'])
        if path is None:
            abort(404, description='No video preview was saved for this report.')
        metadata = public_report(report, app.config['REPORT_VIDEO_DIR']).get('video_preview') or {}
        response = send_file(path, mimetype=metadata.get('mime_type', 'application/octet-stream'), conditional=True)
        response.headers['Cache-Control'] = 'private, no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

    @app.get('/api/reports/<report_id>/pdf')
    @protect()
    def pdf(report_id):
        from io import BytesIO
        report = ensure_report_summary(owned_report(report_id))
        if report['status'] != 'complete':
            abort(409, description='The report is not ready yet.')
        return send_file(BytesIO(report_pdf(report, detailed=request.args.get('details') != '0')), mimetype='application/pdf', as_attachment=True,
                         download_name=f"verification-report-{report['id'][:10]}.pdf")

    def validate_image_upload(filepath):
        """Apply the same decode and decompression bounds to preview and analysis."""
        from PIL import Image
        try:
            with Image.open(filepath) as image:
                if image.format not in ('JPEG', 'PNG', 'WEBP') or image.width * image.height > 30_000_000:
                    raise ValueError()
                image.verify()
        except Exception:
            abort(400, description='The image is invalid or exceeds 30 megapixels.')

    @app.post('/api/media/ocr-preview')
    @protect()
    def ocr_preview():
        """Read an image before the user chooses wording for factual verification."""
        throttle(('ocr-preview', g.user['id']), 10)
        if request.form.get('type', 'image') != 'image':
            abort(400, description='OCR preview supports image uploads only.')
        media = request.files.get('file')
        if not media or not media.filename:
            abort(400, description='Choose an image to review.')
        suffix = Path(secure_filename(media.filename)).suffix.lower()
        if suffix not in {'.png', '.jpg', '.jpeg', '.webp'}:
            abort(400, description='Use a JPG, PNG or WebP image.')
        if not slots.acquire(blocking=False):
            abort(429, description='All analysis slots are busy. Please try again shortly.')
        filepath = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
                filepath = handle.name
            media.save(filepath)
            if not os.path.getsize(filepath):
                abort(400, description='The uploaded file is empty.')
            validate_image_upload(filepath)
            from PIL import Image, ImageOps
            from backend.media_inspection import inspect_media
            with Image.open(filepath) as image:
                image.load()
                frame = ImageOps.exif_transpose(image).convert('RGB')
            try:
                inspection = inspect_media([frame], app.config.get('GEMINI_API_KEY'),
                    app.config.get('GEMINI_MODEL', 'gemini-3.5-flash-lite'))
            finally:
                frame.close()
            with open(filepath, 'rb') as uploaded:
                digest = hashlib.file_digest(uploaded, 'sha256').hexdigest()
            preview_token = ocr_signer.dumps({'owner_id': g.user['id'], 'sha256': digest, 'inspection': inspection})
            return jsonify(inspection=inspection, preview_token=preview_token,
                message='Compare this transcription with the image and correct it before verifying. User-reviewed text is a claim to check, not evidence.')
        finally:
            release_analysis_resources(filepath)

    @app.post('/api/analyze')
    @protect()
    def analyze():
        throttle(('analysis', g.user['id']), 10)
        kind = request.form.get('type', 'text')
        text = request.form.get('text', '').strip()
        language = request.form.get('language', 'en')
        if kind not in ('text', 'image', 'video') or language not in ('en', 'hi', 'te'):
            abort(400, description='Select a supported content type and language.')
        if len(text) > 12000 or (kind == 'text' and len(text) < 20):
            abort(400, description='Enter between 20 and 12,000 characters of text.')
        reviewed_ocr = None
        preview_payload = None
        review_flag = request.form.get('ocr_reviewed', '')
        preview_token = request.form.get('ocr_preview_token', '')
        has_review_fields = any(name in request.form for name in ('ocr_reviewed', 'reviewed_ocr_text', 'ocr_preview_token'))
        if has_review_fields:
            if kind != 'image' or review_flag != 'true' or 'reviewed_ocr_text' not in request.form:
                abort(400, description='Reviewed OCR is supported for images and requires explicit review.')
            corrected_text = request.form['reviewed_ocr_text']
            if len(corrected_text) > 4000:
                abort(400, description='Reviewed image text must be 4,000 characters or fewer.')
            reviewed_ocr = {'text': corrected_text}
            if preview_token:
                if len(preview_token) > 32000:
                    abort(400, description='Review this image text again; its preview is invalid.')
                try:
                    preview_payload = ocr_signer.loads(preview_token, max_age=1800)
                except (BadSignature, SignatureExpired):
                    abort(400, description='Review this image text again; its preview expired or is invalid.')
                if not isinstance(preview_payload, dict) or preview_payload.get('owner_id') != g.user['id']:
                    abort(400, description='Review this image text again in your own session.')
        media = request.files.get('file')
        if kind != 'text' and (not media or not media.filename):
            abort(400, description='Choose an image or video to analyze.')
        filename = secure_filename(media.filename) if media else ''
        allowed = {'.png', '.jpg', '.jpeg', '.webp'} if kind == 'image' else {'.mp4', '.webm', '.mov'}
        if kind != 'text' and Path(filename).suffix.lower() not in allowed:
            abort(400, description='Use JPG, PNG, WebP for images, or MP4, MOV, WebM for videos.')
        if not slots.acquire(blocking=False):
            abort(429, description='All analysis slots are busy. Please try again shortly.')
        filepath = None
        report = None
        try:
            if media and kind != 'text':
                # Random, private temporary names. Never serve user uploads as active web assets.
                with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix.lower(), delete=False) as handle:
                    filepath = handle.name
                media.save(filepath)
                if os.path.getsize(filepath) == 0:
                    abort(400, description='The uploaded file is empty.')
                if kind == 'image':
                    validate_image_upload(filepath)
                    if preview_payload is not None:
                        with open(filepath, 'rb') as uploaded:
                            digest = hashlib.file_digest(uploaded, 'sha256').hexdigest()
                        if preview_payload.get('sha256') != digest or not isinstance(preview_payload.get('inspection'), dict):
                            abort(400, description='The image changed. Review the new image text before verifying.')
                        reviewed_ocr['inspection'] = preview_payload['inspection']
            report_document = dict(owner_id=g.user['id'], type=kind, language=language,
                title=text[:110] if text else filename, content=text, filename=filename, status='queued', result=None)
            if kind == 'image':
                from backend.report_media import make_image_preview
                report_document['_image_preview'] = make_image_preview(filepath)
            report = store.save('reports', report_document)
            evidence_sources = store.find('sources')
            def run():
                try:
                    report['status'] = 'analyzing'
                    store.save('reports', report)
                    report['result'] = (analyzer.analyze(kind, text, filepath, language, evidence_sources, reviewed_ocr=reviewed_ocr)
                        if reviewed_ocr is not None else analyzer.analyze(kind, text, filepath, language, evidence_sources))
                    if kind == 'video':
                        from backend.report_media import make_video_preview
                        report['_video_preview'] = make_video_preview(filepath, report['id'], app.config['REPORT_VIDEO_DIR'])
                    report.update(status='complete', completed_at=now())
                    attach_report_contract(report)
                except Exception as exc:
                    logging.exception('Analysis failed for report %s', report['id'])
                    report.update(status='failed', error=str(exc) if isinstance(exc, ValueError) else 'Analysis could not be completed. Check the installed model and media format.')
                finally:
                    try:
                        store.save('reports', report)
                    finally:
                        release_analysis_resources(filepath)
            executor.submit(run)
            return jsonify(id=report['id'], status='queued'), 202
        except Exception:
            # A report may already have been inserted when queue submission or
            # source loading fails. Do not leave it queued forever in history.
            if report is not None:
                try:
                    report.update(status='failed', error='Analysis could not be scheduled. Please submit again.')
                    store.save('reports', report)
                except Exception:
                    app.logger.exception('Could not persist failed scheduling for report %s', report['id'])
            release_analysis_resources(filepath)
            raise

    register_admin(app, store=store, analyzer=analyzer, protect=protect,
                   public_user=public_user, json_object=json_object,
                   text_field=text_field, unique_record=unique_record)

    @app.get('/')
    @app.get('/<path:path>')
    def frontend(path='index.html'):
        if path.startswith('api/'):
            abort(404, description='API route not found.')
        directory = ROOT / 'frontend' / 'dist'
        if not directory.exists():
            return jsonify(message='Frontend not built. Run npm run build or use the Vite development server.'), 503
        candidate = (directory / path).resolve()
        if directory.resolve() not in candidate.parents:
            abort(404)
        return send_from_directory(directory, path if candidate.is_file() else 'index.html')

    return app


DUMMY_HASH = generate_password_hash('constant-cost-dummy-password')


def create_server_app():
    """Recover jobs only on server startup, never during admin/dataset CLI commands."""
    return create_app({'RECOVER_INTERRUPTED_JOBS': True})

if __name__ == '__main__':
    create_server_app().run(host='127.0.0.1', port=int(os.getenv('PORT', '5000')), debug=False)
