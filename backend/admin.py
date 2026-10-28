"""Administrator endpoints and private account setup commands.

The application supplies its shared authentication and input helpers so admin
requests keep the same session, CSRF, validation, and durable-store behavior.
"""
import json
from pathlib import Path
import re
from urllib.parse import urlparse

import click
from flask import abort, g, jsonify
from werkzeug.security import generate_password_hash

from backend.mailer import ready as mail_ready
from backend.generation import _origin_thresholds
from backend.video_model import video_detection_status


def register_admin(app, *, store, analyzer, protect, public_user, json_object,
                   text_field, unique_record):
    """Register admin HTTP routes and CLI commands without creating accounts."""
    @app.get('/api/admin/overview')
    @protect(admin=True)
    def admin_overview():
        from backend.report_media import public_report
        cards = {}
        for label, path in [('image', Path(app.config['IMAGE_MODEL_PATH']).with_name('model-card.json')),
                            ('text', Path(app.config['TEXT_MODEL_PATH']) / 'model-card.json'),
                            ('video', Path(app.config['TEMPORAL_MODEL_PATH']).with_name('model-card.json')),
                            ('ai_writing', Path(app.config['AI_TEXT_MODEL_PATH']).with_name('model-card.json'))]:
            if path.is_file():
                cards[label] = json.loads(path.read_text(encoding='utf8'))
        origin_directory = app.config.get('AI_IMAGE_MODEL_PATH')
        origin_card_path = Path(origin_directory) / 'model-card.json' if origin_directory else None
        if origin_card_path and origin_card_path.is_file():
            origin_card = json.loads(origin_card_path.read_text(encoding='utf8'))
            cards['ai_origin'] = dict(origin_card, binary_classification_enabled=(
                _origin_thresholds(origin_card, 'photograph') is not None))
        return jsonify(counts={name: store.count(name) for name in ('users', 'reports', 'sources', 'datasets')},
                       models=analyzer.readiness(), model_cards=cards, database=store.kind,
                       video_detection_status=video_detection_status(app.config, analyzer.video),
                       password_reset_email=mail_ready(app.config),
                       reports=[public_report(report, app.config['REPORT_VIDEO_DIR']) for report in store.find('reports', limit=100)], datasets=store.find('datasets'),
                       sources=store.find('sources'), users=[public_user(u) for u in store.find('users')])

    @app.patch('/api/admin/users/<user_id>')
    @protect(admin=True)
    def manage_user(user_id):
        if user_id == g.user['id']:
            abort(400, description='You cannot disable your own administrator account.')
        user = store.one('users', {'id': user_id})
        if not user:
            abort(404)
        disabled = json_object().get('disabled')
        if not isinstance(disabled, bool):
            abort(400, description='disabled must be true or false.')
        user['disabled'] = disabled
        store.save('users', user)
        return jsonify(user=public_user(user))

    @app.post('/api/admin/sources')
    @protect(admin=True)
    def add_source():
        body = json_object()
        url, name = text_field(body, 'url').strip(), text_field(body, 'name').strip()
        description = text_field(body, 'description', 'Trusted source')[:200]
        try:
            parsed = urlparse(url)
            valid_url = (parsed.scheme == 'https' and parsed.hostname and not parsed.username and not parsed.password
                         and parsed.port in (None, 443) and not any(char.isspace() for char in url))
        except ValueError:
            valid_url = False
        if not valid_url or not 2 <= len(name) <= 100 or len(url) > 500:
            abort(400, description='Enter a source name and a valid HTTPS URL without credentials.')
        domain = parsed.hostname.lower().removeprefix('www.')
        if store.one('sources', {'domain': domain}):
            abort(409, description='This source domain is already registered. Enable or disable the existing source.')
        return jsonify(unique_record('sources', dict(name=name, url=url, domain=domain,
            description=description, enabled=True), 'This source domain is already registered.')), 201

    @app.patch('/api/admin/sources/<source_id>')
    @protect(admin=True)
    def toggle_source(source_id):
        source = store.one('sources', {'id': source_id})
        enabled = json_object().get('enabled')
        if not source:
            abort(404)
        if not isinstance(enabled, bool):
            abort(400)
        source['enabled'] = enabled
        return jsonify(store.save('sources', source))

    @app.post('/api/admin/datasets')
    @protect(admin=True)
    def add_dataset():
        body = json_object()
        if not all(isinstance(body.get(k), str) and 1 <= len(body[k]) <= 200 for k in ('name', 'version', 'modality')):
            abort(400, description='Dataset name, version, and modality are required.')
        counts = body.get('counts', {})
        if not isinstance(counts, dict) or len(counts) > 30 or any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in counts.values()):
            abort(400, description='Dataset counts must be nonnegative integers.')
        if body['modality'] not in ('text', 'image', 'video'):
            abort(400, description='Choose text, image, or video.')
        if store.one('datasets', {'name': body['name'], 'version': body['version']}):
            abort(409, description='This dataset version is already registered. Use a new version for updated data.')
        return jsonify(unique_record('datasets', {k: body.get(k, {}) for k in ('name', 'version', 'modality', 'counts')},
                       'This dataset version is already registered. Use a new version for updated data.')), 201

    @app.cli.command('create-admin')
    def create_admin():
        """Create an administrator interactively; no shared default password."""
        name = click.prompt('Name').strip()
        email = click.prompt('Email').strip().lower()
        password = click.prompt('Password (12-128 characters)', hide_input=True, confirmation_prompt=True)
        if (not 2 <= len(name) <= 80 or not 12 <= len(password) <= 128 or len(email) > 254
                or not re.fullmatch(r'[^\s@]{1,64}@[^\s@.]+\.[^\s@]{2,}', email)):
            raise click.ClickException('Use a name with 2-80 characters, a valid email, and a password with 12-128 characters.')
        if store.one('users', {'email': email}):
            raise click.ClickException('Email already exists. Use a separate administrator address.')
        store.save('users', dict(name=name, email=email, password_hash=generate_password_hash(password), role='admin', disabled=False))
        click.echo('Administrator created.')

    @app.cli.command('set-admin-password')
    def set_admin_password():
        """Let the local operator choose an existing admin's password privately."""
        email = click.prompt('Admin email').strip().lower()
        user = store.one('users', {'email': email})
        if not user or user.get('role') != 'admin' or user.get('disabled'):
            raise click.ClickException('No enabled administrator uses that email.')
        def validate_password(value):
            if not 12 <= len(value) <= 128:
                raise click.BadParameter('Use a password with 12-128 characters. Please try again.')
            return value

        password = click.prompt('New password (12-128 characters)', hide_input=True,
                                confirmation_prompt=True, value_proc=validate_password)
        user['password_hash'] = generate_password_hash(password)
        user['session_version'] = int(user.get('session_version', 0)) + 1
        store.save('users', user)
        click.echo('Administrator password updated. Sign in with the new password.')

