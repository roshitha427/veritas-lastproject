import json
import sys
from types import SimpleNamespace

import pytest

from scripts.check_setup import inspect_setup
from scripts import startup


def by_name(checks):
    return {check['name']: check for check in checks}


def test_setup_requires_mysql_and_never_echoes_credentials(tmp_path):
    config = {'DATABASE_BACKEND': 'sqlite',
              'DATABASE_URL': 'mysql+pymysql://private-user:private-password@private-host/veritas'}
    checks = inspect_setup(tmp_path, config, available=lambda name: True)
    assert by_name(checks)['database_backend']['status'] == 'error'
    assert by_name(checks)['database_configuration']['status'] == 'ok'
    assert 'private-' not in json.dumps(checks)


def test_setup_checks_production_security_without_connecting(tmp_path):
    config = {'APP_ENV': 'production', 'MYSQL_DATABASE': 'veritas', 'MYSQL_USER': 'veritas',
              'MYSQL_PASSWORD': 'a-secret-value', 'PUBLIC_BASE_URL': 'http://example.test'}
    checks = by_name(inspect_setup(tmp_path, config, available=lambda name: True))
    assert checks['database_configuration']['status'] == 'ok'
    assert checks['session_secret']['status'] == 'error'
    assert checks['secure_cookie']['status'] == 'error'
    assert checks['public_url']['status'] == 'error'
    assert checks['frontend_build']['status'] == 'error'


def test_optional_inference_dependency_warning_does_not_block_setup(tmp_path):
    checks = by_name(inspect_setup(tmp_path, {'DATABASE_URL': 'mysql+pymysql://user:pass@localhost/veritas'},
                                   available=lambda name: name not in ('torch', 'transformers')))
    assert checks['runtime_dependencies']['status'] == 'ok'
    assert checks['inference_dependencies']['status'] == 'warning'
    assert checks['frontend_build']['status'] == 'warning'


def forwarded_request(config, forwarded='198.51.100.99, 203.0.113.5'):
    from waitress.proxy_headers import proxy_headers_middleware
    options = startup.server_options(config)
    captured = {}

    def app(environ, start_response):
        captured.update(environ)
        start_response('200 OK', [])
        return [b'ok']

    wrapped = proxy_headers_middleware(app,
        trusted_proxy=options.get('trusted_proxy'),
        trusted_proxy_count=options.get('trusted_proxy_count', 1),
        trusted_proxy_headers=options.get('trusted_proxy_headers'), clear_untrusted=True)
    wrapped({'REMOTE_ADDR': '10.0.0.2', 'HTTP_X_FORWARDED_FOR': forwarded,
             'HTTP_X_FORWARDED_PROTO': 'https', 'wsgi.url_scheme': 'http'}, lambda *args: None)
    return captured


def test_direct_server_ignores_forged_forwarded_headers():
    request = forwarded_request({})
    assert request['REMOTE_ADDR'] == '10.0.0.2'
    assert request['wsgi.url_scheme'] == 'http'
    assert 'HTTP_X_FORWARDED_FOR' not in request


def test_render_ingress_uses_trusted_address_and_ignores_forged_prefix():
    config = {'RENDER': 'true', 'WAITRESS_TRUSTED_PROXY': '*'}
    first = forwarded_request(config)
    second = forwarded_request(config, '192.0.2.99, 203.0.113.6')
    assert first['REMOTE_ADDR'] == '203.0.113.5'
    assert second['REMOTE_ADDR'] == '203.0.113.6'
    assert first['wsgi.url_scheme'] == 'https'


@pytest.mark.parametrize('config', [
    {'WAITRESS_TRUSTED_PROXY': '*'},
    {'WAITRESS_TRUSTED_PROXY': 'not-an-ip'},
    {'WAITRESS_TRUSTED_PROXY_COUNT': '2'},
    {'WAITRESS_TRUSTED_PROXY': '10.0.0.2', 'WAITRESS_TRUSTED_PROXY_COUNT': '0'},
    {'WAITRESS_TRUSTED_PROXY': '10.0.0.2', 'WAITRESS_TRUSTED_PROXY_COUNT': '5'},
    {'WAITRESS_TRUSTED_PROXY': '10.0.0.2', 'WAITRESS_TRUSTED_PROXY_COUNT': 'invalid'},
])
def test_invalid_or_unprotected_proxy_trust_is_rejected(config):
    with pytest.raises(ValueError):
        startup.server_options(config)


def test_known_proxy_count_does_not_trust_earlier_prefixes():
    request = forwarded_request({'WAITRESS_TRUSTED_PROXY': '10.0.0.2',
                                 'WAITRESS_TRUSTED_PROXY_COUNT': '2'},
                                '192.0.2.99, 203.0.113.7, 10.0.0.1')
    assert request['REMOTE_ADDR'] == '203.0.113.7'


def test_container_drops_privileges_before_executing_server(tmp_path, monkeypatch):
    events = []
    account = SimpleNamespace(pw_uid=10001, pw_gid=10001, pw_dir='/home/app')
    directory = tmp_path / 'var' / 'data'
    directory.mkdir(parents=True)
    monkeypatch.setitem(sys.modules, 'pwd', SimpleNamespace(getpwnam=lambda name: account))
    monkeypatch.setattr(startup, 'Path', lambda value: directory if value == '/var/data' else __import__('pathlib').Path(value))
    monkeypatch.setattr(startup, 'os', SimpleNamespace(name='posix', geteuid=lambda: 0,
        environ={'DATA_DIR': '/var/data'},
        chown=lambda *args: events.append(('chown', args)),
        setgroups=lambda groups: events.append(('groups', groups)),
        setgid=lambda gid: events.append(('gid', gid)),
        setuid=lambda uid: events.append(('uid', uid)),
        execvp=lambda *args: events.append(('exec', args))))
    startup.container_main(['python', '-m', 'scripts.serve'])
    assert [event[0] for event in events] == ['chown', 'groups', 'gid', 'uid', 'exec']
    assert events[0][1] == (directory, 10001, 10001)
    assert startup.os.environ['HOME'] == '/home/app'


@pytest.mark.parametrize('value', ['/etc', 'relative', '/var/data/../../etc'])
def test_container_refuses_privileged_changes_outside_data_mount(value, monkeypatch):
    monkeypatch.setitem(sys.modules, 'pwd', SimpleNamespace(getpwnam=lambda name: SimpleNamespace()))
    monkeypatch.setattr(startup, 'os', SimpleNamespace(name='posix', geteuid=lambda: 0,
                                                   environ={'DATA_DIR': value}))
    with pytest.raises(ValueError, match='DATA_DIR'):
        startup.container_main(['python', '-m', 'scripts.serve'])


def test_server_handles_container_termination_and_closes_resources(monkeypatch):
    import signal
    from scripts import check_system, serve
    app = object()
    closed = []
    original_handler = signal.getsignal(signal.SIGTERM)
    monkeypatch.setitem(sys.modules, 'backend.app', SimpleNamespace(create_server_app=lambda: app))
    monkeypatch.setitem(sys.modules, 'waitress', SimpleNamespace(
        serve=lambda *args, **kwargs: signal.raise_signal(signal.SIGTERM)))
    monkeypatch.setattr(check_system, 'close_checked_app', lambda value: closed.append(value))
    assert serve.main() == 0
    assert closed == [app]
    assert signal.getsignal(signal.SIGTERM) == original_handler
