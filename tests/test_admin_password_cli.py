import pytest
from werkzeug.security import check_password_hash, generate_password_hash


def test_admin_can_choose_hidden_password_and_revoke_previous_sessions(app):
    store = app.extensions['store']
    admin = store.save('users', {'name': 'Admin test', 'email': 'admin@example.invalid', 'role': 'admin',
        'disabled': False, 'password_hash': generate_password_hash('previous-password-123'), 'session_version': 4})
    secret = 'new-private-password-456'
    response = app.test_cli_runner().invoke(args=['set-admin-password'], input='admin@example.invalid\n' + secret + '\n' + secret + '\n')
    assert response.exit_code == 0
    assert secret not in response.output
    assert 'password updated' in response.output
    saved = store.one('users', {'id': admin['id']})
    assert check_password_hash(saved['password_hash'], secret)
    assert saved['session_version'] == 5
    assert saved['role'] == 'admin'


def test_admin_password_cli_does_not_promote_or_reset_a_regular_user(app):
    store = app.extensions['store']
    user = store.save('users', {'name': 'Regular user', 'email': 'user@example.invalid', 'role': 'user',
        'disabled': False, 'password_hash': generate_password_hash('existing-password-123')})
    response = app.test_cli_runner().invoke(args=['set-admin-password'], input='user@example.invalid\n')
    assert response.exit_code != 0
    assert store.one('users', {'id': user['id']}) == user


@pytest.mark.parametrize('invalid_secret', ['tiny123', 'x' * 129])
def test_admin_password_cli_retries_invalid_length_without_saving_it(app, invalid_secret):
    store = app.extensions['store']
    admin = store.save('users', {'name': 'Admin test', 'email': 'admin@example.invalid', 'role': 'admin',
        'disabled': False, 'password_hash': generate_password_hash('previous-password-123'), 'session_version': 4})
    secret = 'private-1234'
    response = app.test_cli_runner().invoke(args=['set-admin-password'],
        input='admin@example.invalid\n' + invalid_secret + '\n' + secret + '\n' + secret + '\n')
    assert response.exit_code == 0
    assert invalid_secret not in response.output
    assert secret not in response.output
    assert 'password updated' in response.output
    saved = store.one('users', {'id': admin['id']})
    assert check_password_hash(saved['password_hash'], secret)
    assert not check_password_hash(saved['password_hash'], invalid_secret)
    assert saved['session_version'] == 5


def test_admin_password_cli_retries_mismatched_confirmation_privately(app):
    store = app.extensions['store']
    admin = store.save('users', {'name': 'Admin test', 'email': 'admin@example.invalid', 'role': 'admin',
        'disabled': False, 'password_hash': generate_password_hash('previous-password-123'), 'session_version': 4})
    first_secret = 'initial-private-password-123'
    mismatch = 'different-private-password-456'
    secret = 'final-private-password-789'
    response = app.test_cli_runner().invoke(args=['set-admin-password'],
        input='admin@example.invalid\n' + first_secret + '\n' + mismatch + '\n' + secret + '\n' + secret + '\n')
    assert response.exit_code == 0
    assert all(value not in response.output for value in (first_secret, mismatch, secret))
    assert 'password updated' in response.output
    saved = store.one('users', {'id': admin['id']})
    assert check_password_hash(saved['password_hash'], secret)
    assert not check_password_hash(saved['password_hash'], first_secret)
    assert saved['session_version'] == 5
