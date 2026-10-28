from contextlib import AbstractContextManager

import pytest

from backend import mailer


class FakeSMTP(AbstractContextManager):
    def __init__(self, *args, **kwargs):
        self.messages = []
        self.logins = []
        self.tls = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def starttls(self, **kwargs):
        self.tls = True

    def login(self, username, password):
        self.logins.append((username, password))

    def send_message(self, message):
        self.messages.append(message)


def test_password_reset_email_uses_tls_and_escapes_user_content(monkeypatch):
    fake = FakeSMTP()
    monkeypatch.setattr(mailer.smtplib, 'SMTP', lambda *args, **kwargs: fake)
    config = {'MAIL_HOST': 'smtp.example.test', 'MAIL_PORT': 587, 'MAIL_FROM': 'noreply@example.test',
        'MAIL_USERNAME': 'mailer', 'MAIL_PASSWORD': 'private-test-secret', 'MAIL_USE_TLS': True,
        'MAIL_USE_SSL': False, 'MAIL_TIMEOUT_SECONDS': 3}

    mailer.send_password_reset_email(config, 'reader@example.test', '<Reader>',
                                     'https://app.example.test/#reset/safe-token')

    assert fake.tls and fake.logins == [('mailer', 'private-test-secret')]
    message = fake.messages[0]
    assert message['To'] == 'reader@example.test'
    assert '<Reader>' not in message.get_body(preferencelist=('html',)).get_content()
    assert 'https://app.example.test/#reset/safe-token' in message.get_body(preferencelist=('plain',)).get_content()


def test_mailer_requires_complete_and_non_conflicting_configuration():
    assert not mailer.ready({'MAIL_HOST': 'smtp.example.test', 'MAIL_FROM': 'noreply@example.test',
        'MAIL_USERNAME': 'mailer', 'MAIL_PASSWORD': None})
    assert not mailer.ready({'MAIL_HOST': 'smtp.example.test', 'MAIL_FROM': 'noreply@example.test',
        'MAIL_USE_TLS': True, 'MAIL_USE_SSL': True})
    with pytest.raises(RuntimeError):
        mailer.send_password_reset_email({}, 'reader@example.test', 'Reader', 'https://example.test/reset')
