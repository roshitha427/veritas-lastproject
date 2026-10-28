"""SMTP delivery for account recovery messages."""
from email.message import EmailMessage
import html
import smtplib
import ssl


def ready(config):
    host = config.get('MAIL_HOST')
    sender = config.get('MAIL_FROM')
    username = config.get('MAIL_USERNAME')
    password = config.get('MAIL_PASSWORD')
    return bool(host and sender and (bool(username) == bool(password))
                and not (config.get('MAIL_USE_SSL') and config.get('MAIL_USE_TLS')))


def send_password_reset_email(config, recipient, name, reset_url):
    """Send one reset link using configured SMTP; never print the URL or credentials."""
    if not ready(config):
        raise RuntimeError('Password reset email is not configured.')

    safe_name = ''.join(char for char in name if char not in '\r\n')[:80].strip() or 'there'
    message = EmailMessage()
    message['Subject'] = 'Reset your password'
    message['From'] = config['MAIL_FROM']
    message['To'] = recipient
    message.set_content(
        f"Hello {safe_name},\n\n"
        "We received a request to reset your password. Use this link within the next hour:\n\n"
        f"{reset_url}\n\n"
        "If you did not request this change, you can ignore this email. Your current password will remain unchanged.\n"
    )
    message.add_alternative(
        '<p>Hello ' + html.escape(safe_name) + ',</p>'
        '<p>We received a request to reset your password. Use this link within the next hour:</p>'
        '<p><a href="' + html.escape(reset_url, quote=True) + '">Choose a new password</a></p>'
        '<p>If you did not request this change, you can ignore this email. Your current password will remain unchanged.</p>',
        subtype='html',
    )

    host = config['MAIL_HOST']
    port = int(config.get('MAIL_PORT', 587))
    timeout = float(config.get('MAIL_TIMEOUT_SECONDS', 12))
    if config.get('MAIL_USE_SSL'):
        with smtplib.SMTP_SSL(host, port, timeout=timeout, context=ssl.create_default_context()) as server:
            _deliver(server, config, message)
    else:
        with smtplib.SMTP(host, port, timeout=timeout) as server:
            if config.get('MAIL_USE_TLS'):
                server.starttls(context=ssl.create_default_context())
            _deliver(server, config, message)


def _deliver(server, config, message):
    username = config.get('MAIL_USERNAME')
    if username:
        server.login(username, config['MAIL_PASSWORD'])
    server.send_message(message)
