"""Shared credential-free startup diagnostics and bounded Waitress settings."""
import ipaddress
import os
from pathlib import Path
import sys


def server_options(environ=None):
    environ = os.environ if environ is None else environ

    def number(name, default, maximum):
        try:
            value = int(environ.get(name, default))
        except (TypeError, ValueError):
            raise ValueError(name + ' must be an integer.') from None
        if not 1 <= value <= maximum:
            raise ValueError(name + ' is outside the supported range.')
        return value

    host = environ.get('HOST', '127.0.0.1')
    try:
        ipaddress.ip_address(host)
    except ValueError:
        raise ValueError('HOST must be an IPv4 or IPv6 bind address.') from None
    options = {'host': host, 'port': number('PORT', 5000, 65535),
               'threads': number('WAITRESS_THREADS', 6, 64),
               'max_request_body_size': 50 * 1024 * 1024,
               'clear_untrusted_proxy_headers': True, 'expose_tracebacks': False}
    proxy = environ.get('WAITRESS_TRUSTED_PROXY')
    if proxy:
        if proxy == '*':
            if environ.get('RENDER') != 'true':
                raise ValueError('Wildcard proxy trust is restricted to Render ingress.')
        else:
            try:
                ipaddress.ip_address(proxy)
            except ValueError:
                raise ValueError('WAITRESS_TRUSTED_PROXY must identify one proxy IP address.') from None
        options.update(trusted_proxy=proxy,
                       trusted_proxy_count=number('WAITRESS_TRUSTED_PROXY_COUNT', 1, 4),
                       trusted_proxy_headers={'x-forwarded-for', 'x-forwarded-proto'})
    elif environ.get('WAITRESS_TRUSTED_PROXY_COUNT'):
        raise ValueError('WAITRESS_TRUSTED_PROXY_COUNT requires WAITRESS_TRUSTED_PROXY.')
    return options


def container_main(argv=None):
    """Prepare the runtime mount, then exec the container command without root."""
    command = list(sys.argv[1:] if argv is None else argv)
    if not command:
        raise ValueError('A container command is required.')
    if os.name == 'posix' and os.geteuid() == 0:
        from pwd import getpwnam
        account = getpwnam('app')
        # Render mounts disks only at runtime, replacing image-layer ownership.
        # Limit privileged filesystem changes to our documented data mount.
        directory = Path(os.environ.get('DATA_DIR', '/var/data'))
        mount = Path('/var/data')
        if (not directory.is_absolute()
                or any(path.is_symlink() for path in (directory, *directory.parents))
                or not directory.resolve().is_relative_to(mount)):
            raise ValueError('Container DATA_DIR must be inside /var/data without symlinks.')
        directory.mkdir(parents=True, exist_ok=True)
        os.chown(directory, account.pw_uid, account.pw_gid)
        os.setgroups([])
        os.setgid(account.pw_gid)
        os.setuid(account.pw_uid)
        os.environ['HOME'] = account.pw_dir
    os.execvp(command[0], command)


def startup_failure(error):
    """Do not print raw database/provider exception text, URLs, or credentials."""
    name = type(error).__name__
    if name in ('OperationalError', 'InterfaceError', 'DBAPIError', 'DatabaseError'):
        return 'MySQL is unavailable. Check the server, database, account permissions, and TLS settings. Run python -m scripts.check_database.'
    if isinstance(error, ImportError):
        return 'A runtime dependency is missing. Install the documented requirements in this interpreter; run python -m scripts.check_setup.'
    if isinstance(error, OSError):
        return 'The server could not open its address or runtime files. Check the port, file permissions, and whether another server is already running.'
    return 'Application startup failed (' + name + '). Run python -m scripts.check_setup and follow README.md; configuration values are not displayed.'


if __name__ == '__main__':
    try:
        container_main()
    except Exception as error:
        print(startup_failure(error), file=sys.stderr)
        raise SystemExit(1) from None
