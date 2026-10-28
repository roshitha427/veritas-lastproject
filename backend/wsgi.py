"""WSGI factory for an installed app; importing this module starts no workers."""


def create_application():
    from backend.app import create_server_app
    return create_server_app()
