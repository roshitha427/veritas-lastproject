"""Check the configured application without changing users, reports, or models."""
import json

from scripts.startup import startup_failure


def close_checked_app(app):
    for name in ('analysis_executor', 'mail_executor'):
        executor = app.extensions.get(name)
        if executor is not None:
            executor.shutdown(wait=True)
    store = app.extensions.get('store')
    if store is not None:
        store.close()


def main():
    app = None
    try:
        from backend.app import create_app
        app = create_app()
        app.extensions['store'].ping()
        print(json.dumps({'database': 'mysql', 'connected': True,
                          'models': app.extensions['analyzer'].readiness()}, indent=2))
        return 0
    except Exception as error:
        print(startup_failure(error))
        return 1
    finally:
        if app is not None:
            close_checked_app(app)


if __name__ == '__main__':
    raise SystemExit(main())
