"""Start one Waitress process with safe startup diagnostics."""
import signal

from scripts.startup import server_options, startup_failure


def _stop_server(signum, frame):
    raise KeyboardInterrupt


def main():
    app = None
    previous_handler = signal.signal(signal.SIGTERM, _stop_server)
    try:
        from backend.app import create_server_app
        from scripts.check_system import close_checked_app
        from waitress import serve
        app = create_server_app()
        options = server_options()
        print(f"Verification workspace: http://{options['host']}:{options['port']}")
        print('One process; bounded analysis queue. Model readiness is reported by /api/health, not an accuracy guarantee.')
        serve(app, **options)
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as error:
        print(startup_failure(error))
        return 1
    finally:
        try:
            if app is not None:
                close_checked_app(app)
        finally:
            signal.signal(signal.SIGTERM, previous_handler)


if __name__ == '__main__':
    raise SystemExit(main())
