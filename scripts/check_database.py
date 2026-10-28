"""Initialize and verify the configured MySQL store without printing credentials."""
from scripts.check_setup import read_environment


def main():
    store = None
    try:
        from backend.store import Store
        config = read_environment()
        config.setdefault('DATABASE_BACKEND', 'mysql')
        if not config.get('DATABASE_URL') and config.get('MYSQL_PASSWORD') is None:
            print('Set DATABASE_URL or MYSQL_HOST, MYSQL_DATABASE, MYSQL_USER and MYSQL_PASSWORD in backend/.env or the environment.')
            return 1
        store = Store(config)
        store.ping()
        print('MySQL connection and application schema are ready.')
        return 0
    except Exception as error:
        print('MySQL check failed (' + type(error).__name__ + '). Check database availability, credentials, schema permissions and TLS configuration. Configuration values are not displayed.')
        return 1
    finally:
        if store is not None:
            store.close()


if __name__ == '__main__':
    raise SystemExit(main())
