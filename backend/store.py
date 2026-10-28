"""MySQL 8 persistence with indexed fields and SR-compatible JSON documents."""
from datetime import date, datetime, timezone
import json
import ssl
from uuid import uuid4

from sqlalchemy import (Column, DateTime, Index, Integer, JSON, MetaData, String,
                        Table, UniqueConstraint, create_engine, delete, func,
                        insert, or_, select, text, update)
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import ArgumentError, IntegrityError
from sqlalchemy.dialects.mysql import DATETIME

SCHEMA_VERSION = 1
COLLECTIONS = ('users', 'reports', 'sources', 'datasets', 'password_resets')


def now():
    return datetime.now(timezone.utc).isoformat()


def _json_default(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(f'Cannot serialize {type(value).__name__}')


def database_url(config):
    """Parse connection settings without putting credentials in SQL or logs."""
    try:
        if config.get('DATABASE_URL'):
            url = make_url(config['DATABASE_URL'])
            if url.drivername not in ('mysql', 'mysql+pymysql'):
                raise ValueError
            url = url.set(drivername='mysql+pymysql')
        else:
            url = URL.create('mysql+pymysql', username=config.get('MYSQL_USER'),
                password=config.get('MYSQL_PASSWORD'),
                host=config.get('MYSQL_HOST', '127.0.0.1'),
                port=int(config.get('MYSQL_PORT', 3306)),
                database=config.get('MYSQL_DATABASE'))
        if not url.username or not url.database or not url.host:
            raise ValueError
        if url.port is not None and not 1 <= url.port <= 65535:
            raise ValueError
    except (ArgumentError, TypeError, ValueError):
        raise RuntimeError('Set a valid MySQL DATABASE_URL or MYSQL_HOST, MYSQL_PORT, MYSQL_DATABASE and MYSQL_USER.') from None
    return url


def _metadata():
    metadata = MetaData()
    tables = {}
    extras = {
        'users': [Column('email', String(254)), UniqueConstraint('email', name='uq_users_email')],
        'reports': [Column('owner_id', String(128)), Column('status', String(32)),
                    Index('ix_reports_owner_created', 'owner_id', 'created_at'),
                    Index('ix_reports_status', 'status')],
        'sources': [Column('domain', String(253)), UniqueConstraint('domain', name='uq_sources_domain')],
        'datasets': [Column('name', String(200)), Column('version', String(200)),
                     UniqueConstraint('name', 'version', name='uq_datasets_name_version')],
        'password_resets': [Column('user_id', String(128)), Column('expires_at', DATETIME(fsp=6)),
                            Index('ix_password_resets_user', 'user_id'),
                            Index('ix_password_resets_expiry', 'expires_at')],
    }
    for name in COLLECTIONS:
        tables[name] = Table(name, metadata,
            Column('id', String(128), primary_key=True),
            Column('created_at', DATETIME(fsp=6), nullable=False),
            Column('body', JSON, nullable=False), *extras[name],
            mysql_engine='InnoDB', mysql_charset='utf8mb4', mysql_collate='utf8mb4_bin')
    versions = Table('app_schema_version', metadata,
        Column('version', Integer, primary_key=True),
        Column('installed_at', DATETIME(fsp=6), nullable=False),
        mysql_engine='InnoDB', mysql_charset='utf8mb4', mysql_collate='utf8mb4_bin')
    return metadata, tables, versions


def _timestamp(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace('Z', '+00:00'))
    elif isinstance(value, date) and not isinstance(value, datetime):
        value = datetime.combine(value, datetime.min.time())
    if not isinstance(value, datetime):
        raise ValueError('Timestamps must be ISO 8601 strings or datetime values.')
    if value.tzinfo:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


class Store:
    kind = 'mysql'

    def __init__(self, config):
        if config.get('DATABASE_BACKEND', 'mysql') != 'mysql':
            raise RuntimeError('DATABASE_BACKEND must be mysql. No alternate database adapter is configured.')
        url = database_url(config)
        try:
            pool_size = int(config.get('MYSQL_POOL_SIZE', 5))
            if not 1 <= pool_size <= 50:
                raise ValueError
        except (TypeError, ValueError):
            raise RuntimeError('MYSQL_POOL_SIZE must be an integer between 1 and 50.') from None
        connect_args = {'charset': 'utf8mb4', 'connect_timeout': 10,
                        'read_timeout': 30, 'write_timeout': 30}
        if config.get('MYSQL_SSL_CA'):
            connect_args['ssl'] = ssl.create_default_context(cafile=config['MYSQL_SSL_CA'])
        self.engine = create_engine(url, connect_args=connect_args,
            pool_pre_ping=True, pool_recycle=1800, pool_size=pool_size,
            max_overflow=2, pool_timeout=15, isolation_level='READ COMMITTED',
            json_serializer=lambda value: json.dumps(value, default=_json_default),
            hide_parameters=True)
        self.metadata, self.tables, self.versions = _metadata()
        try:
            self._initialize_schema()
        except Exception:
            self.engine.dispose()
            raise

    def _initialize_schema(self):
        with self.engine.connect() as connection:
            version = connection.execute(text('SELECT VERSION()')).scalar_one()
            if 'mariadb' in version.lower() or int(version.split('.', 1)[0]) < 8:
                raise RuntimeError('MySQL 8 or later is required.')
            lock_name = ('veritas_schema_' + self.engine.url.database)[:64]
            acquired = connection.execute(text('SELECT GET_LOCK(:name, 30)'), {'name': lock_name}).scalar_one()
            if acquired != 1:
                raise RuntimeError('Could not acquire the MySQL schema initialization lock.')
            try:
                self.metadata.create_all(connection)
                installed = connection.execute(select(func.max(self.versions.c.version))).scalar_one()
                if installed is not None and installed != SCHEMA_VERSION:
                    raise RuntimeError('Unsupported database schema version; apply the documented migration before starting.')
                if installed is None:
                    connection.execute(insert(self.versions).values(version=SCHEMA_VERSION,
                        installed_at=datetime.now(timezone.utc).replace(tzinfo=None)))
                connection.commit()
            finally:
                connection.execute(text('SELECT RELEASE_LOCK(:name)'), {'name': lock_name})
                connection.commit()

    def _table(self, collection):
        try:
            return self.tables[collection]
        except (KeyError, TypeError):
            raise ValueError('Unknown collection.') from None

    def save(self, collection, doc):
        table = self._table(collection)
        document = dict(doc)
        document.setdefault('id', uuid4().hex)
        document.setdefault('created_at', now())
        document = json.loads(json.dumps(document, default=_json_default))
        if not isinstance(document['id'], str) or not 1 <= len(document['id']) <= 128:
            raise ValueError('Document id must contain 1-128 characters.')
        values = {'id': document['id'], 'created_at': _timestamp(document['created_at']), 'body': document}
        for column in table.columns:
            if column.name not in values:
                value = document.get(column.name)
                values[column.name] = _timestamp(value) if value is not None and isinstance(column.type, DateTime) else value
        for attempt in range(2):
            try:
                with self.engine.begin() as connection:
                    changed = connection.execute(update(table).where(table.c.id == document['id']).values(**values))
                    if changed.rowcount == 0:
                        connection.execute(insert(table).values(**values))
                return document
            except IntegrityError as error:
                # A race on this same primary key can retry. Never overwrite an
                # unrelated account/source/dataset with a colliding unique key.
                if attempt or getattr(error.orig, 'args', (None,))[0] != 1062:
                    raise
                with self.engine.connect() as connection:
                    exists = connection.execute(select(table.c.id).where(table.c.id == document['id'])).scalar_one_or_none()
                if exists is None:
                    raise

    def find(self, collection, query=None, limit=500):
        table = self._table(collection)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError('limit must be a nonnegative integer.')
        conditions = []
        json_filter = False
        for key, value in (query or {}).items():
            if not isinstance(key, str):
                raise ValueError('Query fields must be strings.')
            if key in table.c and key != 'body':
                column = table.c[key]
                if isinstance(column.type, DateTime) and value is not None:
                    value = _timestamp(value)
                conditions.append(column == value)
            else:
                json_filter = True
                field = table.c.body[key]
                if value is None:
                    conditions.append(or_(func.json_contains_path(table.c.body, 'one', '$.' + json.dumps(key)) == 0,
                                          func.json_type(field) == 'NULL'))
                elif isinstance(value, bool):
                    conditions.append(field.as_boolean() == value)
                elif isinstance(value, int):
                    conditions.append(field.as_integer() == value)
                elif isinstance(value, float):
                    conditions.append(field.as_float() == value)
                elif isinstance(value, str):
                    conditions.append(field.as_string() == value)
                else:
                    conditions.append(field == value)
        # Sort only narrow keys. Including large JSON reports in a filesort can
        # exhaust MySQL's sort buffer even when the result limit is very small.
        if json_filter:
            # DISTINCT prevents derived-table merging: otherwise MySQL may
            # carry the JSON predicate's payload into its sort as well.
            matching = select(table.c.id, table.c.created_at).where(*conditions).distinct().subquery()
            statement = select(matching.c.id).order_by(matching.c.created_at.desc(), matching.c.id.desc()).limit(limit)
        else:
            statement = select(table.c.id).where(*conditions).order_by(table.c.created_at.desc(), table.c.id.desc()).limit(limit)
        with self.engine.connect() as connection:
            identifiers = list(connection.execute(statement).scalars())
            if not identifiers:
                return []
            bodies = select(table.c.id, table.c.body).where(table.c.id.in_(identifiers), *conditions)
            documents = dict(connection.execute(bodies).all())
            # Recheck filters on the second read and tolerate concurrent deletes.
            return [documents[identifier] for identifier in identifiers if identifier in documents]

    def one(self, collection, query):
        records = self.find(collection, query, limit=1)
        return records[0] if records else None

    def delete(self, collection, doc_id):
        table = self._table(collection)
        with self.engine.begin() as connection:
            connection.execute(delete(table).where(table.c.id == doc_id))

    def take(self, collection, doc_id):
        """Atomically consume one record across threads and processes."""
        table = self._table(collection)
        with self.engine.begin() as connection:
            document = connection.execute(select(table.c.body).where(table.c.id == doc_id).with_for_update()).scalar_one_or_none()
            if document is not None:
                connection.execute(delete(table).where(table.c.id == doc_id))
            return document

    def ping(self):
        with self.engine.connect() as connection:
            connection.execute(text('SELECT 1')).scalar_one()

    def count(self, collection):
        table = self._table(collection)
        with self.engine.connect() as connection:
            return connection.execute(select(func.count()).select_from(table)).scalar_one()

    def close(self):
        self.engine.dispose()
