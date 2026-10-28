"""Integration checks against TEST_DATABASE_URL, a disposable MySQL 8 database."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Barrier

import pytest
from sqlalchemy import event, inspect, select
from sqlalchemy.exc import IntegrityError

from backend.store import Store, database_url


@pytest.fixture
def store(clean_mysql):
    instance = Store(clean_mysql)
    yield instance
    instance.close()


def test_configuration_rejects_alternate_databases_and_hides_bad_url():
    for backend in ('sqlite', 'mongo'):
        with pytest.raises(RuntimeError, match='must be mysql'):
            Store({'DATABASE_BACKEND': backend})
    for url in ('sqlite:///private-password', 'mysql+pymysql://user:private-password@host/', 'not a URL'):
        with pytest.raises(RuntimeError) as error:
            database_url({'DATABASE_URL': url})
        assert 'private-password' not in str(error.value)
    url = database_url({'MYSQL_USER': 'app', 'MYSQL_PASSWORD': 'p@ss:/word',
                        'MYSQL_HOST': '127.0.0.1', 'MYSQL_DATABASE': 'sr_test'})
    assert url.password == 'p@ss:/word'
    assert database_url({'DATABASE_URL': 'mysql://app@localhost/sr_test'}).drivername == 'mysql+pymysql'


def test_crud_preserves_nested_json_and_serializes_dates(store):
    saved = store.save('reports', {'owner_id': 'owner', 'status': 'complete',
        'result': {'verdict': None, 'probabilities': [0.1, 0.9], 'sources': [{'title': 'తెలుగు'}]},
        'completed_at': datetime(2026, 10, 28, tzinfo=timezone.utc)})
    assert store.kind == 'mysql'
    assert len(saved['id']) == 32
    assert saved['completed_at'] == '2026-10-28T00:00:00+00:00'
    assert store.one('reports', {'id': saved['id']}) == saved
    saved['status'] = 'reviewed'
    saved['result']['sources'].append({'title': 'हिन्दी'})
    store.save('reports', saved)
    assert store.count('reports') == 1
    assert store.one('reports', {'id': saved['id']}) == saved
    assert store.one('reports', {'status': 'complete'}) is None
    store.delete('reports', saved['id'])
    store.delete('reports', saved['id'])
    assert store.count('reports') == 0


@pytest.mark.parametrize(('collection', 'identity', 'different'), [
    ('users', {'email': 'reader@example.com'}, {'email': 'other@example.com'}),
    ('sources', {'domain': 'example.com'}, {'domain': 'other.example.com'}),
    ('datasets', {'name': 'Training', 'version': '1'}, {'name': 'Training', 'version': '2'}),
])
def test_unique_keys_never_overwrite_an_unrelated_document(store, collection, identity, different):
    original = store.save(collection, {**identity, 'name_note': 'keep'})
    with pytest.raises(IntegrityError):
        store.save(collection, {**identity, 'name_note': 'must not overwrite'})
    assert store.find(collection) == [original]
    other = store.save(collection, {**different, 'name_note': 'second'})
    with pytest.raises(IntegrityError):
        store.save(collection, {**other, **identity})
    assert store.one(collection, {'id': original['id']}) == original
    assert store.one(collection, {'id': other['id']}) == other


def test_filters_ordering_and_limits_execute_in_mysql(store):
    first = store.save('reports', {'owner_id': 'one', 'status': 'complete',
        'created_at': '2026-10-28T09:00:00+00:00', 'flag': True, 'number': 4, 'optional': None})
    second = store.save('reports', {'owner_id': 'one', 'status': 'queued',
        'created_at': '2026-10-28T09:01:00+00:00', 'flag': False, 'number': 7})
    store.save('reports', {'owner_id': 'two', 'status': 'complete',
        'created_at': '2026-10-28T09:02:00+00:00', 'flag': True, 'number': 9, 'optional': 'present'})
    assert store.find('reports', {'owner_id': 'one'}) == [second, first]
    assert store.find('reports', {'owner_id': 'one', 'status': 'complete'}) == [first]
    assert store.find('reports', {'flag': False}) == [second]
    assert store.find('reports', {'number': 4}) == [first]
    assert store.find('reports', {'optional': None}) == [second, first]
    assert store.find('reports', {'owner_id': 'one'}, limit=1) == [second]
    assert store.find('reports', limit=0) == []
    assert store.find('reports', {'owner_id': "one' OR 1=1 --"}) == []
    for limit in (-1, True, '3'):
        with pytest.raises(ValueError):
            store.find('reports', limit=limit)


def test_created_timestamps_retain_subsecond_order(store):
    older = store.save('reports', {'id': 'z', 'created_at': '2026-10-28T09:00:00.100000+00:00'})
    newer = store.save('reports', {'id': 'a', 'created_at': '2026-10-28T09:00:00.200000+00:00'})
    assert store.find('reports') == [newer, older]


def test_large_report_json_does_not_exhaust_small_sort_buffer(store):
    def small_sort_buffer(connection, record, proxy):
        with connection.cursor() as cursor:
            cursor.execute('SET SESSION sort_buffer_size = 32768')
    event.listen(store.engine, 'checkout', small_sort_buffer)
    try:
        payload = {'sources': [{'excerpt': 'x' * (512 * 1024)}], 'preview': {'bytes': 'y' * 32768}}
        older = store.save('reports', {'id': 'large-a', 'owner_id': 'one', 'selected': True,
            'created_at': '2026-10-28T09:00:00.100000+00:00', 'result': payload})
        newer = store.save('reports', {'id': 'large-b', 'owner_id': 'one', 'selected': True,
            'created_at': '2026-10-28T09:00:00.200000+00:00', 'result': payload})
        tied = store.save('reports', {'id': 'large-z', 'owner_id': 'two', 'selected': False,
            'created_at': '2026-10-28T09:00:00.200000+00:00', 'result': payload})
        assert store.find('reports', limit=1) == [tied]
        assert store.find('reports', limit=100) == [tied, newer, older]
        assert store.find('reports', {'selected': True}, limit=1) == [newer]
        assert store.find('reports', {'owner_id': 'one'}) == [newer, older]
        assert store.one('reports', {'id': 'large-a'}) == older
        assert store.find('reports', {'owner_id': 'missing'}) == []
        assert store.find('reports', limit=0) == []
    finally:
        event.remove(store.engine, 'checkout', small_sort_buffer)


def test_take_is_atomic_between_separate_pools(store, clean_mysql):
    other = Store(clean_mysql)
    token = store.save('password_resets', {'id': 'single-use-token', 'user_id': 'reader',
        'expires_at': datetime.now(timezone.utc) + timedelta(minutes=10)})
    barrier = Barrier(8)
    def consume(index):
        barrier.wait()
        return (store if index % 2 else other).take('password_resets', token['id'])
    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(consume, range(8)))
        assert results.count(token) == 1
        assert results.count(None) == 7
        assert store.count('password_resets') == 0
    finally:
        other.close()


def test_connections_are_thread_safe_and_schema_initialization_is_idempotent(store, clean_mysql):
    def save(index):
        document = store.save('users', {'email': f'reader-{index}@example.com', 'disabled': False})
        return store.one('users', {'id': document['id']})
    with ThreadPoolExecutor(max_workers=8) as executor:
        saved = list(executor.map(save, range(20)))
    assert len({item['id'] for item in saved}) == 20
    assert store.count('users') == 20
    other = Store(clean_mysql)
    try:
        other.ping()
        assert other.count('users') == 20
        with other.engine.connect() as connection:
            assert connection.execute(select(other.versions.c.version)).scalars().all() == [1]
        indexes = inspect(other.engine).get_indexes('reports')
        assert {'ix_reports_owner_created', 'ix_reports_status'} <= {row['name'] for row in indexes}
    finally:
        other.close()


def test_unknown_collections_are_rejected_before_sql_execution(store):
    for collection in ('users; DROP TABLE reports', 'app_schema_version', ''):
        with pytest.raises(ValueError, match='Unknown collection'):
            store.find(collection)
        with pytest.raises(ValueError, match='Unknown collection'):
            store.save(collection, {})

