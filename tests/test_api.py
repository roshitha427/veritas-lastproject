from io import BytesIO
import json
import time
import pytest
from PIL import Image
from backend.store import now
from tests.conftest import register, token


def wait_report(client, report_id):
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        response = client.get('/api/reports/' + report_id)
        if response.json['status'] in ('complete', 'failed'):
            return response.json
        time.sleep(.05)
    raise AssertionError('Analysis did not finish')


def test_auth_csrf_duplicate_email_and_disabled_session(client, app):
    assert client.post('/api/auth/register', json={}).status_code == 403
    auth = register(client)
    duplicate = client.post('/api/auth/register', json={'name':'Other', 'email':'reader@example.com', 'password':'other-long-password'},
                            headers={'X-CSRF-Token':auth['csrf_token']})
    assert duplicate.status_code == 409
    assert app.extensions['store'].count('users') == 1
    assert client.get('/api/session').json['user']['role'] == 'user'
    user = app.extensions['store'].one('users', {'id':auth['user']['id']})
    user['disabled'] = True
    app.extensions['store'].save('users', user)
    assert client.get('/api/reports').status_code == 401


def test_logout_rotates_csrf_and_allows_sign_in_again(client):
    auth = register(client)
    logout = client.post('/api/auth/logout', headers={'X-CSRF-Token': auth['csrf_token']})
    assert logout.status_code == 200
    assert logout.json['csrf_token']

    login = client.post('/api/auth/login', json={
        'email': 'reader@example.com', 'password': 'long-test-password-123'
    }, headers={'X-CSRF-Token': logout.json['csrf_token']})
    assert login.status_code == 200
    assert login.json['user']['email'] == 'reader@example.com'


def test_database_outage_returns_safe_service_error(client, app, monkeypatch):
    from sqlalchemy.exc import OperationalError

    auth = register(client)

    def disconnected(*args, **kwargs):
        raise OperationalError('SELECT 1', {}, Exception('private-host-and-connection-details'))

    monkeypatch.setattr(app.extensions['store'], 'one', disconnected)
    # Session restoration also accesses the database, before the route runs.
    for path in ('/api/session', '/api/reports'):
        response = client.get(path)
        assert response.status_code == 503
        assert response.json['code'] == 'database_unavailable'
        assert 'database' in response.json['error'].lower()
        assert 'private-host' not in response.get_data(as_text=True)


def test_profile_updates_require_authentication_and_current_password(client, app):
    csrf = token(client)
    assert client.patch('/api/profile', json={'name': 'Reader'}, headers={'X-CSRF-Token': csrf}).status_code == 401

    auth = register(client)
    headers = {'X-CSRF-Token': auth['csrf_token']}
    update = {'name': 'Updated Reader', 'email': 'updated@example.com',
              'current_password': 'wrong-password', 'password': 'another-long-password-456',
              'confirm_password': 'another-long-password-456'}
    assert client.patch('/api/profile', json=update, headers=headers).status_code == 401

    update['current_password'] = 'long-test-password-123'
    response = client.patch('/api/profile', json=update, headers=headers)
    assert response.status_code == 200
    assert response.json['user']['name'] == 'Updated Reader'
    assert response.json['user']['email'] == 'updated@example.com'
    assert 'password_hash' not in response.json['user']

    stored = app.extensions['store'].one('users', {'id': auth['user']['id']})
    from werkzeug.security import check_password_hash
    assert check_password_hash(stored['password_hash'], 'another-long-password-456')

    new_client = app.test_client()
    login = new_client.post('/api/auth/login', json={
        'email': 'updated@example.com', 'password': 'another-long-password-456'
    }, headers={'X-CSRF-Token': token(new_client)})
    assert login.status_code == 200
    assert login.json['user']['name'] == 'Updated Reader'


def test_profile_rejects_duplicate_email_and_weak_password(client, app):
    auth = register(client)
    other = app.test_client()
    register(other, 'other@example.com')
    headers = {'X-CSRF-Token': auth['csrf_token']}

    duplicate = client.patch('/api/profile', json={
        'email': 'other@example.com', 'current_password': 'long-test-password-123'
    }, headers=headers)
    assert duplicate.status_code == 409

    weak_password = client.patch('/api/profile', json={
        'password': 'short', 'confirm_password': 'short',
        'current_password': 'long-test-password-123'
    }, headers=headers)
    assert weak_password.status_code == 400
    assert app.extensions['store'].one('users', {'email': 'reader@example.com'})['name'] == 'Reader'


def test_private_report_and_pdf_end_to_end(client, app):
    auth = register(client)
    response = client.post('/api/analyze', data={'type':'text', 'text':'A sample news claim with no trained classifier installed.', 'language':'te'},
                           headers={'X-CSRF-Token':auth['csrf_token']})
    assert response.status_code == 202
    report = wait_report(client, response.json['id'])
    assert report['status'] == 'complete'
    assert all(report['result'][k] is None for k in ('fake_percentage','real_percentage','ai_percentage','confidence'))
    assert report['result']['text_model_assessment']['verdict'] == 'not_assessed'
    assert report['result']['text_model_assessment']['reason']
    assert report['result']['text_metrics']['word_count'] > 0
    assert report['result']['text_metrics']['source_count'] == 0
    assert report['result']['verification']['status'] == 'unverified'
    pdf = client.get(f"/api/reports/{report['id']}/pdf")
    assert pdf.status_code == 200 and pdf.data.startswith(b'%PDF')
    assert f"verification-report-{report['id'][:10]}.pdf" in pdf.headers['Content-Disposition']
    other = app.test_client()
    register(other, 'other@example.com')
    assert other.get('/api/reports').json['items'] == []
    assert other.get(f"/api/reports/{report['id']}").status_code == 404
    assert other.get(f"/api/reports/{report['id']}/pdf").status_code == 404
    assert other.delete(f"/api/reports/{report['id']}", headers={'X-CSRF-Token':token(other)}).status_code == 404
    assert client.delete(f"/api/reports/{report['id']}", headers={'X-CSRF-Token':token(client)}).status_code == 200
    assert not client.get('/api/reports').json['items']


def test_image_validation_and_analysis(client):
    auth = register(client)
    headers={'X-CSRF-Token':auth['csrf_token']}
    bad = client.post('/api/analyze',data={'type':'image','file':(BytesIO(b'not a jpeg'),'lie.jpg')},headers=headers)
    assert bad.status_code == 400
    data=BytesIO()
    Image.new('RGB',(128,128),(100,120,80)).save(data,format='PNG')
    data.seek(0)
    response=client.post('/api/analyze',data={'type':'image','text':'The city council approved the new transport budget on Monday.','file':(data,'photo.png'),'language':'en'},headers=headers)
    assert response.status_code == 202
    report=wait_report(client,response.json['id'])
    assert report['status']=='complete'
    assert report['result']['sha256']
    assert report['type']=='image'
    assert report['result']['report_summary']['schema_version']==5
    assert report['result']['report_summary']['verdict']=='unverified'
    assert report['result']['media_text_analysis']['text_model_assessment']['status'] == 'not_configured'
    assert report['result']['fake_percentage'] is None
    assert report['result']['verification']['provider_status']=='not_configured'
    pdf=client.get(f"/api/reports/{report['id']}/pdf")
    assert pdf.status_code==200 and pdf.data.startswith(b'%PDF')


def test_report_question_returns_only_retrieved_source_passages(client, app):
    auth = register(client)
    report = app.extensions['store'].save('reports', {
        'owner_id':auth['user']['id'], 'type':'text', 'title':'Andhra Pradesh leadership',
        'content':'Who is the current Chief Minister of Andhra Pradesh?', 'status':'complete',
        'result':{'verification':{'evidence':[{'kind':'related_reporting', 'title':'Andhra Pradesh Chief Minister',
            'publisher':'Reuters', 'url':'https://www.reuters.com/world/india/example',
            'snippet':'N Chandrababu Naidu is the Chief Minister of Andhra Pradesh.', 'rating':'', 'claim':'',
            'retrieval_status':'article_excerpt_retrieved', 'retrieved_at':now()}]}}
    })
    headers={'X-CSRF-Token':auth['csrf_token']}
    answer=client.post(f"/api/reports/{report['id']}/ask", json={'question':'Who is the Chief Minister of Andhra Pradesh?'}, headers=headers)
    assert answer.status_code==200
    assert answer.json['status']=='source_excerpts'
    assert answer.json['sources'][0]['text']=='N Chandrababu Naidu is the Chief Minister of Andhra Pradesh.'
    assert answer.json['sources'][0]['url']=='https://www.reuters.com/world/india/example'
    absent=client.post(f"/api/reports/{report['id']}/ask", json={'question':'What is the weather tomorrow?'}, headers=headers)
    assert absent.json['status']=='insufficient_evidence' and absent.json['sources']==[]


def test_report_question_uses_configured_grounded_answer_api(client, app, monkeypatch):
    from types import SimpleNamespace
    import backend.rag as rag
    auth = register(client)
    report = app.extensions['store'].save('reports', {
        'owner_id':auth['user']['id'], 'type':'text', 'title':'Andhra Pradesh leadership',
        'content':'Who is the current Chief Minister of Andhra Pradesh?', 'status':'complete',
        'result':{'verification':{'evidence':[{'kind':'related_reporting', 'title':'Leadership report',
            'publisher':'Reuters', 'url':'https://www.reuters.com/world/india/example',
            'snippet':'N Chandrababu Naidu is the Chief Minister of Andhra Pradesh.', 'claim':'',
            'retrieval_status':'article_excerpt_retrieved', 'retrieved_at':now()}]}}
    })
    app.config['GEMINI_API_KEY'] = 'test-secret'
    seen = {}
    def generate(url, **kwargs):
        seen.update(url=url, **kwargs)
        return SimpleNamespace(raise_for_status=lambda: None,
            json=lambda: {'candidates':[{'content':{'parts':[{'text':'The source identifies N. Chandrababu Naidu as Chief Minister [S1].'}]}}]})
    monkeypatch.setattr(rag.requests, 'post', generate)
    answer = client.post(f"/api/reports/{report['id']}/ask",
        json={'question':'Who is the Chief Minister of Andhra Pradesh?'},
        headers={'X-CSRF-Token':auth['csrf_token']})
    assert answer.status_code == 200
    assert answer.json['status'] == 'answered'
    assert answer.json['answer_mode'] == 'gemini_grounded'
    assert answer.json['sources'][0]['url'] == 'https://www.reuters.com/world/india/example'
    assert seen['headers']['x-goog-api-key'] == 'test-secret'
    prompt = seen['json']['contents'][0]['parts'][0]['text']
    assert 'N Chandrababu Naidu is the Chief Minister' in prompt
    assert 'Do not use prior knowledge' in prompt


def test_video_frame_pipeline_without_model(client, tmp_path):
    import cv2
    import numpy as np
    path = tmp_path / 'clip.mp4'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),5,(64,64))
    for _ in range(10):
        writer.write(np.full((64,64,3),120,dtype=np.uint8))
    writer.release()
    auth=register(client)
    with path.open('rb') as handle:
        response=client.post('/api/analyze',data={'type':'video','text':'This clip makes a claim about a news event.','file':(handle,'clip.mp4')},headers={'X-CSRF-Token':auth['csrf_token']})
    report=wait_report(client,response.json['id'])
    assert report['status']=='complete'
    assert report['type']=='video'
    assert report['result']['report_summary']['schema_version']==5
    assert report['result']['report_summary']['verdict']=='unverified'
    assert report['result']['media_text_analysis']['text_model_assessment']['status'] == 'not_configured'
    assert any(item['label']=='Sampled frames' and int(item['value'])==2 for item in report['result']['observations'])
    assert report['result']['ai_percentage'] is None
    pdf=client.get(f"/api/reports/{report['id']}/pdf")
    assert pdf.status_code==200 and pdf.data.startswith(b'%PDF')


def test_old_report_summary_is_upgraded_on_open(client, app):
    auth=register(client)
    report=app.extensions['store'].save('reports', {
        'owner_id':auth['user']['id'], 'type':'text', 'title':'Old saved report',
        'content':'A sample claim that has not been checked by the available sources.', 'status':'complete',
        'result':{'fake_percentage':None, 'real_percentage':None, 'ai_percentage':None, 'confidence':None,
            'verification':{'provider_status':'searched', 'evidence':[]},
            'report_summary':{'schema_version':2, 'verdict':'unverified', 'headline':'This saved report predates evidence synthesis'}}
    })
    response=client.get(f"/api/reports/{report['id']}")
    assert response.status_code==200
    assert response.json['result']['report_summary']['schema_version']==5
    assert 'No source provider is configured' in response.json['result']['report_summary']['summary']


def test_health_never_exposes_service_keys(client, app):
    app.config['GEMINI_API_KEY']='gemini-test-secret-do-not-return'
    app.config['TAVILY_API_KEY']='search-test-secret-do-not-return'
    response=client.get('/api/health')
    assert response.status_code==200
    assert b'gemini-test-secret-do-not-return' not in response.data
    assert b'search-test-secret-do-not-return' not in response.data


def test_saved_time_sensitive_knowledge_is_corrected_on_open_without_api_calls(client, app, monkeypatch):
    auth = register(client)
    def forbidden(*a, **k):
        raise AssertionError('Saved-answer correction must not make external requests')
    monkeypatch.setattr('requests.post', forbidden)
    record = app.extensions['store'].save('reports', {
        'owner_id':auth['user']['id'], 'type':'text', 'status':'complete',
        'content':'current tamilnadu chief minister vijay is dead',
        'result':{'report_summary':{'schema_version':5, 'verdict':'unverified',
            'gemini_model_assessment':{'status':'completed', 'answer':'Vijay is alive and Stalin is current CM.',
                                      'reason':'Stored knowledge'}}}})
    updated = client.get('/api/reports/' + record['id'])
    assert updated.status_code == 200
    summary = updated.json['result']['report_summary']
    assert summary['gemini_model_assessment']['status'] == 'needs_current_sources'
    assert 'Stalin' not in str(summary)
    saved = app.extensions['store'].one('reports', {'id':record['id']})
    assert saved['result']['report_summary']['knowledge_policy_version'] == 2


def test_admin_access_and_source_controls(client, app):
    auth=register(client)
    assert client.get('/api/admin/overview').status_code==403
    store=app.extensions['store']
    user=store.one('users',{'id':auth['user']['id']})
    user['role']='admin'
    store.save('users',user)
    headers={'X-CSRF-Token':token(client)}
    assert client.get('/api/admin/overview').status_code==200
    assert client.patch('/api/admin/users/'+user['id'],json={'disabled':True},headers=headers).status_code==400
    assert client.post('/api/admin/sources',json={'name':'Bad source','url':'javascript:alert(1)'},headers=headers).status_code==400
    source=client.post('/api/admin/sources',json={'name':'Example source','url':'https://example.org'},headers=headers)
    assert source.status_code==201
    assert client.patch('/api/admin/sources/'+source.json['id'],json={'enabled':False},headers=headers).status_code==200
    assert all(s['id']!=source.json['id'] for s in client.get('/api/sources').json['items'])
    dataset=client.post('/api/admin/datasets',json={'name':'Archive','version':'v1','modality':'image','counts':{'train':100}},headers=headers)
    assert dataset.status_code==201


def test_admin_overview_reads_large_reports_without_sort_memory_failure(client, app):
    auth = register(client)
    store = app.extensions['store']
    admin = store.one('users', {'id': auth['user']['id']})
    admin['role'] = 'admin'
    store.save('users', admin)
    for index in range(3):
        store.save('reports', {'owner_id': admin['id'], 'type': 'text', 'status': 'complete',
            'title': f'Large report {index}', 'result': {'source_excerpt': 'x' * (512 * 1024)}})
    overview = client.get('/api/admin/overview')
    assert overview.status_code == 200
    assert overview.json['database'] == 'mysql'
    assert overview.json['counts']['reports'] == 3
    assert len(overview.json['reports']) == 3
    assert all(len(report['result']['source_excerpt']) == 512 * 1024 for report in overview.json['reports'])


def test_non_mysql_backend_does_not_silently_fallback():
    import pytest
    from backend.store import Store
    with pytest.raises(RuntimeError,match='mysql'):
        Store({'DATABASE_BACKEND':'sqlite'})


def test_only_server_startup_recovers_interrupted_jobs(app):
    from backend.app import create_app
    store=app.extensions['store']
    report=store.save('reports',{'owner_id':'test-owner','type':'text','status':'analyzing','title':'Interrupted job'})
    config=dict(app.config)
    cli_app=create_app(config)
    assert cli_app.extensions['store'].one('reports',{'id':report['id']})['status']=='analyzing'
    cli_app.extensions['analysis_executor'].shutdown(wait=True)
    config['RECOVER_INTERRUPTED_JOBS']=True
    server_app=create_app(config)
    recovered=server_app.extensions['store'].one('reports',{'id':report['id']})
    assert recovered['status']=='failed' and 'restart' in recovered['error']
    server_app.extensions['analysis_executor'].shutdown(wait=True)


def test_production_requires_mysql_and_secure_cookies(tmp_path):
    import pytest
    from backend.app import create_app
    with pytest.raises(RuntimeError,match='Production requires'):
        create_app({'APP_ENV':'production','SECRET_KEY':'test-production-secret','DATABASE_BACKEND':'sqlite',
                    'SESSION_COOKIE_SECURE':False,'SQLITE_PATH':str(tmp_path/'not-created.sqlite')})
    assert not (tmp_path/'not-created.sqlite').exists()


def test_json_endpoints_reject_non_object_bodies(client, app):
    auth = register(client)
    user = app.extensions['store'].one('users', {'id': auth['user']['id']})
    user['role'] = 'admin'
    app.extensions['store'].save('users', user)
    headers = {'X-CSRF-Token': auth['csrf_token']}
    endpoints = [('post', '/api/auth/register'), ('post', '/api/auth/login'),
                 ('post', '/api/admin/sources'), ('post', '/api/admin/datasets'),
                 ('patch', '/api/admin/users/missing'), ('patch', '/api/admin/sources/missing')]
    # Existing-record lookups can return 404 first. Use real IDs for patch paths.
    other = app.extensions['store'].save('users', {'name': 'Other', 'email': 'other@invalid.example',
        'role': 'user', 'disabled': False, 'password_hash': 'unused'})
    source = client.get('/api/sources').json['items'][0]
    endpoints[-2] = ('patch', '/api/admin/users/' + other['id'])
    endpoints[-1] = ('patch', '/api/admin/sources/' + source['id'])
    for method, endpoint in endpoints:
        for payload in ([], ['unexpected'], 'unexpected', 42, None):
            response = getattr(client, method)(endpoint, data=json.dumps(payload),
                content_type='application/json', headers=headers)
            assert response.status_code == 400, (endpoint, payload, response.json)
            assert 'JSON object' in response.json['error']


def test_auth_fields_require_strings_and_csrf_rejects_unicode(client, app):
    csrf = token(client)
    headers = {'X-CSRF-Token': csrf}
    body = {'name': 'Reader', 'email': 'reader@example.com', 'password': 'long-test-password-123'}
    for field in body:
        invalid = dict(body, **{field: ['long-enough-to-have-previously-passed-validation']})
        assert client.post('/api/auth/register', json=invalid, headers=headers).status_code == 400
    assert app.extensions['store'].count('users') == 0
    oversized = dict(body, name='x' * (64 * 1024))
    assert client.post('/api/auth/register', json=oversized, headers=headers).status_code == 413
    assert client.post('/api/auth/login', json={'email': [], 'password': {}}, headers=headers).status_code == 400
    assert client.post('/api/auth/login', json={}, headers={'X-CSRF-Token': '\u00e9'}).status_code == 403


def test_password_reset_email_flow_is_opaque_single_use_and_revokes_sessions(client, app, monkeypatch):
    from queue import Queue
    import re

    auth = register(client)
    sent = Queue()
    monkeypatch.setattr('backend.app.send_password_reset_email',
        lambda config, recipient, name, reset_url: sent.put((recipient, reset_url)))
    app.config.update(MAIL_HOST='smtp.example.test', MAIL_FROM='noreply@example.test',
        MAIL_USERNAME=None, MAIL_PASSWORD=None, MAIL_USE_TLS=True, MAIL_USE_SSL=False)
    headers = {'X-CSRF-Token': auth['csrf_token']}

    requested = client.post('/api/auth/forgot-password', json={'email': 'reader@example.com'}, headers=headers)
    assert requested.status_code == 200
    assert requested.json['message'] == 'If an account uses that email, a reset link will be sent shortly.'
    recipient, reset_url = sent.get(timeout=2)
    assert recipient == 'reader@example.com'
    token_match = re.search(r'#reset/([A-Za-z0-9_-]+)$', reset_url)
    assert token_match
    raw_token = token_match.group(1)
    saved = app.extensions['store'].find('password_resets', {'user_id': auth['user']['id']})
    assert len(saved) == 1 and raw_token not in str(saved[0])

    changed = client.post('/api/auth/reset-password', json={
        'token': raw_token, 'password': 'new-password-for-reader-789'
    }, headers=headers)
    assert changed.status_code == 200
    assert changed.json['ok'] is True

    another = app.test_client()
    replay = another.post('/api/auth/reset-password', json={
        'token': raw_token, 'password': 'another-password-for-reader-456'
    }, headers={'X-CSRF-Token': token(another)})
    assert replay.status_code == 400

    login = client.post('/api/auth/login', json={
        'email': 'reader@example.com', 'password': 'new-password-for-reader-789'
    }, headers={'X-CSRF-Token': token(client)})
    assert login.status_code == 200

    unknown = client.post('/api/auth/forgot-password', json={'email': 'unknown@example.com'},
        headers={'X-CSRF-Token': login.json['csrf_token']})
    assert unknown.status_code == 200 and unknown.json == requested.json
    assert sent.empty()


def test_password_reset_requires_mail_configuration_and_rejects_expired_tokens(client, app):
    import hashlib
    from datetime import datetime, timedelta, timezone

    headers = {'X-CSRF-Token': token(client)}
    unavailable = client.post('/api/auth/forgot-password', json={'email': 'nobody@example.com'}, headers=headers)
    assert unavailable.status_code == 503

    auth = register(client)
    raw_token = 'expired-token-for-test'
    app.extensions['store'].save('password_resets', {
        'id': hashlib.sha256(raw_token.encode()).hexdigest(), 'user_id': auth['user']['id'],
        'expires_at': datetime.now(timezone.utc) - timedelta(seconds=1),
    })
    response = client.post('/api/auth/reset-password', json={
        'token': raw_token, 'password': 'another-new-password-789'
    }, headers={'X-CSRF-Token': auth['csrf_token']})
    assert response.status_code == 400


def test_source_validation_and_unique_metadata_conflicts(client, app, monkeypatch):
    auth = register(client)
    store = app.extensions['store']
    user = store.one('users', {'id': auth['user']['id']})
    store.save('users', dict(user, role='admin'))
    headers = {'X-CSRF-Token': auth['csrf_token']}
    for url in ('https://[broken', 'https://example.org:invalid', 'https://bad host.example/'):
        assert client.post('/api/admin/sources', json={'name': 'Invalid', 'url': url}, headers=headers).status_code == 400
    source = {'name': 'Example', 'url': 'https://example.org'}
    assert client.post('/api/admin/sources', json=source, headers=headers).status_code == 201
    dataset = {'name': 'Dataset', 'version': 'v1', 'modality': 'image', 'counts': {'train': 1}}
    assert client.post('/api/admin/datasets', json=dataset, headers=headers).status_code == 201
    original_one = store.one
    # Simulate two administrators passing the duplicate precheck concurrently.
    monkeypatch.setattr(store, 'one', lambda collection, query: None if collection in ('sources', 'datasets') else original_one(collection, query))
    assert client.post('/api/admin/sources', json=source, headers=headers).status_code == 409
    assert client.post('/api/admin/datasets', json=dataset, headers=headers).status_code == 409
    assert store.count('datasets') == 1


def test_image_requires_supported_decoded_format(client):
    auth = register(client)
    image = BytesIO()
    Image.new('RGB', (10, 10)).save(image, format='GIF')
    image.seek(0)
    response = client.post('/api/analyze', data={'type': 'image', 'file': (image, 'disguised.jpg')},
        headers={'X-CSRF-Token': auth['csrf_token']})
    assert response.status_code == 400
    assert client.get('/api/reports').json['items'] == []


@pytest.mark.parametrize('mode', ['media_only', 'claim_review'])
def test_saved_image_origin_is_corrected_without_changing_source_verdict(client, app, mode):
    auth = register(client)
    store = app.extensions['store']
    report = store.save('reports', {
        'owner_id': auth['user']['id'], 'type': 'image', 'status': 'complete', 'title': 'Old image',
        'result': {'ai_percentage': 99.9, 'confidence': 99.9,
                   'ai_assessment': {'status': 'experimental', 'calibrated': False},
                   'media_origin': {'verdict': 'ai_generated', 'score': 99.9},
                   'report_summary': {'schema_version': 5, 'mode': mode, 'verdict': 'false',
                                      'headline': 'Likely AI-generated',
                                      'summary': 'The model classifies this as likely AI-generated. A portrait.'}}})
    updated = client.get('/api/reports/' + report['id']).json['result']
    assert updated['media_origin']['verdict'] == 'inconclusive'
    assert updated['ai_percentage'] == 99.9
    assert updated['report_summary']['verdict'] == 'false'
    if mode == 'media_only':
        assert updated['report_summary']['headline'] == 'Origin inconclusive'
        assert 'A portrait.' in updated['report_summary']['summary']
        assert 'classifies this as likely AI-generated' not in updated['report_summary']['summary']
    else:
        assert updated['report_summary']['summary'] == report['result']['report_summary']['summary']
    saved = store.one('reports', {'id': report['id']})
    assert saved['result']['media_origin']['policy_version'] == 8
    assert saved['result']['media_origin']['score'] is None


def test_saved_illustration_loses_unsupported_real_label_without_losing_metrics(client, app):
    auth = register(client)
    report = app.extensions['store'].save('reports', {
        'owner_id': auth['user']['id'], 'type': 'image', 'status': 'complete',
        'result': {'ai_percentage': 8., 'real_percentage': 92., 'confidence': 92.,
                   'ai_assessment': {'prediction_enabled': True},
                   'media_origin': {'verdict': 'likely_real', 'score': 8.},
                   'media_inspection': {'visual_type': 'cartoon_or_illustration', 'description': 'An illustrated portrait.'},
                   'report_summary': {'schema_version': 5, 'mode': 'media_only',
                       'headline': 'Likely real image',
                       'summary': 'The model classifies this as a likely real image. An illustrated portrait.'}}})
    result = client.get('/api/reports/' + report['id']).json['result']
    assert result['media_origin']['verdict'] == 'inconclusive'
    assert result['ai_percentage'] == 8.
    assert 'likely real' not in result['report_summary']['summary'].lower()
    assert result['report_summary']['headline'] == 'Origin inconclusive'


def test_saved_watermark_report_is_repaired_even_with_current_policy(client, app):
    auth = register(client)
    report = app.extensions['store'].save('reports', {
        'owner_id': auth['user']['id'], 'type': 'image', 'status': 'complete', 'content': '',
        'result': {'ai_percentage': 99.9, 'ai_assessment': {'locally_validated': False},
                   'media_origin': {'policy_version': 1, 'verdict': 'ai_generated'},
                   'media_inspection': {'visible_text': 'REDMI ON JOYEE', 'visual_type': 'photograph',
                                        'description': 'People at a table.'},
                   'report_summary': {'schema_version': 5, 'verdict': 'unverified'},
                   'verification': {'status': 'unverified', 'evidence': []}}})
    url = '/api/reports/' + report['id']
    result = client.get(url).json['result']
    assert result['media_origin']['verdict'] == 'inconclusive'
    assert result['report_summary']['mode'] == 'media_only'
    assert result['verification']['status'] == 'not_applicable'
    assert result['media_inspection']['visible_text'] == 'REDMI ON JOYEE'
    assert client.get(url).json['result'] == result


def test_admin_cli_does_not_create_account_with_unusable_password(app):
    # Login supports passwords up to 128 characters. The CLI must enforce the
    # same bound so it cannot create an administrator who can never sign in.
    password = 'x' * 129
    response = app.test_cli_runner().invoke(args=['create-admin'],
        input=f'Administrator\nadmin@example.com\n{password}\n{password}\n')
    assert response.exit_code != 0
    assert '12-128' in response.output
    assert app.extensions['store'].count('users') == 0


@pytest.mark.parametrize('secret', ['short-key', 'x' * 31, 12345])
def test_production_rejects_weak_session_keys_before_database_access(monkeypatch, secret):
    import backend.app as module
    def unexpected_store(*args, **kwargs):
        pytest.fail('Invalid production configuration must fail before connecting to the database.')
    monkeypatch.setattr(module, 'Store', unexpected_store)
    with pytest.raises(RuntimeError, match='at least 32 characters'):
        module.create_app({'APP_ENV': 'production', 'SECRET_KEY': secret,
            'DATABASE_BACKEND': 'mysql', 'SESSION_COOKIE_SECURE': True})


def test_non_mysql_storage_is_rejected_before_initialization():
    from backend.app import create_app
    with pytest.raises(RuntimeError, match='DATABASE_BACKEND must be mysql'):
        create_app({'APP_ENV': 'production', 'SECRET_KEY': 'x' * 32,
            'DATABASE_BACKEND': 'sqlite', 'SESSION_COOKIE_SECURE': True})


def test_health_reports_database_outage_for_signed_in_sessions(client, app, monkeypatch):
    register(client)
    def disconnected(*args, **kwargs):
        raise RuntimeError('Simulated database unavailable')
    monkeypatch.setattr(app.extensions['store'], 'one', disconnected)
    monkeypatch.setattr(app.extensions['store'], 'ping', disconnected)
    response = client.get('/api/health')
    assert response.status_code == 503
    assert response.json == {'status': 'degraded', 'database': 'mysql', 'database_connected': False,
                             'password_reset_email': False}


@pytest.mark.parametrize('failure_point', ['executor', 'sources', 'cleanup'])
def test_scheduling_failure_marks_report_failed_and_releases_resources(client, app, monkeypatch, tmp_path, failure_point):
    import backend.app as app_module
    auth = register(client)
    store = app.extensions['store']
    original_tempfile = app_module.tempfile.NamedTemporaryFile
    monkeypatch.setattr(app_module.tempfile, 'NamedTemporaryFile',
        lambda **kwargs: original_tempfile(dir=tmp_path, **kwargs))
    def unavailable(*args, **kwargs):
        raise RuntimeError('Simulated unavailable scheduler or source store')
    if failure_point in ('executor', 'cleanup'):
        monkeypatch.setattr(app.extensions['analysis_executor'], 'submit', unavailable)
        if failure_point == 'cleanup':
            original_unlink = app_module.Path.unlink
            def locked_upload(path, *args, **kwargs):
                if path.suffix == '.png':
                    raise PermissionError('Simulated temporarily locked upload')
                return original_unlink(path, *args, **kwargs)
            monkeypatch.setattr(app_module.Path, 'unlink', locked_upload)
    else:
        original_find = store.find
        monkeypatch.setattr(store, 'find', lambda collection, *args, **kwargs:
            unavailable() if collection == 'sources' else original_find(collection, *args, **kwargs))
    for _ in range(5):  # Exceeds all four slots, catching leaked capacity.
        image = BytesIO()
        Image.new('RGB', (10, 10)).save(image, format='PNG')
        image.seek(0)
        response = client.post('/api/analyze', data={'type': 'image', 'text':'This image has a disputed claim.', 'file': (image, 'sample.png')},
            headers={'X-CSRF-Token': auth['csrf_token']})
        assert response.status_code == 500
    reports = client.get('/api/reports').json['items']
    assert len(reports) == 5
    assert all(report['status'] == 'failed' and 'scheduled' in report['error'] for report in reports)
    if failure_point != 'cleanup':
        assert not list(tmp_path.glob('*.png'))
