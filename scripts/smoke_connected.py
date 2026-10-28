"""Exercise the configured database/API/model/report path, then remove QA records."""
import json
import secrets
import time
from uuid import uuid4
from backend.app import create_app


def main():
    app=create_app()
    store=app.extensions['store']
    client=app.test_client()
    user_id=report_id=None
    try:
        csrf=client.get('/api/session').json['csrf_token']
        response=client.post('/api/auth/register',json={'name':'Connection QA','email':f'veritas-qa-{uuid4().hex}@example.invalid',
            'password':secrets.token_urlsafe(24)},headers={'X-CSRF-Token':csrf})
        assert response.status_code==200,'Registration failed'
        user_id=response.json['user']['id']
        csrf=response.json['csrf_token']
        text='The city council met on Monday to review a proposal for additional bus services. Officials said the draft budget would be discussed at the next public meeting.'
        response=client.post('/api/analyze',data={'type':'text','text':text,'language':'en'},headers={'X-CSRF-Token':csrf})
        assert response.status_code==202,'Submission failed'
        report_id=response.json['id']
        deadline=time.monotonic()+120
        while time.monotonic()<deadline:
            report=client.get('/api/reports/'+report_id).json
            if report['status'] in ('complete','failed'):break
            time.sleep(.3)
        assert report['status']=='complete','Analysis failed or timed out'
        assert report['result']['fake_percentage'] is not None,'Trained text inference was unavailable'
        assert len(client.get('/api/reports').json['items'])==1,'Private history mismatch'
        pdf=client.get('/api/reports/'+report_id+'/pdf')
        assert pdf.status_code==200 and pdf.data.startswith(b'%PDF'),'PDF generation failed'
        assert client.delete('/api/reports/'+report_id,headers={'X-CSRF-Token':csrf}).status_code==200,'Report deletion failed'
        print(json.dumps({'database':store.kind,'registration':'passed','analysis':'passed','history':'passed',
            'pdf':'passed','pdf_bytes':len(pdf.data),'report_delete':'passed',
            'text_model_version':report['result']['models'][0]['version'],'verification_status':report['result']['verification']['status']}))
    finally:
        app.extensions['analysis_executor'].shutdown(wait=True)
        app.extensions['mail_executor'].shutdown(wait=True)
        try:
            if report_id:
                store.delete('reports',report_id)
            if user_id:
                store.delete('users',user_id)
        finally:
            store.close()


if __name__=='__main__':
    main()
