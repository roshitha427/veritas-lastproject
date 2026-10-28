from io import BytesIO
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import pytest

from backend import report_media as media
from backend.pdf_report import report_pdf
from tests.conftest import register, token


def result_for(kind):
    return {'media_origin': {'verdict': 'inconclusive'}, 'ai_percentage': 51.9,
            'report_summary': {'schema_version': 5, 'mode': 'media_only', 'verdict': 'unverified'},
            'media_inspection': {'description': 'A local preview test.'}}


def finished(client, report_id):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        report = client.get('/api/reports/' + report_id).json
        if report['status'] in ('complete', 'failed'):
            assert report['status'] == 'complete', report.get('error')
            return report
        time.sleep(.02)
    raise AssertionError('Preview test job did not finish')


def test_uploaded_image_history_report_pdf_and_private_deletion(client, app, monkeypatch):
    auth = register(client)
    seen = {}
    def analyze(kind, text, filepath, language, sources):
        seen['original_path'] = filepath
        return result_for(kind)
    monkeypatch.setattr(app.extensions['analyzer'], 'analyze', analyze)
    image = BytesIO()
    Image.new('RGB', (180, 120), '#3279a8').save(image, format='PNG')
    response = client.post('/api/analyze', data={'type': 'image', 'file': (BytesIO(image.getvalue()), 'my-picture.png')},
        headers={'X-CSRF-Token': auth['csrf_token']})
    assert response.status_code == 202
    report = finished(client, response.json['id'])
    preview_url = report['image_preview']['url']
    assert '_image_preview' not in report and 'data_base64' not in str(report)
    history = client.get('/api/reports').json['items']
    assert history[0]['image_preview']['url'] == preview_url
    assert 'data_base64' not in str(history)
    preview = client.get(preview_url)
    assert preview.status_code == 200 and preview.mimetype == 'image/jpeg'
    assert 'no-store' in preview.headers['Cache-Control']
    assert preview.headers['X-Content-Type-Options'] == 'nosniff'
    assert app.test_client().get(preview_url).status_code == 401
    other = app.test_client()
    register(other, email='other-preview@example.com')
    assert other.get(preview_url).status_code == 404
    pdf = client.get('/api/reports/' + report['id'] + '/pdf')
    assert pdf.status_code == 200 and b'/Subtype /Image' in pdf.data
    assert report['result']['report_contract']['report_id'] == report['id']
    deleted = client.delete('/api/reports/' + report['id'], headers={'X-CSRF-Token': token(client)})
    assert deleted.status_code == 200
    assert client.get(preview_url).status_code == 404
    assert not Path(seen['original_path']).exists()


def test_video_history_playback_range_and_poster_are_private_and_not_in_pdf(client, app, monkeypatch, tmp_path):
    app.config['REPORT_VIDEO_DIR'] = str(tmp_path / 'retained-video')
    source = tmp_path / 'short.mp4'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'), 5., (96, 64))
    if not writer.isOpened():
        pytest.skip('MP4 test codec unavailable')
    for value in range(5):
        writer.write(np.full((64, 96, 3), 40 + value * 25, dtype=np.uint8))
    writer.release()
    content = source.read_bytes()
    auth = register(client)
    monkeypatch.setattr(app.extensions['analyzer'], 'analyze', lambda *args, **kwargs: result_for('video'))
    response = client.post('/api/analyze', data={'type': 'video', 'file': (BytesIO(content), 'my-video.mp4')},
        headers={'X-CSRF-Token': auth['csrf_token']})
    assert response.status_code == 202
    report = finished(client, response.json['id'])
    preview = report['video_preview']
    assert 'storage_key' not in str(report) and 'data_base64' not in str(report)
    assert client.get('/api/reports').json['items'][0]['video_preview']['url'] == preview['url']
    poster = client.get(preview['poster_url'])
    assert poster.status_code == 200 and poster.mimetype == 'image/jpeg'
    stream = client.get(preview['url'], headers={'Range': 'bytes=0-20'})
    assert stream.status_code == 206 and stream.data == content[:21]
    assert stream.headers['Content-Range'].startswith('bytes 0-20/')
    assert 'no-store' in stream.headers['Cache-Control']
    stream.close()
    other = app.test_client()
    register(other, email='other-video@example.com')
    assert other.get(preview['url']).status_code == 404
    assert other.get(preview['poster_url']).status_code == 404
    internal = app.extensions['store'].one('reports', {'id': report['id']})
    retained = media.video_preview_path(internal, app.config['REPORT_VIDEO_DIR'])
    assert retained.is_file()
    pdf = client.get('/api/reports/' + report['id'] + '/pdf')
    assert pdf.status_code == 200 and b'/Subtype /Image' not in pdf.data
    assert client.delete('/api/reports/' + report['id'], headers={'X-CSRF-Token': token(client)}).status_code == 200
    assert not retained.exists()
    assert client.get(preview['url']).status_code == 404


@pytest.mark.parametrize('verdict', ['true', 'unverified'])
def test_both_pdf_renderers_embed_only_uploaded_image_previews(verdict):
    report = {'id': 'pdf-image', 'type': 'image', 'created_at': '2026-10-25', 'content': 'A bridge opened.',
        '_image_preview': media.make_image_preview(Image.new('RGB', (180, 90), 'blue')),
        'result': {'claim_text': 'A bridge opened.', 'media_origin': {'verdict': 'ai_generated'},
                   'report_summary': {'schema_version': 5, 'verdict': verdict, 'summary': 'Recorded claim assessment.'}}}
    assert b'/Subtype /Image' in report_pdf(report)
    report['type'] = 'video'
    assert b'/Subtype /Image' not in report_pdf(report)
