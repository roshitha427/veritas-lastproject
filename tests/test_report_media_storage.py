from copy import deepcopy
from io import BytesIO

from PIL import Image
import pytest

from backend import report_media as media


def test_bounded_image_preview_keeps_aspect_and_strips_metadata(tmp_path):
    path = tmp_path / 'photo.png'
    Image.new('RGBA', (2400, 1200), (100, 150, 200, 100)).save(path)
    stored = media.make_image_preview(path)
    report = {'id': 'private-image', '_image_preview': stored}
    content = media.preview_bytes(report)
    assert len(content) <= media.MAX_PREVIEW_BYTES
    with Image.open(BytesIO(content)) as image:
        assert image.format == 'JPEG'
        assert image.size == (1280, 640)
        assert not image.getexif()
    public = media.public_report(report)
    assert '_image_preview' not in public
    assert 'data_base64' not in str(public)
    assert public['image_preview']['url'] == '/api/reports/private-image/preview'


def test_corrupt_preview_is_not_served_and_old_report_has_no_fake_image():
    report = {'id': 'old', 'title': 'Old report'}
    assert media.preview_bytes(report) is None
    assert media.public_report(report) == report
    image = media.make_image_preview(Image.new('RGB', (20, 20)))
    image['sha256'] = 'wrong'
    assert media.preview_bytes(dict(report, _image_preview=image)) is None


def test_private_video_copy_is_bound_to_report_and_removed_on_delete(tmp_path, monkeypatch):
    monkeypatch.setattr(media, 'VIDEO_DIRECTORY', tmp_path / 'private-videos')
    source = tmp_path / 'source.mp4'
    source.write_bytes(b'bounded-test-video')
    stored = media.make_video_preview(source, 'report-one')
    report = {'id': 'report-one', '_video_preview': stored}
    retained = media.video_preview_path(report)
    assert retained.read_bytes() == source.read_bytes()
    assert retained != source
    assert media.video_preview_path(dict(report, id='report-two')) is None
    public = media.public_report(report)
    assert '_video_preview' not in public and 'storage_key' not in str(public)
    assert public['video_preview']['url'] == '/api/reports/report-one/video'
    assert media.preview_bytes(report) is None  # Videos never become PDF images.
    media.remove_report_media(report)
    assert not retained.exists()
    assert source.exists()


def test_video_path_cannot_escape_private_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(media, 'VIDEO_DIRECTORY', tmp_path / 'private')
    outside = tmp_path / 'outside.mp4'
    outside.write_bytes(b'must-remain')
    report = {'id': 'safe', '_video_preview': {'version': 1, 'mime_type': 'video/mp4',
        'storage_key': '../outside.mp4', 'size_bytes': outside.stat().st_size}}
    original = deepcopy(report)
    assert media.video_preview_path(report) is None
    media.remove_report_media(report)
    assert outside.read_bytes() == b'must-remain'
    assert report == original


def test_preview_strips_private_jpeg_comments(tmp_path):
    path = tmp_path / 'commented.jpg'
    Image.new('RGB', (80, 60)).save(path, comment=b'private-location-comment')
    result = media.make_image_preview(path)
    with Image.open(BytesIO(media.preview_bytes({'_image_preview': result}))) as image:
        assert not image.info.get('comment')
        assert not image.getexif()


def test_corrupt_video_is_still_removed_by_report_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(media, 'VIDEO_DIRECTORY', tmp_path / 'private-videos')
    source = tmp_path / 'source.mp4'
    source.write_bytes(b'bounded-video-data')
    stored = media.make_video_preview(source, 'owned-corrupt')
    report = {'id': 'owned-corrupt', '_video_preview': stored}
    path = media.video_preview_path(report)
    path.write_bytes(b'truncated')
    assert media.video_preview_path(report) is None
    media.remove_report_media(report)
    assert not path.exists()
    assert media.video_poster_bytes({'_video_preview': 'invalid'}) is None


def test_video_limit_is_enforced_during_copy(tmp_path, monkeypatch):
    monkeypatch.setattr(media, 'VIDEO_DIRECTORY', tmp_path / 'private-videos')
    monkeypatch.setattr(media, 'MAX_VIDEO_BYTES', 10)
    source = tmp_path / 'growing.mp4'
    source.write_bytes(b'small')
    original_open = type(source).open
    def changed_content(path, mode='r', *args, **kwargs):
        return BytesIO(b'x' * 20) if path == source and mode == 'rb' else original_open(path, mode, *args, **kwargs)
    monkeypatch.setattr(type(source), 'open', changed_content)
    with pytest.raises(ValueError, match='size limit'):
        media.make_video_preview(source, 'growing')
    assert list(media.VIDEO_DIRECTORY.iterdir()) == []
