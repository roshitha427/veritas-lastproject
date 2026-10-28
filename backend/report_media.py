"""Bounded, account-private report images and retained video previews.

Image previews and video posters travel with the private report document.
Videos use ordinary copies in a private data directory, never file links.
Only authenticated report routes expose the generated media URLs.
"""
import base64
from hashlib import file_digest, sha256
from io import BytesIO
import os
from pathlib import Path
import tempfile
from urllib.parse import quote

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
VIDEO_DIRECTORY = ROOT / 'data' / 'report-videos'
MAX_PREVIEW_BYTES = 512 * 1024
MAX_VIDEO_BYTES = 50 * 1024 * 1024
VIDEO_TYPES = {'.mp4': 'video/mp4', '.webm': 'video/webm', '.mov': 'video/quicktime'}


def make_image_preview(source):
    """Create a whole-image JPEG preview without EXIF/location metadata."""
    opened = source.copy() if isinstance(source, Image.Image) else Image.open(source)
    with opened:
        if opened.width * opened.height > 30_000_000:
            raise ValueError('Image is too large for a report preview.')
        image = ImageOps.exif_transpose(opened)
        if 'A' in image.getbands() or image.mode == 'P':
            rgba = image.convert('RGBA')
            image = Image.new('RGB', rgba.size, 'white')
            image.paste(rgba, mask=rgba.getchannel('A'))
        else:
            image = image.convert('RGB')
        image.info.clear()
        image.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
        for edge in (1280, 1000, 800, 600, 400):
            image.thumbnail((edge, edge), Image.Resampling.LANCZOS)
            for quality in (88, 78, 65):
                output = BytesIO()
                image.save(output, format='JPEG', quality=quality, optimize=True, exif=b'', comment=b'')
                content = output.getvalue()
                if len(content) <= MAX_PREVIEW_BYTES:
                    return {'version': 1, 'mime_type': 'image/jpeg', 'width': image.width,
                            'height': image.height, 'sha256': sha256(content).hexdigest(),
                            'data_base64': base64.b64encode(content).decode('ascii')}
    raise ValueError('A bounded report preview could not be created.')


def _image_bytes(stored):
    if not isinstance(stored, dict) or stored.get('version') != 1 or stored.get('mime_type') != 'image/jpeg':
        return None
    encoded = stored.get('data_base64')
    if not isinstance(encoded, str) or len(encoded) > (MAX_PREVIEW_BYTES + 2) // 3 * 4:
        return None
    try:
        content = base64.b64decode(encoded, validate=True)
        if (len(content) > MAX_PREVIEW_BYTES or not content.startswith(b'\xff\xd8')
                or sha256(content).hexdigest() != stored.get('sha256')):
            return None
        return content
    except (ValueError, TypeError):
        return None


def preview_bytes(report):
    return _image_bytes(report.get('_image_preview'))


def preview_metadata(report):
    stored = report.get('_image_preview')
    if not isinstance(stored, dict) or not report.get('id') or not stored.get('data_base64'):
        return None
    return {'url': '/api/reports/' + quote(str(report['id']), safe='') + '/preview',
            **{key: stored.get(key) for key in ('width', 'height', 'mime_type', 'sha256')}}


def _video_filename(report_id, suffix):
    return sha256(str(report_id).encode('utf-8')).hexdigest() + suffix


def make_video_preview(filepath, report_id, directory=None):
    """Retain a bounded copy after video analysis; caller saves its private metadata."""
    video_directory = Path(directory) if directory is not None else VIDEO_DIRECTORY
    source = Path(filepath)
    suffix = source.suffix.lower()
    size = source.stat().st_size
    if suffix not in VIDEO_TYPES or not 0 < size <= MAX_VIDEO_BYTES:
        raise ValueError('Only bounded MP4, WebM, and MOV previews are supported.')
    video_directory.mkdir(parents=True, exist_ok=True)
    filename = _video_filename(report_id, suffix)
    destination = video_directory / filename
    if destination.resolve().parent != video_directory.resolve():
        raise ValueError('Invalid private video location.')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=video_directory, suffix='.part', delete=False) as output:
            temporary = Path(output.name)
            with source.open('rb') as original:
                copied = 0
                while chunk := original.read(1024 * 1024):
                    copied += len(chunk)
                    if copied > MAX_VIDEO_BYTES:
                        raise ValueError('The video exceeds the private preview size limit.')
                    output.write(chunk)
                if not copied:
                    raise ValueError('The video preview is empty.')
                size = copied
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)
    with destination.open('rb') as handle:
        digest = file_digest(handle, 'sha256').hexdigest()
    stored = {'version': 1, 'storage_key': filename, 'mime_type': VIDEO_TYPES[suffix],
              'size_bytes': size, 'sha256': digest}
    try:
        import cv2
    except ImportError:
        return stored
    capture = None
    try:
        capture = cv2.VideoCapture(str(source))
        decoded, frame = capture.read()
        if decoded:
            stored['poster'] = make_image_preview(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    except (OSError, ValueError, RuntimeError, cv2.error):
        pass  # The retained video can still be played without a poster.
    finally:
        if capture is not None:
            capture.release()
    return stored


def video_preview_path(report, directory=None):
    video_directory = Path(directory) if directory is not None else VIDEO_DIRECTORY
    stored = report.get('_video_preview')
    if not isinstance(stored, dict) or stored.get('version') != 1 or not report.get('id'):
        return None
    suffix = next((ext for ext, mime in VIDEO_TYPES.items() if stored.get('mime_type') == mime), None)
    if suffix is None or stored.get('storage_key') != _video_filename(report['id'], suffix):
        return None
    path = video_directory / stored['storage_key']
    try:
        if (path.resolve().parent != video_directory.resolve() or not path.is_file()
                or not 0 < path.stat().st_size <= MAX_VIDEO_BYTES
                or path.stat().st_size != stored.get('size_bytes')):
            return None
    except OSError:
        return None
    return path


def video_poster_bytes(report):
    stored = report.get('_video_preview')
    return _image_bytes(stored.get('poster')) if isinstance(stored, dict) else None


def video_preview_metadata(report, directory=None):
    if video_preview_path(report, directory) is None:
        return None
    stored = report['_video_preview']
    prefix = '/api/reports/' + quote(str(report['id']), safe='')
    metadata = {'url': prefix + '/video',
                **{key: stored.get(key) for key in ('mime_type', 'size_bytes', 'sha256')}}
    if stored.get('poster'):
        metadata['poster_url'] = prefix + '/poster'
    return metadata


def public_report(report, directory=None):
    """Never send retained media bytes or storage keys in report JSON."""
    public = {key: value for key, value in report.items() if key not in ('_image_preview', '_video_preview')}
    for key, metadata in (('image_preview', preview_metadata(report)), ('video_preview', video_preview_metadata(report, directory))):
        public.pop(key, None)
        if metadata:
            public[key] = metadata
    return public


def remove_report_media(report, directory=None):
    """Delete only this report's validated private video file; no recursive deletion."""
    video_directory = Path(directory) if directory is not None else VIDEO_DIRECTORY
    if not report.get('id'):
        return
    # Deletion is bound to identity, not playback validity or mutable metadata.
    # A truncated file must still be removed when its report is deleted.
    for suffix in VIDEO_TYPES:
        path = video_directory / _video_filename(report['id'], suffix)
        if path.parent.resolve() == video_directory.resolve():
            path.unlink(missing_ok=True)
