"""Install a trusted, SHA-256-bound inference bundle. Never loads model objects."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import tempfile
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, build_opener
import zipfile

from scripts.export_models import FORMAT, FILES, digest

MAX_ARCHIVE_BYTES = 2 * 1024 ** 3
MAX_EXPANDED_BYTES = 4 * 1024 ** 3


def _inherit_model_permissions(target, destination):
    """Remove Windows staging-only ACLs after moving a verified model into place."""
    if os.name != 'nt':
        return
    if (target.name not in FILES or target.is_symlink() or target.is_junction()
            or not target.is_dir() or target.resolve().parent != destination):
        raise ValueError('Unsafe model permission destination.')
    executable = Path(os.environ.get('SystemRoot', r'C:\Windows')) / 'System32' / 'icacls.exe'
    try:
        result = subprocess.run([str(executable), str(target), '/reset', '/T', '/C', '/Q'],
                                capture_output=True, check=False, timeout=30)
    except subprocess.TimeoutExpired:
        raise OSError('Applying installed model permissions timed out.') from None
    if result.returncode:
        raise OSError('Could not apply destination permissions to the installed model.')


def safe_path(name):
    path = PurePosixPath(name)
    if (not name or '\\' in name or ':' in name or path.is_absolute()
            or any(part in ('', '.', '..') for part in name.split('/'))
            or len(path.parts) != 2 or path.parts[0] not in FILES
            or path.parts[1] not in FILES[path.parts[0]]):
        raise ValueError('Bundle contains an unsupported file path.')
    return path


def _https_url(url):
    parsed = urlparse(url)
    if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
        raise ValueError('Model downloads require an HTTPS URL without embedded credentials.')


class HTTPSRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _https_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download(url, destination):
    _https_url(url)
    total = 0
    try:
        with build_opener(HTTPSRedirectHandler()).open(url, timeout=60) as response, destination.open('wb') as output:
            _https_url(response.geturl())
            while content := response.read(1024 * 1024):
                total += len(content)
                if total > MAX_ARCHIVE_BYTES:
                    raise ValueError('Model archive exceeds the download limit.')
                output.write(content)
    except ValueError:
        raise
    except Exception:
        raise ValueError('Model download failed. Check the artifact URL and network access.') from None


def install_bundle(archive, sha256, destination, *, replace=False):
    if not re.fullmatch(r'[0-9a-fA-F]{64}', sha256 or ''):
        raise ValueError('An explicit 64-character SHA-256 digest is required.')
    archive, destination = Path(archive), Path(destination)
    if not archive.is_file() or archive.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError('Model archive is missing or exceeds the size limit.')
    if digest(archive) != sha256.lower():
        raise ValueError('Model archive checksum does not match. No models were installed.')
    destination.mkdir(parents=True, exist_ok=True)
    destination = destination.resolve()
    with tempfile.TemporaryDirectory(prefix='.install-', dir=destination) as temporary:
        stage = Path(temporary)
        with zipfile.ZipFile(archive) as bundle:
            entries = bundle.infolist()
            names = [entry.filename for entry in entries]
            if len(names) > 200 or len(set(names)) != len(names) or 'manifest.json' not in names:
                raise ValueError('Invalid model bundle inventory.')
            if sum(entry.file_size for entry in entries) > MAX_EXPANDED_BYTES:
                raise ValueError('Expanded model bundle exceeds the size limit.')
            manifest_info = bundle.getinfo('manifest.json')
            if manifest_info.file_size > 256 * 1024:
                raise ValueError('Model manifest is too large.')
            manifest = json.loads(bundle.read('manifest.json'))
            records = manifest.get('files')
            if manifest.get('format') != FORMAT or not isinstance(records, dict) or not records:
                raise ValueError('Unsupported model bundle manifest.')
            if set(names) != {'manifest.json', *records}:
                raise ValueError('Archive files do not match the model manifest.')
            models = set()
            for name, record in records.items():
                path = safe_path(name)
                models.add(path.parts[0])
                info = bundle.getinfo(name)
                if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16) or info.flag_bits & 1:
                    raise ValueError('Links, directories and encrypted files are not permitted.')
                if not isinstance(record, dict) or record.get('size') != info.file_size:
                    raise ValueError('Invalid model file size.')
                expected = record.get('sha256', '')
                if not isinstance(expected, str) or not re.fullmatch(r'[0-9a-f]{64}', expected):
                    raise ValueError('Invalid model file checksum.')
                target = stage.joinpath(*path.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                checksum = hashlib.sha256()
                with bundle.open(info) as source, target.open('xb') as output:
                    while content := source.read(1024 * 1024):
                        checksum.update(content)
                        output.write(content)
                if checksum.hexdigest() != expected:
                    raise ValueError('Model file checksum does not match the manifest.')
            declared_models = manifest.get('models')
            if (not isinstance(declared_models, list)
                    or not all(isinstance(model, str) for model in declared_models)
                    or len(set(declared_models)) != len(declared_models)
                    or set(declared_models) != models):
                raise ValueError('Model names do not match the manifest.')
        for model in models:
            target = destination / model
            if target.is_symlink() or (target.exists() and not target.is_dir()):
                raise ValueError('A model destination is not an ordinary directory.')
            if target.exists() and not replace:
                raise ValueError('A model is already installed. Use --replace only after stopping the service.')
        # Verify everything before replacing any installed model; roll back on failure.
        backup = stage / '.backup'
        backup.mkdir()
        moved, installed = [], []
        try:
            for model in sorted(models):
                target = destination / model
                if target.exists():
                    target.rename(backup / model)
                    moved.append(model)
                (stage / model).rename(target)
                installed.append(model)
                _inherit_model_permissions(target, destination)
        except Exception:
            for model in installed:
                target = destination / model
                if target.is_symlink() or target.resolve().parent != destination:
                    raise ValueError('Unsafe rollback destination; installed files were retained.') from None
                shutil.rmtree(target)
            for model in moved:
                (backup / model).rename(destination / model)
            raise
    return {'models': sorted(models), 'files': len(records), 'destination': str(destination)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--archive', help='Trusted local ZIP')
    source.add_argument('--url', default=os.getenv('MODEL_BUNDLE_URL'), help='Public trusted HTTPS artifact URL')
    parser.add_argument('--sha256', default=os.getenv('MODEL_BUNDLE_SHA256'))
    parser.add_argument('--destination', default='models')
    parser.add_argument('--replace', action='store_true')
    parser.add_argument('--if-configured', action='store_true', help='Skip when neither bundle URL nor archive is provided')
    args = parser.parse_args()
    if not args.archive and not args.url:
        if args.if_configured and not args.sha256:
            print('No model bundle configured. Uninstalled modules remain unavailable.')
            return
        parser.error('Provide --archive or --url and --sha256.')
    if not re.fullmatch(r'[0-9a-fA-F]{64}', args.sha256 or ''):
        parser.error('Provide an explicit SHA-256 digest using --sha256 or MODEL_BUNDLE_SHA256.')
    try:
        if args.archive:
            result = install_bundle(args.archive, args.sha256, args.destination, replace=args.replace)
        else:
            with tempfile.TemporaryDirectory(prefix='model-download-') as temporary:
                archive = Path(temporary) / 'models.zip'
                download(args.url, archive)
                result = install_bundle(archive, args.sha256, args.destination, replace=args.replace)
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, zipfile.BadZipFile, json.JSONDecodeError) as error:
        parser.exit(1, str(error) + '\n')


if __name__ == '__main__':
    main()
