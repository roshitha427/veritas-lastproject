"""Trust boundaries for moving trained inference artifacts between machines."""
import hashlib
import json
import os
from pathlib import Path
import stat
import zipfile

import pytest

from scripts.export_models import CORE, FORMAT, digest, export_bundle
from scripts.install_models import install_bundle


def bundle(tmp_path, files, *, wrong_file_hash=False, symlink=False):
    records = {name: {'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
               for name, data in files.items()}
    if wrong_file_hash:
        next(iter(records.values()))['sha256'] = '0' * 64
    archive = tmp_path / 'bundle.zip'
    with zipfile.ZipFile(archive, 'w') as output:
        output.writestr('manifest.json', json.dumps({'format': FORMAT,
            'models': sorted({name.split('/')[0] for name in files}), 'files': records}))
        for name, data in files.items():
            info = zipfile.ZipInfo(name)
            if symlink:
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
            output.writestr(info, data)
    return archive


def test_installs_without_deserializing_trusted_artifacts(tmp_path):
    archive = bundle(tmp_path, {'ai-text/model.joblib': b'opaque model bytes'})
    destination = tmp_path / 'models'
    result = install_bundle(archive, digest(archive), destination)
    assert result['models'] == ['ai-text']
    assert (destination / 'ai-text/model.joblib').read_bytes() == b'opaque model bytes'


def test_rejects_wrong_archive_hash_before_touching_models(tmp_path):
    archive = bundle(tmp_path, {'ai-text/model.joblib': b'model'})
    destination = tmp_path / 'models'
    with pytest.raises(ValueError, match='archive checksum'):
        install_bundle(archive, '0' * 64, destination)
    assert not destination.exists()


@pytest.mark.parametrize('name', ['../outside.joblib', 'ai-text/../../outside.joblib',
                                 '/ai-text/model.joblib', 'ai-text\\model.joblib'])
def test_rejects_path_traversal(tmp_path, name):
    archive = bundle(tmp_path, {name: b'model'})
    with pytest.raises(ValueError, match='unsupported file path|files do not match'):
        install_bundle(archive, digest(archive), tmp_path / 'models')
    assert not (tmp_path / 'outside.joblib').exists()


def test_rejects_symlinks(tmp_path):
    archive = bundle(tmp_path, {'ai-text/model.joblib': b'outside'}, symlink=True)
    with pytest.raises(ValueError, match='Links'):
        install_bundle(archive, digest(archive), tmp_path / 'models')


def test_validates_entire_bundle_before_replacing_models(tmp_path):
    destination = tmp_path / 'models'
    (destination / 'ai-text').mkdir(parents=True)
    original = destination / 'ai-text/model.joblib'
    original.write_bytes(b'previous version')
    archive = bundle(tmp_path, {'ai-text/model.joblib': b'bad version'}, wrong_file_hash=True)
    with pytest.raises(ValueError, match='file checksum'):
        install_bundle(archive, digest(archive), destination, replace=True)
    assert original.read_bytes() == b'previous version'
    assert list(destination.iterdir()) == [original.parent]


def test_replace_requires_explicit_option(tmp_path):
    destination = tmp_path / 'models'
    (destination / 'ai-text').mkdir(parents=True)
    original = destination / 'ai-text/model.joblib'
    original.write_bytes(b'old')
    archive = bundle(tmp_path, {'ai-text/model.joblib': b'new'})
    with pytest.raises(ValueError, match='already installed'):
        install_bundle(archive, digest(archive), destination)
    assert original.read_bytes() == b'old'
    install_bundle(archive, digest(archive), destination, replace=True)
    assert original.read_bytes() == b'new'


def test_export_uses_active_aliases_and_excludes_training_files(tmp_path):
    source = tmp_path / 'source'
    mappings = {'text': 'text-continued-20261020', 'ai-image': 'image-origin-artwork-20261021',
                'video-temporal': 'video-temporal-retrained-20261020'}
    for alias in CORE:
        directory = source / mappings.get(alias, alias)
        directory.mkdir(parents=True)
        names = {'text': ('config.json', 'tokenizer_config.json', 'model-card.json', 'model.safetensors'),
                 'ai-image': ('model.joblib', 'backbone.pt', 'model-card.json'),
                 'image-v2': ('model.joblib', 'backbone.pt', 'model-card.json'),
                 'ai-text': ('model.joblib', 'model-card.json'),
                 'video-temporal': ('model.joblib', 'model-card.json')}[alias]
        for name in names:
            (directory / name).write_bytes(b'inference')
        (directory / 'checkpoint-1').mkdir()
        (directory / 'checkpoint-1/optimizer.pt').write_bytes(b'training')
        (directory / 'training-manifest.json').write_bytes(b'private paths')
    archive = tmp_path / 'export.zip'
    result = export_bundle(source, archive)
    assert result['sha256'] == digest(archive)
    with zipfile.ZipFile(archive) as exported:
        assert 'ai-image/backbone.pt' in exported.namelist()
        assert not any('checkpoint' in name or 'training-manifest' in name for name in exported.namelist())
    install_bundle(archive, result['sha256'], tmp_path / 'installed')


@pytest.mark.skipif(os.name != 'nt', reason='Windows ACL inheritance regression')
def test_installed_models_inherit_destination_access_on_windows(tmp_path):
    import ctypes
    from ctypes import wintypes

    def protected_dacl(path):
        security = ctypes.WinDLL('advapi32', use_last_error=True)
        get_security = security.GetFileSecurityW
        get_security.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, ctypes.c_void_p,
                                 wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
        get_security.restype = wintypes.BOOL
        get_control = security.GetSecurityDescriptorControl
        get_control.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.WORD),
                                ctypes.POINTER(wintypes.DWORD)]
        get_control.restype = wintypes.BOOL
        needed = wintypes.DWORD()
        get_security(str(path), 4, None, 0, ctypes.byref(needed))
        descriptor = ctypes.create_string_buffer(needed.value)
        if not get_security(str(path), 4, descriptor, needed.value, ctypes.byref(needed)):
            raise ctypes.WinError(ctypes.get_last_error())
        control, revision = wintypes.WORD(), wintypes.DWORD()
        if not get_control(descriptor, ctypes.byref(control), ctypes.byref(revision)):
            raise ctypes.WinError(ctypes.get_last_error())
        return bool(control.value & 0x1000)

    archive = bundle(tmp_path, {'ai-text/model.joblib': b'usable by the destination owner'})
    destination = tmp_path / 'models'
    install_bundle(archive, digest(archive), destination)
    assert not protected_dacl(destination / 'ai-text')
    assert not protected_dacl(destination / 'ai-text/model.joblib')
    assert (destination / 'ai-text/model.joblib').read_bytes() == b'usable by the destination owner'


def test_permission_failure_restores_previous_models_and_removes_new_models(tmp_path, monkeypatch):
    import scripts.install_models as installer
    destination = tmp_path / 'models'
    previous = destination / 'ai-text/model.joblib'
    previous.parent.mkdir(parents=True)
    previous.write_bytes(b'previous verified model')
    archive = bundle(tmp_path, {'ai-text/model.joblib': b'replacement',
                                'image-v2/model.joblib': b'new model'})
    apply_permissions = installer._inherit_model_permissions

    def failing_permissions(target, root):
        if target.name == 'image-v2':
            raise OSError('Synthetic filesystem permission failure')
        apply_permissions(target, root)

    monkeypatch.setattr(installer, '_inherit_model_permissions', failing_permissions)
    with pytest.raises(OSError, match='permission failure'):
        install_bundle(archive, digest(archive), destination, replace=True)
    assert previous.read_bytes() == b'previous verified model'
    assert list(destination.iterdir()) == [previous.parent]
