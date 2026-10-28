"""Package trusted inference files, excluding datasets and training checkpoints."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile


FORMAT = 'sr-inference-bundle-v1'
FILES = {
    'text': ('config.json', 'model-card.json', 'model.safetensors', 'pytorch_model.bin',
             'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'vocab.txt'),
    'image-v2': ('model.joblib', 'backbone.pt', 'model-card.json'),
    'ai-text': ('model.joblib', 'model-card.json'),
    'video-temporal': ('model.joblib', 'model-card.json'),
    'ai-image': ('model.joblib', 'backbone.pt', 'classifier.joblib', 'model-card.json',
                 'config.json', 'model.safetensors', 'preprocessor_config.json'),
    'evidence-nli': ('config.json', 'model-card.json', 'model.safetensors', 'pytorch_model.bin',
                     'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
                     'added_tokens.json', 'spm.model'),
    'whisper-tiny': ('config.json', 'model.safetensors', 'pytorch_model.bin', 'generation_config.json',
                     'preprocessor_config.json', 'tokenizer.json', 'tokenizer_config.json',
                     'special_tokens_map.json', 'added_tokens.json', 'normalizer.json',
                     'vocab.json', 'merges.txt'),
}
CORE = ('text', 'image-v2', 'ai-text', 'video-temporal', 'ai-image')
ACTIVE = {'text': 'text-continued-20261020',
          'ai-image': 'image-origin-artwork-20261021',
          'video-temporal': 'video-temporal-retrained-20261020'}


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def export_bundle(source, output, *, profile='core', mappings=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_dir():
        raise ValueError('The source must be a trusted models directory.')
    if output.exists():
        raise ValueError('Output already exists. Choose a new archive path.')
    names = CORE if profile == 'core' else tuple(FILES)
    selected, records = [], {}
    for alias in names:
        configured = (mappings or {}).get(alias)
        default = ACTIVE.get(alias, alias)
        directory = source / (configured or (default if (source / default).is_dir() else alias))
        directory = directory.resolve()
        if not directory.is_relative_to(source) or not directory.is_dir():
            raise ValueError('Missing model directory for ' + alias)
        included = [directory / name for name in FILES[alias] if (directory / name).is_file()]
        found = {path.name for path in included}
        if alias in ('image-v2', 'ai-text', 'video-temporal'):
            required = set(FILES[alias])
        elif alias == 'ai-image':
            required = {'model-card.json'}
            if not ({'model.joblib', 'backbone.pt'} <= found or 'model.safetensors' in found or 'classifier.joblib' in found):
                raise ValueError('Missing inference weights for ' + alias)
        else:
            required = {'config.json', 'tokenizer_config.json'}
            if alias != 'whisper-tiny':
                required.add('model-card.json')
            if not found & {'model.safetensors', 'pytorch_model.bin'}:
                raise ValueError('Missing inference weights for ' + alias)
        if not required <= found:
            raise ValueError('Missing inference files for ' + alias + ': ' + ', '.join(sorted(required - found)))
        for path in included:
            if path.is_symlink() or not path.resolve().is_relative_to(source):
                raise ValueError('Model files must be ordinary files inside the source directory.')
            name = alias + '/' + path.name
            records[name] = {'sha256': digest(path), 'size': path.stat().st_size}
            selected.append((name, path))
    manifest = {'format': FORMAT, 'created_at': datetime.now(timezone.utc).isoformat(),
                'profile': profile, 'models': list(names), 'files': records}
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
            bundle.writestr('manifest.json', json.dumps(manifest, indent=2))
            for name, path in selected:
                bundle.write(path, name)
    except Exception:
        output.unlink(missing_ok=True)
        raise
    return {'archive': str(output), 'sha256': digest(output), 'models': list(names),
            'files': len(records), 'bytes': output.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='Trusted source models directory, such as ../SR/models')
    parser.add_argument('--output', required=True, help='New output ZIP; keep it outside Git')
    parser.add_argument('--profile', choices=('core', 'full'), default='core',
                        help='Core includes active text/image/origin/temporal/AI-writing; full adds NLI and speech')
    parser.add_argument('--text', help='Source directory name mapped to text/')
    parser.add_argument('--origin', help='Source directory name mapped to ai-image/')
    parser.add_argument('--temporal', help='Source directory name mapped to video-temporal/')
    args = parser.parse_args()
    mappings = {name: value for name, value in (
        ('text', args.text), ('ai-image', args.origin), ('video-temporal', args.temporal)) if value}
    try:
        print(json.dumps(export_bundle(args.source, args.output, profile=args.profile, mappings=mappings), indent=2))
    except (ValueError, OSError, zipfile.BadZipFile) as error:
        parser.exit(1, str(error) + '\n')


if __name__ == '__main__':
    main()
