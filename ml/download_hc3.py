"""Download author-published HC3 JSON data, never its loader code or inference API."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(ROOT / '.tools/huggingface'))
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')


def main():
    from huggingface_hub import HfApi, snapshot_download
    repository = 'Hello-SimpleAI/HC3'
    revision = HfApi().dataset_info(repository).sha
    output = ROOT / 'data/hc3'
    names = ['reddit_eli5.jsonl', 'wiki_csai.jsonl', 'README.md']
    snapshot_download(repository, repo_type='dataset', revision=revision, local_dir=output,
                      allow_patterns=names, max_workers=2)
    files = {}
    for name in names:
        path = output / name
        files[name] = {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest = {'repository': repository, 'revision': revision, 'files': files,
        'source_url': 'https://huggingface.co/datasets/' + repository,
        'license': 'CC-BY-SA-4.0 subject to upstream terms; selected ELI5 (BSD) and Wikipedia (CC-BY-SA) domains only.'}
    (output/'source-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
    print(json.dumps({'downloaded': list(files), 'revision': revision, 'total_bytes': sum(f['bytes'] for f in files.values())}))


if __name__ == '__main__':
    main()
