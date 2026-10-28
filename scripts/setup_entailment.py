"""Download a pinned public evidence-comparison model; never upload project data."""
import hashlib
import json
from pathlib import Path

REPOSITORY = 'cross-encoder/nli-deberta-v3-small'
REVISION = 'fa2804872c3b4bd748f38c0185cc85775361e735'


def main():
    from huggingface_hub import snapshot_download
    output = Path('models/evidence-nli')
    output.mkdir(parents=True, exist_ok=True)
    snapshot_download(REPOSITORY, revision=REVISION, local_dir=str(output), token=False,
        allow_patterns=['config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json',
                        'special_tokens_map.json', 'added_tokens.json', 'spm.model', 'README.md'])
    hashes = {}
    for name in ('config.json', 'model.safetensors', 'tokenizer.json'):
        with (output / name).open('rb') as handle:
            hashes[name] = hashlib.file_digest(handle, 'sha256').hexdigest()
    card = {'name': 'Local DeBERTa evidence comparison', 'version': REVISION,
            'source_url': 'https://huggingface.co/' + REPOSITORY, 'source_revision': REVISION,
            'license': 'Apache-2.0', 'languages': ['en'], 'sha256': hashes,
            'labels': ['contradiction', 'entailment', 'neutral'],
            'scope': 'Compare retrieved passages against a claim; not standalone factual knowledge.',
            'status': 'experimental', 'local_evaluation': None}
    (output / 'model-card.json').write_text(json.dumps(card, indent=2), encoding='utf-8')
    print('Pinned evidence-comparison model installed in models/evidence-nli')


if __name__ == '__main__':
    main()
