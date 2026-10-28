"""Preview audited local dataset metadata, or register it with --apply. Never trains models."""
import argparse
import json
from pathlib import Path

from backend.dataset_registry import discover_local_datasets, reconcile_dataset_registry

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='Save metadata in the configured database; preserves custom records.')
    args = parser.parse_args()
    records, warnings = discover_local_datasets(ROOT)
    if not args.apply:
        print(json.dumps({'mode': 'preview', 'datasets': records, 'warnings': warnings}, indent=2))
        return 0
    from backend.app import create_app
    app = create_app()
    store = app.extensions['store']
    try:
        stats = reconcile_dataset_registry(store, records, apply=True)
        print(json.dumps({'mode': 'applied', 'datasets_discovered': len(records), **stats, 'warnings': warnings,
                          'storage': {'mode': store.kind}}, indent=2))
    finally:
        app.extensions['analysis_executor'].shutdown(wait=True)
        app.extensions['mail_executor'].shutdown(wait=True)
        store.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
