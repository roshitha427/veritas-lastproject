"""Register an audited dataset manifest in the selected database."""
import argparse
import json
from pathlib import Path
from backend.app import create_app
from scripts.check_system import close_checked_app


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',default='data/image-manifest.json')
    args=parser.parse_args()
    manifest=json.loads(Path(args.manifest).read_text(encoding='utf8'))
    if manifest.get('download_status') not in (None, 'complete'):
        raise ValueError('Complete and verify every source archive before registering this dataset version.')
    app=create_app()
    store=app.extensions['store']
    try:
        doc={key:manifest[key] for key in ('name','version','modality','counts')}
        old=store.one('datasets',{'name':doc['name'],'version':doc['version']})
        if old:
            doc['id']=old['id']
        for key in ('archive_sha256', 'limitations', 'provenance', 'license', 'label_meaning', 'task', 'label_schema', 'split_method'):
            if key in manifest:
                doc[key]=manifest[key]
        store.save('datasets',doc)
        print('Dataset version registered in '+store.kind+'.')
    finally:
        close_checked_app(app)
