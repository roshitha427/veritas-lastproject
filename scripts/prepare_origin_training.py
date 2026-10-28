"""Prepare bounded, disjoint public train/validation/test subsets for origin classification."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import time
import requests

DATASET='Rajarshi-Roy-research/Defactify_Image_Dataset'


def main():
    root=Path('data/origin-training')
    root.mkdir(parents=True,exist_ok=True)
    records=[]
    captions={}
    digests={}
    for split,count,start in [('train',600,0),('validation',180,0),('test',180,1000)]:
        rows=[]
        for offset in range(start,start+count,100):
            r=requests.get('https://datasets-server.huggingface.co/rows',params={
                'dataset':DATASET,'config':'default','split':split,'offset':offset,
                'length':min(100,start+count-offset)},timeout=(4,30))
            r.raise_for_status()
            rows.extend(r.json()['rows'])
        folder=root/split;folder.mkdir(exist_ok=True)
        def download(item):
            row=item['row'];index=item['row_idx']
            path=folder/f'{index:05d}.jpg'
            if not path.exists():
                for attempt in range(4):
                    try:
                        response=requests.get(row['Image']['src'],timeout=(4,30))
                        response.raise_for_status()
                        if len(response.content)>20*1024*1024:
                            raise ValueError('Unexpectedly large diagnostic image')
                        path.write_bytes(response.content)
                        break
                    except requests.RequestException:
                        if attempt==3:
                            raise
                        time.sleep(.5)
            return {'path':str(path),'label':int(row['Label_A']),'generator':int(row['Label_B']),
                'caption':row['Caption'],'row':index,'split':split,
                'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'dataset_revision':re.search(r'/--/([a-f0-9]{40})/--/',row['Image']['src']).group(1)}
        with ThreadPoolExecutor(max_workers=6) as executor:
            downloaded=list(executor.map(download,rows))
        kept=[]
        for row in downloaded:
            caption=' '.join(row['caption'].casefold().split())
            if captions.get(caption,split)!=split or digests.get(row['sha256'],split)!=split:
                continue
            captions[caption]=split;digests[row['sha256']]=split
            kept.append(row)
        records.extend(kept)
        print(json.dumps({'split':split,'downloaded':len(downloaded),'retained':len(kept),
            'real':sum(r['label']==0 for r in kept),'generated':sum(r['label']==1 for r in kept)}),flush=True)
    (root/'manifest.json').write_text(json.dumps({'dataset':DATASET,
        'source_url':'https://huggingface.co/datasets/'+DATASET,'labels':{'0':'real','1':'generated'},
        'deduplication':'No caption group or image SHA-256 shared across splits. Private acceptance images excluded.',
        'records':records},indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
