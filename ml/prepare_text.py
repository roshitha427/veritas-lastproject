"""Prepare the public ISOT corpus, removing direct publisher markers and overlap."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.text_cleaning import clean_text


def main(args):
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.archive) as archive:
        frames=[]
        for name,label in [('Fake.csv','fake'),('True.csv','real')]:
            member=next(n for n in archive.namelist() if n.endswith(name))
            with archive.open(member) as handle:
                frame=pd.read_csv(handle)
            frame['label']=label
            frames.append(frame)
    frame=pd.concat(frames,ignore_index=True).dropna(subset=['text'])
    raw_count=len(frame)
    frame['text']=frame['text'].map(clean_text)
    frame=frame[frame['text'].str.len()>=200].copy()
    # Same leading passage must never appear in multiple splits (model truncates long articles).
    frame['key']=frame['text'].map(lambda text:hashlib.sha256(' '.join(re.findall(r'\w+',text.lower())[:140]).encode()).hexdigest())
    conflicts=frame.groupby('key')['label'].nunique()
    frame=frame[~frame['key'].isin(conflicts[conflicts>1].index)].drop_duplicates('key')
    train,remainder=train_test_split(frame,test_size=.30,stratify=frame['label'],random_state=42)
    validation,test=train_test_split(remainder,test_size=.5,stratify=remainder['label'],random_state=42)
    # Reduce near-duplicate leakage using character-gram MinHash-like signatures is deferred;
    # corpus limitations are recorded explicitly instead of claiming independent publishers.
    counts={}
    for split,rows in [('train',train),('validation',validation),('test',test)]:
        rows[['text','label']].to_csv(out/(split+'.csv'),index=False)
        counts[split]=len(rows)
    with open(args.archive,'rb') as handle:
        digest=hashlib.file_digest(handle,'sha256').hexdigest()
    manifest={'name':'ISOT fake/real news (Kaggle mirror)','version':digest[:12],'modality':'text','counts':counts,
        'source':'https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset',
        'original_source':'https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/',
        'archive_sha256':digest,'license':'CC BY-NC-SA 4.0 (Kaggle mirror data card)',
        'raw_rows':raw_count,'removed_short_duplicate_or_conflicting_rows':raw_count-len(frame),
        'languages':['en'],'preprocessing':'Body only, no subject/date/title; Reuters dateline, publisher marker, URLs and HTML removed; normalized leading 140-word passage deduplicated before splitting.',
        'limitations':['English historical news corpus; not a benchmark for current truth, Hindi, Telugu, or AI-generated writing.',
                       'Random article split is not publisher-, time-, or event-independent. Residual style/topic/near-duplicate leakage may inflate test accuracy.']}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--archive',default='data/isot.zip')
    parser.add_argument('--output',default='data/text')
    main(parser.parse_args())
