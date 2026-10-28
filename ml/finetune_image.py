"""Fine-tune visual CNN and calibrate it on validation data."""
import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import random
import time
import zipfile
import joblib
import numpy as np
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from ml.train_image import metrics


class ArchiveImages(Dataset):
    def __init__(self, archive, entries, augment=False):
        self.archive=zipfile.ZipFile(archive)
        self.entries=entries
        steps=[transforms.Resize((128,128))]
        if augment:
            steps.append(transforms.RandomHorizontalFlip())
        steps.extend([transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
        self.transform=transforms.Compose(steps)
    def __len__(self):return len(self.entries)
    def __getitem__(self, index):
        item=self.entries[index]
        with Image.open(BytesIO(self.archive.read(item['member']))) as image:
            tensor=self.transform(ImageOps.exif_transpose(image).convert('RGB'))
        return tensor,int(item['label']=='fake')


def infer(model, loader, horizontal_flip_average=False):
    model.eval()
    logits,labels=[],[]
    with torch.inference_mode():
        for x,y in loader:
            output=model(x)
            score=output[:,1]-output[:,0]
            if horizontal_flip_average:
                flipped=model(torch.flip(x,dims=[3]))
                flipped_score=flipped[:,1]-flipped[:,0]
                score=(score+flipped_score)/2
            logits.extend(score.tolist())
            labels.extend(y.tolist())
    return np.array(logits),np.array(labels)


def main(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    np.random.seed(42)
    manifest=json.loads(Path(args.manifest).read_text(encoding='utf8'))
    if manifest.get('archive_sha256'):
        with open(args.archive,'rb') as handle:
            if hashlib.file_digest(handle,'sha256').hexdigest() != manifest['archive_sha256']:
                raise ValueError('Archive fingerprint differs from the audited manifest.')
    pools={split:[e for e in manifest['entries'] if e['split']==split] for split in ('train','validation','test')}
    random.Random(42).shuffle(pools['validation'])
    if args.validation:
        pools['validation']=pools['validation'][:args.validation]
    if all(row.get('source_group') for row in pools['validation']):
        calibration_mask=np.array([int(hashlib.sha256(row['source_group'].encode()).hexdigest()[:8],16)%2==0
                                   for row in pools['validation']])
        tune,cal=np.flatnonzero(~calibration_mask),np.flatnonzero(calibration_mask)
    else:
        tune,cal=np.arange(0,len(pools['validation']),2),np.arange(1,len(pools['validation']),2)
    datasets={s:ArchiveImages(args.archive,e,augment=s=='train') for s,e in pools.items()}
    loaders={s:DataLoader(d,batch_size=args.batch_size,shuffle=s=='train',num_workers=0) for s,d in datasets.items()}
    model=mobilenet_v3_small(weights=None if args.backbone else MobileNet_V3_Small_Weights.DEFAULT)
    if args.backbone:
        model.load_state_dict(torch.load(args.backbone,map_location='cpu',weights_only=True))
    model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,2)
    # Keep early general-purpose filters fixed; adapt later filters and the head.
    for p in model.features[:6].parameters():p.requires_grad=False
    optimizer=torch.optim.AdamW([{'params':model.features[6:].parameters(),'lr':2e-4},
                                 {'params':model.classifier.parameters(),'lr':1e-3}],weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=1e-5)
    loss_fn=nn.CrossEntropyLoss()
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    best=-1
    started=time.monotonic()
    history=[]
    for epoch in range(args.epochs):
        model.train()
        model.features[:6].eval()
        running=correct=count=0
        for step,(x,y) in enumerate(loaders['train']):
            optimizer.zero_grad(set_to_none=True)
            logits=model(x)
            loss=loss_fn(logits,y)
            loss.backward()
            optimizer.step()
            running+=loss.item()*len(y)
            correct+=(logits.argmax(1)==y).sum().item()
            count+=len(y)
            if step%100==0:
                print(f'epoch {epoch+1}/{args.epochs} | {count}/{len(datasets["train"])} | loss {running/count:.4f} | train accuracy {correct/count:.3f} | {time.monotonic()-started:.0f}s',flush=True)
        scheduler.step()
        val_logits,val_y=infer(model,loaders['validation'])
        score=float(balanced_accuracy_score(val_y[tune],val_logits[tune]>=0))
        history.append({'epoch':epoch+1,'validation_balanced_accuracy':score,'train_loss':running/count})
        print(json.dumps(history[-1]),flush=True)
        if score>best:
            best=score
            torch.save(model.state_dict(),output/'backbone.pt')
    model.load_state_dict(torch.load(output/'backbone.pt',map_location='cpu',weights_only=True))
    val_logits,val_y=infer(model,loaders['validation'],horizontal_flip_average=True)
    calibrator=LogisticRegression(C=1e6,solver='lbfgs',max_iter=1000).fit(val_logits[cal,None],val_y[cal])
    slope=float(calibrator.coef_[0,0])
    intercept=float(calibrator.intercept_[0])
    sigmoid=lambda x:1/(1+np.exp(-np.clip(x,-30,30)))
    test_logits,test_y=infer(model,loaders['test'],horizontal_flip_average=True)
    evaluation=metrics(test_y,sigmoid(slope*test_logits+intercept))
    card=dict(name='Fine-tuned MobileNetV3 visual classifier',version='2.1-'+manifest['version'],
        architecture='Fine-tuned MobileNetV3 Small CNN with horizontal-flip averaging and Platt calibration',
        scope='Dataset-specific visual fake/real classification. Not an assessment of news truth, all manipulation types, or AI generation.',
        dataset_version=manifest['version'],dataset_group=manifest['group'],evaluation=evaluation,
        split_counts={s:len(d) for s,d in datasets.items()},
        inference={'test_time_augmentation':'horizontal_flip_average','calibration':'platt'},
        validation_calibration={'method':'platt','slope':slope,'intercept':intercept,'examples':len(cal)},seed=42,
        target_accuracy=.85,target_met=evaluation['accuracy']>=.85,status='experimental',input_size=128,
        training_history=history,selected_validation_balanced_accuracy=best,runtime_seconds=round(time.monotonic()-started),
        limitations=manifest['limitations']+([
            'Source-group test split created from the supplied CASIA data; independent external evaluation is still required.']
            if manifest['group']=='CASIA2' else [
            'This test split was previously used to report an initial baseline. CNN selection used validation only; an independent external test is still required.']))
    joblib.dump({'kind':'cnn','calibration':{'type':'platt','slope':slope,'intercept':intercept},
                 'tta':'horizontal_flip_average','card':card},output/'model.joblib')
    (output/'model-card.json').write_text(json.dumps(card,indent=2),encoding='utf8')
    print(json.dumps(card,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--archive',required=True)
    parser.add_argument('--backbone',help='Trusted local ImageNet backbone; avoids another network download.')
    parser.add_argument('--manifest',default='data/image-manifest.json')
    parser.add_argument('--output',default='models/image-v2')
    parser.add_argument('--epochs',type=int,default=5)
    parser.add_argument('--validation',type=int,default=3000)
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--threads',type=int,default=4)
    main(parser.parse_args())
