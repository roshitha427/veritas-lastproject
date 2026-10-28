"""Transfer learning baseline, calibration, and untouched held-out evaluation.

Uses a frozen ImageNet MobileNetV3 CNN feature extractor and a fitted binary head.
No claim of attaining target accuracy is made until the metrics are measured.
"""
import argparse
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import random
import time
import zipfile
import joblib
import numpy as np
from PIL import Image, ImageOps
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, confusion_matrix, f1_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def metrics(y, p):
    predictions = p >= .5
    n = len(y)
    accuracy = accuracy_score(y, predictions)
    z = 1.96
    center = (accuracy + z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(accuracy*(1-accuracy)/n+z*z/(4*n*n))/(1+z*z/n)
    ece = 0.
    confidence = np.maximum(p, 1-p)
    for low in np.arange(.5, 1, .05):
        mask = (confidence >= low) & (confidence < low + .05 + 1e-10)
        if mask.any():
            ece += mask.mean() * abs(np.mean(predictions[mask] == y[mask]) - confidence[mask].mean())
    return dict(accuracy=float(accuracy), balanced_accuracy=float(balanced_accuracy_score(y, predictions)),
        f1=float(f1_score(y, predictions)), roc_auc=float(roc_auc_score(y, p)), brier_score=float(brier_score_loss(y, p)),
        expected_calibration_error=float(ece), confusion_matrix=confusion_matrix(y, predictions).tolist(),
        labels=['real', 'fake'], accuracy_95_percent_interval=[float(center-half), float(center+half)], test_count=n)


def main(args):
    import torch
    from torchvision import transforms
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    started = time.time()
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(args.threads)
    manifest = json.loads(Path(args.manifest).read_text(encoding='utf8'))
    with open(args.archive, 'rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    if digest != manifest['archive_sha256']:
        raise ValueError('Archive fingerprint differs from the audited manifest.')
    selected = {}
    for split, cap in [('train', args.train), ('validation', args.validation), ('test', args.test)]:
        entries = []
        for label in ('real', 'fake'):
            pool = [e for e in manifest['entries'] if e['split'] == split and e['label'] == label]
            random.Random(42).shuffle(pool)
            entries.extend(pool[:cap//2] if cap else pool)
        random.Random(42).shuffle(entries)
        selected[split] = entries
    if any(len(v)<20 for v in selected.values()):
        raise ValueError('Too few independent examples in one or more splits.')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}; split sizes: ' + str({k:len(v) for k,v in selected.items()}), flush=True)
    model = mobilenet_v3_small(weights=None if args.backbone else MobileNet_V3_Small_Weights.DEFAULT)
    if args.backbone:
        state = torch.load(args.backbone, map_location='cpu', weights_only=True)
        model.load_state_dict({key:value for key,value in state.items() if not key.startswith('classifier.')}, strict=False)
    torch.save(model.state_dict(), output / 'backbone.pt')
    model.classifier = torch.nn.Identity()
    model.to(device).eval()
    transform = transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    base_digest = hashlib.sha256(Path(args.backbone).read_bytes()).hexdigest() if args.backbone else 'imagenet-default'
    fingerprint = hashlib.sha256((json.dumps(selected,sort_keys=True) + base_digest).encode()).hexdigest()
    cachefile = output / f'features-{fingerprint[:12]}.npz'
    if cachefile.exists():
        loaded = np.load(cachefile)
        features = {split:(loaded[split+'_x'],loaded[split+'_y']) for split in selected}
        print('Using matching feature cache', flush=True)
    else:
        features = {}
        with zipfile.ZipFile(args.archive) as archive:
            for split, entries in selected.items():
                vectors, labels = [], []
                for offset in range(0,len(entries),args.batch_size):
                    if offset % (args.batch_size * 64) == 0:
                        print(f'Extracting {split}: {offset}/{len(entries)}', flush=True)
                    batch = entries[offset:offset+args.batch_size]
                    tensors = []
                    for item in batch:
                        with Image.open(BytesIO(archive.read(item['member']))) as image:
                            tensors.append(transform(ImageOps.exif_transpose(image).convert('RGB')))
                        labels.append(int(item['label']=='fake'))
                    with torch.inference_mode():
                        vectors.append(model(torch.stack(tensors).to(device)).cpu().numpy())
                    if offset % (args.batch_size*10)==0:
                        print(f'{split}: {offset+len(batch)}/{len(entries)} | {time.time()-started:.0f}s', flush=True)
                features[split]=(np.concatenate(vectors),np.array(labels))
        np.savez_compressed(cachefile, **{f'{s}_{k}':a for s,(x,y) in features.items() for k,a in [('x',x),('y',y)]})
    x_train,y_train = features['train']
    x_val,y_val = features['validation']
    transfer_audit = None
    if args.backbone_manifest:
        prior = json.loads(Path(args.backbone_manifest).read_text(encoding='utf8'))
        prior_train = {row['sha256'] for row in prior['entries'] if row['split'] == 'train'}
        test_overlap = sum(row['sha256'] in prior_train for row in selected['test'])
        if test_overlap:
            raise ValueError('The transferred backbone has seen test images; use an independent backbone or test split.')
        independent = np.array([row['sha256'] not in prior_train for row in selected['validation']])
        x_val, y_val = x_val[independent], y_val[independent]
        transfer_audit = {'prior_training_manifest': args.backbone_manifest,
                          'test_training_sha_overlap': test_overlap,
                          'validation_training_overlap_excluded': int((~independent).sum()),
                          'validation_used': len(y_val)}
    # Validation half for head selection; other half for temperature calibration.
    tune,cal = np.arange(0,len(y_val),2),np.arange(1,len(y_val),2)
    best, best_score = None,-1
    for c in (.001,.01,.1,1,10):
        clf = make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=2000,random_state=42,class_weight='balanced'))
        with threadpool_limits(limits=args.threads):
            clf.fit(x_train,y_train)
        score = balanced_accuracy_score(y_val[tune],clf.predict(x_val[tune]))
        print(f'C={c}, validation balanced accuracy={score:.4f}',flush=True)
        if score > best_score:
            best,best_score=clf,score
    logits = best.decision_function(x_val[cal])
    sigmoid = lambda x: 1/(1+np.exp(-np.clip(x,-30,30)))
    fit = minimize_scalar(lambda temp:log_loss(y_val[cal],sigmoid(logits/temp),labels=[0,1]), bounds=(.1,10),method='bounded')
    temperature=float(fit.x)
    # Test is first used here, after fitting and calibration are complete.
    x_test,y_test=features['test']
    evaluation=metrics(y_test,sigmoid(best.decision_function(x_test)/temperature))
    card=dict(name='MobileNetV3 + calibrated classification head',version='1.1-group'+str(manifest['group'])+'-'+manifest['version'],
        architecture='Frozen pretrained MobileNetV3 Small CNN + StandardScaler + LogisticRegression',
        base_backbone_sha256=base_digest,
        transfer_audit=transfer_audit,
        scope='Dataset-specific visual fake/real classification. Not an assessment of news truth, all manipulation types, or AI generation.',
        dataset_version=manifest['version'],dataset_group=manifest['group'],seed=42,
        split_counts={k:len(v) for k,v in selected.items()},evaluation=evaluation,
        temperature=temperature,target_accuracy=.85,target_met=evaluation['accuracy']>=.85,
        limitations=manifest['limitations'],runtime_seconds=round(time.time()-started),
        status='experimental', input_size=128)
    joblib.dump({'classifier':best,'temperature':temperature,'card':card},output/'model.joblib')
    (output/'model-card.json').write_text(json.dumps(card,indent=2),encoding='utf8')
    print(json.dumps(card,indent=2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--archive',required=True)
    parser.add_argument('--backbone',help='Optional trusted local backbone to transfer from; classifier weights are refitted.')
    parser.add_argument('--backbone-manifest',help='Prior backbone training manifest, used to exclude validation overlap and reject test overlap.')
    parser.add_argument('--manifest',default='data/image-manifest.json')
    parser.add_argument('--output',default='models/image')
    parser.add_argument('--train',type=int,default=8000,help='Total across both classes; 0 uses all')
    parser.add_argument('--validation',type=int,default=2000)
    parser.add_argument('--test',type=int,default=2000)
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--threads',type=int,default=4)
    main(parser.parse_args())
