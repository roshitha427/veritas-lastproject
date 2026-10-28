"""Calibrate and evaluate a completed local BERT training checkpoint.

Use when a saved checkpoint already meets the validation objective. This never
uses test data to select checkpoints or adjust thresholds.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from ml.train_image import metrics


def main(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    checkpoint=Path(args.checkpoint)
    state=json.loads((checkpoint/'trainer_state.json').read_text(encoding='utf8'))
    model=AutoModelForSequenceClassification.from_pretrained(str(checkpoint),local_files_only=True).eval()
    tokenizer=AutoTokenizer.from_pretrained(str(checkpoint),local_files_only=True)
    def infer(path):
        frame=pd.read_csv(path)
        logits=[]
        with torch.inference_mode():
            for offset in range(0,len(frame),args.batch_size):
                inputs=tokenizer(frame['text'].iloc[offset:offset+args.batch_size].tolist(),padding=True,truncation=True,max_length=args.max_length,return_tensors='pt')
                output=model(**inputs).logits
                logits.extend((output[:,1]-output[:,0]).tolist())
        return np.array(logits),frame['label'].map({'real':0,'fake':1}).to_numpy()
    val_logits,val_y=infer(args.validation)
    sigmoid=lambda value:1/(1+np.exp(-np.clip(value,-30,30)))
    temperature=float(minimize_scalar(lambda temp:log_loss(val_y,sigmoid(val_logits/temp),labels=[0,1]),bounds=(.1,10),method='bounded').x)
    print('Calibration complete; evaluating held-out test articles.',flush=True)
    logits,labels=infer(args.test)
    evaluation=metrics(labels,sigmoid(logits/temperature))
    manifest=json.loads(Path(args.validation).with_name('manifest.json').read_text(encoding='utf8'))
    card=dict(name='Fine-tuned compact BERT news classifier',version='1.0-'+manifest['version'],fake_label='fake',
        evaluation=evaluation,base_model='prajjwal1/bert-tiny',temperature=temperature,max_length=args.max_length,
        target_accuracy=.85,target_met=evaluation['accuracy']>=.85,languages=['en'],status='experimental',
        selected_checkpoint=checkpoint.name,completed_training_epochs=state['epoch'],seed=42,
        calibration='Validation-set temperature scaling; test not used for fitting or checkpoint selection.',
        training_history=state['log_history'],dataset=manifest,
        limitations=manifest['limitations']+['Hindi and Telugu classifier performance is unvalidated; the app abstains for unsupported scripts.',
            'Article-level historical classification does not establish current claim truth or AI generation.'])
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    model.save_pretrained(str(output),safe_serialization=True)
    tokenizer.save_pretrained(str(output))
    (output/'model-card.json').write_text(json.dumps(card,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in card.items() if k not in ('training_history','dataset')},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--validation',default='data/text/validation.csv')
    parser.add_argument('--test',default='data/text/test.csv')
    parser.add_argument('--output',default='models/text')
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--max-length',type=int,default=192)
    main(parser.parse_args())
