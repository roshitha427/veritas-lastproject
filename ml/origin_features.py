"""Shared frozen vision features and deterministic views for train/runtime parity."""
import numpy as np
from PIL import ImageOps

def generation_views(image):
    """Three deterministic views reduce dependence on a photo's border/background."""
    image = ImageOps.exif_transpose(image).convert('RGB')
    width, height = image.size
    def center(scale):
        return image.crop((round(width*(1-scale)/2), round(height*(1-scale)/2),
                           round(width*(1+scale)/2), round(height*(1+scale)/2)))
    detail = center(.7)
    try:
        import cv2
        import numpy as np
        detection = image.copy()
        detection.thumbnail((640, 640))
        ratio = width / detection.width
        gray = cv2.cvtColor(np.asarray(detection), cv2.COLOR_RGB2GRAY)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces):
            x, y, fw, fh = (round(float(v)*ratio) for v in max(faces, key=lambda box: int(box[2])*int(box[3])))
            detail = image.crop((max(0,round(x-fw*.4)), max(0,round(y-fh*.5)),
                                 min(width,round(x+fw*1.4)), min(height,round(y+fh*1.6))))
    except (ImportError, ValueError, RuntimeError):
        pass
    return [image, center(.85), detail]


class OriginEncoder:
    def __init__(self, directory):
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        torch.set_num_threads(min(torch.get_num_threads(), 4))
        self.processor=AutoImageProcessor.from_pretrained(str(directory),local_files_only=True,trust_remote_code=False,use_fast=False)
        self.model=AutoModelForImageClassification.from_pretrained(str(directory),local_files_only=True,
            trust_remote_code=False,use_safetensors=True).vit.eval()

    def extract(self, images):
        import torch
        views=[view for image in images for view in generation_views(image)]
        batches=[]
        for offset in range(0,len(views),4):
            inputs=self.processor(images=views[offset:offset+4],return_tensors='pt')
            with torch.inference_mode():
                embedding=self.model(**inputs).last_hidden_state[:,0,:].numpy()
            embedding=embedding/np.maximum(np.linalg.norm(embedding,axis=1,keepdims=True),1e-8)
            batches.append(embedding)
        values=np.concatenate(batches).reshape(len(images),3,-1)
        return np.concatenate([values.mean(axis=1),values.std(axis=1)],axis=1)
