"""Learned video consistency from shared sampled frames, not an audio deepfake detector."""
from pathlib import Path
import threading

import cv2
import numpy as np
from PIL import Image, ImageOps

FEATURE_VERSION = 'imagenet-face-consistency-v3'


def _sequence_crops(images):
    frames = [cv2.resize(np.asarray(ImageOps.exif_transpose(image).convert('RGB')), (256,256),
                         interpolation=cv2.INTER_AREA) for image in images]
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    boxes = []
    for index in sorted({0, len(frames)//2, len(frames)-1}):
        gray = cv2.cvtColor(frames[index], cv2.COLOR_RGB2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(40,40))
        if len(faces):
            boxes.append(max(faces, key=lambda box: int(box[2])*int(box[3])))
    if boxes:
        x, y, width, height = np.rint(np.median(boxes, axis=0)).astype(int)
        pad = int(max(width, height) * .12)
        frames = [rgb[max(0,y-pad):min(256,y+height+pad), max(0,x-pad):min(256,x+width+pad)] for rgb in frames]
    # One shared median crop avoids detector jitter being mistaken for motion.
    return [cv2.resize(rgb, (96,96), interpolation=cv2.INTER_AREA) for rgb in frames]


def temporal_features(images, frame_probabilities, crops=None):
    if not 3 <= len(images) <= 24 or len(images) != len(frame_probabilities):
        raise ValueError('Temporal analysis requires 3–24 ordered frames and matching visual scores.')
    rows, gray_frames = [], []
    for image, score in zip(crops if crops is not None else _sequence_crops(images), frame_probabilities):
        rgb = image.astype(np.float32) / 255
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray_frames.append(gray)
        row = []
        for channel in np.moveaxis(rgb, -1, 0):
            row.extend([channel.mean(), channel.std(), *np.quantile(channel, [.1, .5, .9])])
        laplace = cv2.Laplacian(gray, cv2.CV_32F)
        gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        magnitude = np.sqrt(gx*gx + gy*gy)
        residual = gray - cv2.GaussianBlur(gray, (5, 5), 1)
        row.extend([np.log1p(laplace.var()*1000), np.abs(laplace).mean(), magnitude.mean(), magnitude.std(),
                    np.quantile(magnitude, .9), residual.std(), np.abs(residual).mean(), float(score)])
        spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray))) ** 2)
        spectrum[48, 48] = 0
        row.extend(spectrum.reshape(8, 12, 8, 12).mean(axis=(1, 3)).ravel())
        rows.append(row)
    rows = np.asarray(rows, dtype=np.float32)
    differences = np.diff(rows, axis=0)
    # Ordered adjacent-frame differences provide a temporal signal. Source IDs,
    # codec metadata, duration and frame count are excluded from the features.
    vectors = [rows.mean(axis=0), rows.std(axis=0), np.quantile(rows, .1, axis=0),
               np.quantile(rows, .9, axis=0), np.abs(differences).mean(axis=0), differences.std(axis=0)]
    flicker = []
    for previous, current in zip(gray_frames, gray_frames[1:]):
        change = np.abs(current - previous)
        flicker.extend([float(change.mean()), float(change.std()), float(np.quantile(change, .9))])
    vectors.append(np.asarray([np.mean(flicker), np.std(flicker), np.max(flicker)], dtype=np.float32))
    result = np.concatenate(vectors).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Non-finite video features')
    return result


class ImageNetFrameEncoder:
    """Frozen ImageNet appearance features, with no user-dataset fake/real head."""
    def __init__(self, path):
        self.path = Path(path)
        self.network = None

    def features(self, images):
        import torch
        from torchvision import transforms
        from torchvision.models import mobilenet_v3_small
        if self.network is None:
            model = mobilenet_v3_small(weights=None)
            model.load_state_dict(torch.load(self.path, map_location='cpu', weights_only=True))
            model.classifier = torch.nn.Identity()
            self.network = model.eval()
        transform = transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor(),
            transforms.Normalize([.485,.456,.406], [.229,.224,.225])])
        crops = _sequence_crops(images)
        batch = torch.stack([transform(Image.fromarray(crop)) for crop in crops])
        with torch.inference_mode():
            appearance = self.network(batch).numpy()
        statistics = temporal_features(images, [0.] * len(images), crops=crops)
        return np.concatenate([statistics, appearance.mean(axis=0), appearance.std(axis=0),
            np.abs(np.diff(appearance, axis=0)).mean(axis=0)]).astype(np.float32)


class TemporalModel:
    def __init__(self, path):
        self.path = Path(path) if path else None
        self.lock = threading.Lock()
        self.bundle = None
        self.encoder = None

    def ready(self):
        return bool(self.path and self.path.is_file())

    def _load(self):
        if self.bundle is not None:
            return
        import joblib
        bundle = joblib.load(self.path)  # Local administrator-produced training artifact only.
        if bundle.get('feature_version') != FEATURE_VERSION:
            raise ValueError('Unknown video feature format')
        base = (self.path.parent / bundle['base_backbone_relative_path']).resolve()
        if not base.is_file():
            raise ValueError('The video model requires its recorded visual backbone.')
        self.bundle, self.encoder = bundle, ImageNetFrameEncoder(base)

    def predict(self, images, frame_probabilities=None, frame_model_version=None):
        with self.lock:
            self._load()
            # Archive-trained frame probabilities are deliberately excluded: their
            # source provenance is unknown. Only the sampled images are shared.
            features = self.encoder.features(images).reshape(1, -1)
            raw = self.bundle['classifier'].predict_proba(features)[0, 1]
            logit = np.log(np.clip(raw, 1e-7, 1-1e-7) / np.clip(1-raw, 1e-7, 1))
            probability = 1 / (1 + np.exp(-np.clip(logit/self.bundle['temperature'], -30, 30)))
            return float(probability), dict(self.bundle['card'])
