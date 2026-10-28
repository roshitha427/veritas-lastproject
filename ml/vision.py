"""One visual pipeline shared by image and sampled video frames."""
from pathlib import Path
import threading
import numpy as np
from PIL import Image, ImageOps


class VisualModel:
    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.bundle = None
        self.network = None

    def ready(self):
        return self.path.is_file() and self.path.with_name('backbone.pt').is_file()

    def _load(self):
        if self.bundle is not None:
            return
        import joblib
        import torch
        from torchvision.models import mobilenet_v3_small
        self.bundle = joblib.load(self.path)  # Only administrator-generated local artifacts.
        self.network = mobilenet_v3_small(weights=None)
        if self.bundle.get('kind') == 'cnn':
            self.network.classifier[-1] = torch.nn.Linear(self.network.classifier[-1].in_features, 2)
        self.network.load_state_dict(torch.load(self.path.with_name('backbone.pt'), map_location='cpu', weights_only=True))
        if self.bundle.get('kind') != 'cnn':
            self.network.classifier = torch.nn.Identity()
        self.network.eval()
        torch.set_num_threads(min(4, torch.get_num_threads()))

    def predict(self, images):
        import torch
        from torchvision import transforms
        with self.lock:
            self._load()
            transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(),
                transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
            tensor = torch.stack([transform(ImageOps.exif_transpose(img).convert('RGB')) for img in images])
            with torch.inference_mode():
                if self.bundle.get('tta') == 'horizontal_flip_average':
                    count = len(tensor)
                    features = self.network(torch.cat((tensor, torch.flip(tensor, dims=[3])))).numpy()
                    original, flipped = features[:count], features[count:]
                    if self.bundle.get('kind') == 'cnn':
                        logits = ((original[:, 1] - original[:, 0]) + (flipped[:, 1] - flipped[:, 0])) / 2
                    else:
                        logits = (self.bundle['classifier'].decision_function(original) +
                                  self.bundle['classifier'].decision_function(flipped)) / 2
                else:
                    features = self.network(tensor).numpy()
                    logits = features[:, 1] - features[:, 0] if self.bundle.get('kind') == 'cnn' else self.bundle['classifier'].decision_function(features)
            calibration = self.bundle.get('calibration', {})
            if calibration.get('type') == 'platt':
                calibrated_logits = calibration['slope'] * logits + calibration['intercept']
            else:
                calibrated_logits = logits / self.bundle.get('temperature', 1.)
            fake = 1. / (1. + np.exp(-np.clip(calibrated_logits, -30, 30)))
            return fake.tolist(), self.bundle['card']
