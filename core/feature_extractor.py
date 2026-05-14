"""
core/feature_extractor.py
Wraps the ResNet18-based image feature extraction pipeline.
Produces L2-normalized 512-dimensional feature vectors.
"""

import logging
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

log = logging.getLogger(__name__)

import torch
log = logging.getLogger(__name__)

import torchvision.models as models
from torchvision.models import ResNet18_Weights


class FeatureExtractor:
    """
    Extracts a 512-dimensional L2-normalized feature vector from an image
    using a pretrained ResNet18 backbone (global average pool output).
    """

    def __init__(self):
        self.weights = ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=self.weights)
        # Drop the final classification head — keep up to global avg pool
        self.model = torch.nn.Sequential(*list(model.children())[:-1])
        self.model.eval()
        self.transform = self.weights.transforms()

    def extract(self, image_path: Path) -> np.ndarray | None:
        """
        Extract a feature vector for the image at *image_path*.

        Returns
        -------
        np.ndarray of shape (512,), L2-normalised, or None on failure.
        """
        try:
            img_bytes = np.fromfile(str(image_path), dtype=np.uint8)
            img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            if img is None:
                return None

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            tensor = self.transform(img_pil).unsqueeze(0)

            with torch.no_grad():
                features = self.model(tensor)

            features = features.squeeze().numpy()
            norm = np.linalg.norm(features)
            if norm == 0:
                return None
            return features / norm

        except Exception as exc:
            log.warning("FeatureExtractor: error processing %s: %s", getattr(image_path, "name", image_path), exc)
            return None
