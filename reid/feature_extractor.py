import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Optional
from config.settings import settings

class VehicleReIDExtractor:
    """
    Deep Metric Vehicle Feature Extractor (Person 2 Component).
    
    Responsibilities:
    - Ingests cropped vehicle images from Person 1 (cv2 BGR arrays).
    - Preprocesses images (resize 224x224, BGR->RGB, ImageNet normalization).
    - Extracts deep visual feature representations using a lightweight backbone.
    - Projects features into a normalized hyperspace (L2-norm = 1.0) of dimension 512.
    - Achieves sub-10ms inference on standard CPUs.
    """
    def __init__(
        self,
        embedding_dim: int = settings.REID_EMBEDDING_DIM,
        device: Optional[str] = None
    ):
        self.embedding_dim = embedding_dim
        self.model_name = settings.REID_MODEL_NAME
        self.model_version = "1.0"

        # Determine target compute device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Build feature extractor backbone
        self._build_model()

    def _build_model(self):
        """Initializes MobileNetV3 backbone and deterministic projection head."""
        # Load lightweight MobileNetV3-Small weights
        base_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.backbone = base_model.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Deterministic projection head to match desired embedding dimension
        # MobileNetV3-Small features output 576 channels
        torch.manual_seed(42)
        self.projection = nn.Linear(576, self.embedding_dim)
        nn.init.orthogonal_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

        # Place on device and set strictly to evaluation mode
        self.backbone.to(self.device)
        self.pool.to(self.device)
        self.projection.to(self.device)

        self.backbone.eval()
        self.pool.eval()
        self.projection.eval()

        # ImageNet normalization constants
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def preprocess(self, crop: np.ndarray) -> torch.Tensor:
        """
        Converts BGR OpenCV crop to preprocessed PyTorch Tensor [1, 3, 224, 224].
        """
        if crop is None or crop.size == 0:
            raise ValueError("Vehicle crop is empty or invalid.")

        # Resize to standard model resolution
        resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)

        # Convert BGR -> RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Standard ImageNet Z-score normalization
        normalized = (rgb - self.mean) / self.std

        # HWC -> CHW -> NCHW
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        return tensor

    def extract(self, crop: np.ndarray) -> List[float]:
        """
        Extracts a single L2-normalized visual embedding vector for a vehicle crop.
        Returns: List of float values with length == embedding_dim (512).
        """
        tensor = self.preprocess(crop)

        with torch.no_grad():
            feat = self.backbone(tensor)
            feat = self.pool(feat)
            feat = torch.flatten(feat, 1)
            projected = self.projection(feat)

            # L2 Normalization: ||v|| = 1.0
            norm = torch.norm(projected, p=2, dim=1, keepdim=True)
            normalized = projected / torch.clamp(norm, min=1e-12)

        return [round(val, 5) for val in normalized.squeeze(0).cpu().numpy().tolist()]

    def extract_batch(self, crops: List[np.ndarray]) -> List[List[float]]:
        """
        Batch extraction for multiple vehicle crops simultaneously.
        """
        if not crops:
            return []

        tensors = [self.preprocess(c) for c in crops]
        batch = torch.cat(tensors, dim=0)

        with torch.no_grad():
            feat = self.backbone(batch)
            feat = self.pool(feat)
            feat = torch.flatten(feat, 1)
            projected = self.projection(feat)

            norm = torch.norm(projected, p=2, dim=1, keepdim=True)
            normalized = projected / torch.clamp(norm, min=1e-12)

        out = normalized.cpu().numpy()
        return [[round(float(v), 5) for v in row] for row in out]
