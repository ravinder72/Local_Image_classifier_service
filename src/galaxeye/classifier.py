"""ResNet18 classifier implementing the small Classifier interface.

The frozen pretrained backbone + trained 7-class linear head are loaded
from a local artifact once at construction. Inference never downloads
anything and never uses a hosted model/API.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18

from galaxeye.config import MODEL_VERSION, MODEL_WEIGHTS_PATH
from galaxeye.model import LandClass, Prediction

CLASSES = list(LandClass)  # stable index -> LandClass mapping for the head


class Classifier(ABC):
    """Small boundary between the application and any model."""

    model_version: str

    @abstractmethod
    def predict(self, input_array: np.ndarray) -> Prediction: ...


class ResNet18Classifier(Classifier):
    """ResNet18 with a frozen pretrained backbone and a trainable 7-class head.

    Converts the numpy input array to a torch.Tensor at this boundary, runs a
    forward pass, and returns label + confidence via softmax.
    """

    def __init__(
        self,
        weights_path=MODEL_WEIGHTS_PATH,
        model_version: str = MODEL_VERSION,
        num_classes: int = len(CLASSES),
    ) -> None:
        self.model_version = model_version
        self._model = self._build_model(num_classes)
        self._load_weights(weights_path)
        self._model.eval()

    @staticmethod
    def _build_model(num_classes: int) -> nn.Module:
        model = resnet18(weights=None)  # no download; weights loaded from local artifact
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    def _load_weights(self, weights_path) -> None:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        self._model.load_state_dict(state)

    @torch.no_grad()
    def predict(self, input_array: np.ndarray) -> Prediction:
        tensor = torch.from_numpy(input_array)  # (1, 3, 64, 64) float32 -> Tensor
        logits = self._model(tensor)[0]         # (7,)
        probs = torch.softmax(logits, dim=0)
        confidence, index = torch.max(probs, dim=0)
        return Prediction(
            label=CLASSES[index.item()],
            confidence=float(confidence.item()),
        )
