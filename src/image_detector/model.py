"""
model.py — Baseline Neural Architecture & Model Loader for Image AI Detection

Sprint 2: Baseline Detection Service (Member 1: Divisha Manak Bohra - 23ESKCA038)
Task 1: Developing an image inference service with a defined input/output interface
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torchvision import models


MODEL_VERSION = "baseline-resnet18-v1.0"
NUM_CLASSES = 2  # Class 0: REAL, Class 1: AI_GENERATED


def get_optimal_device(preferred_device: Optional[str] = None) -> torch.device:
    """
    Selects the optimal hardware acceleration device available.
    Supports CUDA (NVIDIA), MPS (Apple Silicon), or high-performance CPU.
    """
    if preferred_device:
        try:
            return torch.device(preferred_device)
        except Exception:
            pass

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class BaselineImageClassifier(nn.Module):
    """
    ResNet-18 based forensic image classifier for detecting synthetic/AI imagery.
    Produces unnormalized logits for [REAL, AI_GENERATED].
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout_rate: float = 0.2):
        super().__init__()
        # Initialize standard ResNet18 backbone
        self.backbone = models.resnet18(weights=None)
        in_features = self.backbone.fc.in_features

        # Replace 1000-class head with 2-class forensic head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Kaiming uniform initialization for deterministic baseline behavior."""
        for m in self.backbone.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits (Batch, 2)."""
        return self.backbone(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract intermediate 512-dim embedding before the final classification head."""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def load_baseline_model(
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
) -> Tuple[BaselineImageClassifier, torch.device]:
    """
    Instantiates and loads the baseline image classifier onto the specified device.

    If checkpoint_path is None or missing, loads a clean initialized baseline ready for inference.
    """

    target_device = device or get_optimal_device()
    model = BaselineImageClassifier(num_classes=NUM_CLASSES)

    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=target_device)
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)

    model.to(target_device)
    model.eval()
    return model, target_device
