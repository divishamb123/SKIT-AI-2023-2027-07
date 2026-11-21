"""
model.py — Baseline Neural Architecture & Model Loader for Image AI Detection

Forensic image classifier using deep residual neural networks with hardware acceleration.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torchvision import models

from src.image_detector.config import (
    DEFAULT_RESNET_CHECKPOINT,
    MODEL_VERSION_RESNET,
)

MODEL_VERSION = MODEL_VERSION_RESNET
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


def _remap_state_dict(state_dict: dict, model: nn.Module) -> dict:
    """Remaps state_dict keys with or without 'backbone.' prefix for compatibility."""
    target_keys = set(model.state_dict().keys())
    new_sd = {}
    for k, v in state_dict.items():
        if k in target_keys:
            new_sd[k] = v
        elif f"backbone.{k}" in target_keys:
            new_sd[f"backbone.{k}"] = v
        elif k.startswith("backbone.") and k[9:] in target_keys:
            new_sd[k[9:]] = v
        else:
            new_sd[k] = v
    return new_sd


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

    # Auto-discover trained checkpoint if not explicitly provided
    if checkpoint_path is None:
        if DEFAULT_RESNET_CHECKPOINT.is_file():
            checkpoint_path = DEFAULT_RESNET_CHECKPOINT

    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=target_device)
        sd = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(_remap_state_dict(sd, model))

    model.to(target_device)
    model.eval()
    return model, target_device
