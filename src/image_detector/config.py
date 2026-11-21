"""
config.py — Centralized Configuration & Environment Settings for Image Detector

Production configuration settings, file paths, model versions, and operating constants.
"""

from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Model Checkpoint Locations
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
DEFAULT_RESNET_CHECKPOINT = CHECKPOINTS_DIR / "resnet18_cifake.pth"
DEFAULT_MOBILENET_CHECKPOINT = CHECKPOINTS_DIR / "mobilenetv3_cifake.pth"

# Model Versions
MODEL_VERSION_RESNET = "baseline-resnet18-v1.0"
MODEL_VERSION_ENSEMBLE = "ensemble-resnet18-mobilenetv3-v1.0"

# Target Image Processing Constants
DEFAULT_IMAGE_SIZE: Tuple[int, int] = (224, 224)
SUPPORTED_FORMATS: List[str] = ["PNG", "JPEG", "JPG", "WEBP", "BMP"]
IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]

# Validation & Service Bounds
MAX_FILE_SIZE_BYTES: int = 15 * 1024 * 1024  # 15 MB
MAX_BATCH_SIZE: int = 32  # Dynamic chunking ceiling
MIN_DIMENSION: int = 16
MAX_DIMENSION: int = 8192

# Mathematically Optimized Weights (Derived from Validation Grid Search)
OPTIMIZED_ENSEMBLE_WEIGHTS: Dict[str, float] = {
    "resnet18": 0.50,
    "mobilenet_v3": 0.50,
}

# Calibrated Decision Thresholds
TUNED_THRESHOLDS: Dict[str, float] = {
    "balanced_eer": 0.6700,
    "high_precision": 0.9900,
    "high_recall": 0.9600,
    "max_f1": 0.5400,
    "default": 0.5000,
}


class ModelNotReadyError(Exception):
    """Raised when inference is requested but required model checkpoints are missing."""

    pass
