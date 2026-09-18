"""
Image Detection Module — Sprint 2: Baseline Detection Service

Member 1: Divisha Manak Bohra (23ESKCA038)
Department of CSE (Artificial Intelligence), SKIT Jaipur
"""

from src.image_detector.schemas import (
    ClassProbabilities,
    ImageInferenceRequest,
    ImageInferenceResponse,
    ImageMetadata,
    PredictionVerdict,
    ServiceInfo,
)
from src.image_detector.preprocessing import (
    CorruptedImageError,
    EmptyImageError,
    ImageProcessingError,
    UnsupportedFormatError,
    decode_image,
    preprocess_image_tensor,
)
from src.image_detector.model import (
    BaselineImageClassifier,
    get_optimal_device,
    load_baseline_model,
)
from src.image_detector.service import ImageDetectionService

__all__ = [
    "ImageDetectionService",
    "BaselineImageClassifier",
    "ImageInferenceRequest",
    "ImageInferenceResponse",
    "ImageMetadata",
    "PredictionVerdict",
    "ClassProbabilities",
    "ServiceInfo",
    "decode_image",
    "preprocess_image_tensor",
    "get_optimal_device",
    "load_baseline_model",
    "ImageProcessingError",
    "EmptyImageError",
    "CorruptedImageError",
    "UnsupportedFormatError",
]
