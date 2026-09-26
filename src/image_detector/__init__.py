"""
Image Detection Module — Sprint 2: Baseline Detection Service

Member 1: Divisha Manak Bohra (23ESKCA038)
Department of CSE (Artificial Intelligence), SKIT Jaipur
"""

from src.image_detector.schemas import (
    APIErrorResponse,
    ClassProbabilities,
    ErrorDetail,
    ImageBase64Payload,
    ImageDetectionAPIResponse,
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
from src.image_detector.api import router as image_detection_router

__all__ = [
    "ImageDetectionService",
    "BaselineImageClassifier",
    "ImageInferenceRequest",
    "ImageInferenceResponse",
    "ImageDetectionAPIResponse",
    "ImageBase64Payload",
    "APIErrorResponse",
    "ErrorDetail",
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
    "image_detection_router",
]
