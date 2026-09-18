"""
schemas.py — Strictly Defined Input/Output Interface for Image Detection Service

Sprint 2: Baseline Detection Service (Member 1: Divisha Manak Bohra - 23ESKCA038)
Task 1: Developing an image inference service with a defined input/output interface
"""

from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator


class PredictionVerdict(str, Enum):
    REAL = "REAL"
    AI_GENERATED = "AI_GENERATED"


class ClassProbabilities(BaseModel):
    """Softmax class probabilities for binary classification."""
    real: float = Field(..., ge=0.0, le=1.0, description="Probability that the image is authentic/real")
    ai_generated: float = Field(..., ge=0.0, le=1.0, description="Probability that the image is AI-generated")

    @field_validator("real", "ai_generated")
    @classmethod
    def round_probability(cls, v: float) -> float:
        return round(float(v), 4)


class ImageMetadata(BaseModel):
    """Extracted visual properties of the processed image."""
    width: int = Field(..., gt=0, description="Original image width in pixels")
    height: int = Field(..., gt=0, description="Original image height in pixels")
    channels: int = Field(..., ge=1, le=4, description="Number of color channels")
    format: str = Field(..., description="Detected image encoding format (PNG, JPEG, WEBP, etc.)")


class ImageInferenceRequest(BaseModel):
    """Request payload supporting raw bytes, base64 string, or filepath."""
    image_bytes: Optional[bytes] = Field(None, description="Raw binary image payload")
    image_base64: Optional[str] = Field(None, description="Base64 encoded image string or data URI")
    image_path: Optional[str] = Field(None, description="Local filesystem path to image")
    filename: str = Field(default="sample.png", description="Source filename for tracking")


class ImageInferenceResponse(BaseModel):
    """Standardized output response contract conforming to Form-2 specification."""
    filename: str = Field(..., description="Target image filename")
    verdict: PredictionVerdict = Field(..., description="Categorical classification verdict (REAL / AI_GENERATED)")
    is_ai: bool = Field(..., description="Convenience boolean flag indicating whether the image is synthetic")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score associated with the verdict")
    probabilities: ClassProbabilities = Field(..., description="Decomposed class probability distributions")
    image_metadata: ImageMetadata = Field(..., description="Extracted resolution and encoding metadata")
    latency_ms: float = Field(..., ge=0.0, description="Inference execution duration in milliseconds")
    model_version: str = Field(..., description="Identifier and version of the detection model")
    device: str = Field(..., description="Computing hardware used for inference (cpu, cuda, mps)")


class ServiceInfo(BaseModel):
    """Operational status and metadata of the inference service."""
    service_name: str = "Image Detection Inference Service"
    model_name: str
    model_version: str
    device: str
    supported_formats: List[str]
    status: str = "healthy"
