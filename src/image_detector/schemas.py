"""
schemas.py — Strictly Defined Input/Output Interface for Image Detection Service

Production Pydantic schema contracts for single and batched image forensic inference.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid

from pydantic import BaseModel, Field, field_validator


class PredictionVerdict(str, Enum):
    REAL = "REAL"
    AI_GENERATED = "AI_GENERATED"


class ClassProbabilities(BaseModel):
    """Softmax class probabilities for binary classification."""

    real: float = Field(
        ..., ge=0.0, le=1.0, description="Probability that the image is authentic/real"
    )
    ai_generated: float = Field(
        ..., ge=0.0, le=1.0, description="Probability that the image is AI-generated"
    )

    @field_validator("real", "ai_generated")
    @classmethod
    def round_probability(cls, v: float) -> float:
        return round(float(v), 4)


class ImageMetadata(BaseModel):
    """Extracted visual properties of the processed image."""

    width: int = Field(..., gt=0, description="Original image width in pixels")
    height: int = Field(..., gt=0, description="Original image height in pixels")
    channels: int = Field(..., ge=1, le=4, description="Number of color channels")
    format: str = Field(
        ..., description="Detected image encoding format (PNG, JPEG, WEBP, etc.)"
    )


class ImageInferenceRequest(BaseModel):
    """Request payload supporting raw bytes, base64 string, or filepath."""

    image_bytes: Optional[bytes] = Field(None, description="Raw binary image payload")
    image_base64: Optional[str] = Field(
        None, description="Base64 encoded image string or data URI"
    )
    image_path: Optional[str] = Field(
        None, description="Local filesystem path to image"
    )
    filename: str = Field(
        default="sample.png", description="Source filename for tracking"
    )


class ImageBase64Payload(BaseModel):
    """JSON payload for Base64 image detection request."""

    image_data: str = Field(
        ...,
        description="Base64-encoded image string or RFC 2397 Data URI (e.g. data:image/png;base64,...)",
    )
    filename: str = Field(
        default="upload.png",
        description="Optional client-provided filename identifier",
    )


class ImageInferenceResponse(BaseModel):
    """Standardized output response contract conforming to Form-2 specification."""

    filename: str = Field(..., description="Target image filename")
    verdict: PredictionVerdict = Field(
        ..., description="Categorical classification verdict (REAL / AI_GENERATED)"
    )
    is_ai: bool = Field(
        ...,
        description="Convenience boolean flag indicating whether the image is synthetic",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score associated with the verdict"
    )
    probabilities: ClassProbabilities = Field(
        ..., description="Decomposed class probability distributions"
    )
    image_metadata: ImageMetadata = Field(
        ..., description="Extracted resolution and encoding metadata"
    )
    latency_ms: float = Field(
        ..., ge=0.0, description="Inference execution duration in milliseconds"
    )
    model_version: str = Field(
        ..., description="Identifier and version of the detection model"
    )
    device: str = Field(
        ..., description="Computing hardware used for inference (cpu, cuda, mps)"
    )


class ImageDetectionAPIResponse(ImageInferenceResponse):
    """Enhanced public API response payload with tracking metadata."""

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique UUID tracking ID for this inference request",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 UTC timestamp of prediction execution",
    )
    status: str = Field(default="success", description="API response execution status")


class ErrorDetail(BaseModel):
    """Specific error descriptor."""

    field: Optional[str] = Field(
        None, description="Associated request parameter or field"
    )
    issue: str = Field(..., description="Explanation of validation failure or error")


class APIErrorResponse(BaseModel):
    """RFC 7807 compliant standardized API error response."""

    error_code: str = Field(
        ..., description="Machine-readable error classification code"
    )
    message: str = Field(..., description="Human-readable error explanation")
    details: Optional[List[ErrorDetail]] = Field(
        None, description="Granular error breakdowns"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 UTC timestamp of error event",
    )
    extra: Optional[Dict[str, Any]] = Field(
        None, description="Contextual debugging info"
    )


class ServiceInfo(BaseModel):
    """Operational status and metadata of the inference service."""

    service_name: str = "Image Detection Inference Service"
    model_name: str
    model_version: str
    device: str
    supported_formats: List[str]
    status: str = "healthy"


# --------------------------------------------------------------------------
# Batch Processing Schemas
# --------------------------------------------------------------------------


class BatchItemError(BaseModel):
    """Failure descriptor for an individual image item within a batch."""

    index: int = Field(
        ..., description="0-indexed position of the item in the submitted batch"
    )
    filename: str = Field(..., description="Filename identifier of the failed image")
    error_code: str = Field(
        ..., description="Machine-readable error classification code"
    )
    message: str = Field(..., description="Explanation of processing failure")


class BatchSummary(BaseModel):
    """Aggregated analytical summary of batch inference results."""

    total_submitted: int = Field(
        ..., ge=0, description="Total number of images submitted in the batch"
    )
    successful_count: int = Field(
        ..., ge=0, description="Number of images successfully processed"
    )
    failed_count: int = Field(
        ..., ge=0, description="Number of images that failed processing"
    )
    real_count: int = Field(
        ..., ge=0, description="Number of images classified as authentic REAL"
    )
    ai_generated_count: int = Field(
        ..., ge=0, description="Number of images classified as synthetic AI_GENERATED"
    )
    mean_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Mean confidence across successful predictions"
    )

    @field_validator("mean_confidence")
    @classmethod
    def round_mean_confidence(cls, v: float) -> float:
        return round(float(v), 4)


class BatchImageInferenceResponse(BaseModel):
    """Standardized batch inference output contract."""

    results: List[ImageInferenceResponse] = Field(
        default_factory=list,
        description="Individual inference predictions for successfully processed images",
    )
    errors: List[BatchItemError] = Field(
        default_factory=list,
        description="Details of any failed images in the batch",
    )
    summary: BatchSummary = Field(
        ..., description="Aggregated summary statistics for the processed batch"
    )
    batch_latency_ms: float = Field(
        ..., ge=0.0, description="Total batch execution time in milliseconds"
    )
    avg_latency_per_image_ms: float = Field(
        ...,
        ge=0.0,
        description="Average execution latency per successfully processed image",
    )
    chunk_size_used: int = Field(
        ..., gt=0, description="Tensor collation chunk size applied during inference"
    )
    model_version: str = Field(
        ..., description="Identifier and version of the detection model"
    )
    device: str = Field(
        ..., description="Computing hardware used for inference (cpu, cuda, mps)"
    )


class BatchImageDetectionAPIResponse(BatchImageInferenceResponse):
    """Public API batch response contract with tracking metadata."""

    batch_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique UUID tracking ID for this batch request",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 UTC timestamp of batch execution",
    )
    status: str = Field(
        default="success", description="API batch response execution status"
    )


class BatchBase64Item(BaseModel):
    """Single item entry for Base64 batch submission."""

    image_data: str = Field(
        ..., description="Base64-encoded image string or RFC 2397 Data URI"
    )
    filename: Optional[str] = Field(
        default=None, description="Optional filename identifier"
    )


class BatchBase64Payload(BaseModel):
    """Request payload for multi-image Base64 batch inference."""

    images: List[BatchBase64Item] = Field(
        ...,
        description="List of base64-encoded images to process in batch (1 to 32 items)",
    )
