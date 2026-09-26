"""
api.py — FastAPI Router for Image Detection Service

Sprint 2: Baseline Detection Service (Member 1: Divisha Manak Bohra - 23ESKCA038)
Task 1: Developing an image inference service with a defined input/output interface
Week 2: I/O API Specification and /api/detect/image endpoint with strict validation
"""

import base64
import os
import uuid
from typing import Tuple

from fastapi import APIRouter, FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from PIL import Image

from src.image_detector.preprocessing import (
    CorruptedImageError,
    EmptyImageError,
    UnsupportedFormatError,
)
from src.image_detector.schemas import (
    APIErrorResponse,
    ErrorDetail,
    ImageBase64Payload,
    ImageDetectionAPIResponse,
    ServiceInfo,
)
from src.image_detector.service import ImageDetectionService

router = APIRouter(prefix="/api/detect/image", tags=["Image Detection"])

# --------------------------------------------------------------------------
# Validation Constants
# --------------------------------------------------------------------------
MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB
ALLOWED_MIME_TYPES = {"image/png", "image/jpeg", "image/webp"}
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
MIN_DIMENSION = 16
MAX_DIMENSION = 8192

# Singleton service instance
_service: ImageDetectionService | None = None


def get_service() -> ImageDetectionService:
    """Lazy loader for shared inference service instance."""
    global _service
    if _service is None:
        _service = ImageDetectionService()
    return _service


# --------------------------------------------------------------------------
# Validation Helpers
# --------------------------------------------------------------------------
def verify_magic_bytes(header: bytes) -> str:
    """Inspects leading binary bytes to ensure format integrity."""
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if len(header) >= 12 and header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return "image/webp"
    return "unknown"


def validate_image_payload(
    file_bytes: bytes,
    filename: str,
    declared_content_type: str | None = None,
) -> Tuple[int, int]:
    """
    Executes strict multi-tier validation on uploaded image payload:
    1. Size boundary checks (0 bytes to 15 MB).
    2. Extension and declared MIME sanity.
    3. Deep magic bytes verification.
    4. Dimension sanity checks (16x16 to 8192x8192).
    """
    # Tier 1: Empty payload check
    if not file_bytes or len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=APIErrorResponse(
                error_code="EMPTY_PAYLOAD",
                message="Uploaded image payload is empty (0 bytes).",
                details=[ErrorDetail(field="file", issue="File contains 0 bytes.")],
            ).model_dump(),
        )

    # Tier 2: File size boundary check
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=APIErrorResponse(
                error_code="PAYLOAD_TOO_LARGE",
                message=f"File exceeds maximum allowed size of {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB.",
                details=[
                    ErrorDetail(
                        field="file",
                        issue=f"Received {len(file_bytes)} bytes; max is {MAX_FILE_SIZE_BYTES} bytes.",
                    )
                ],
            ).model_dump(),
        )

    # Tier 3: Filename extension check
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=APIErrorResponse(
                error_code="UNSUPPORTED_EXTENSION",
                message=f"File extension '{ext}' is unsupported.",
                details=[
                    ErrorDetail(
                        field="filename",
                        issue=f"Allowed extensions: {sorted(list(ALLOWED_EXTENSIONS))}",
                    )
                ],
            ).model_dump(),
        )

    # Tier 4: Magic bytes verification (guards against spoofed extension)
    detected_mime = verify_magic_bytes(file_bytes[:16])
    if detected_mime not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=APIErrorResponse(
                error_code="INVALID_MAGIC_BYTES",
                message=f"Image binary header is invalid or unsupported ({detected_mime}).",
                details=[
                    ErrorDetail(
                        field="file",
                        issue=f"Detected MIME '{detected_mime}' not in allowed set: {sorted(list(ALLOWED_MIME_TYPES))}",
                    )
                ],
            ).model_dump(),
        )

    # Tier 5: Dimension bounds verification
    import io

    try:
        with Image.open(io.BytesIO(file_bytes)) as img:
            w, h = img.size
            if w < MIN_DIMENSION or h < MIN_DIMENSION:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=APIErrorResponse(
                        error_code="DIMENSION_TOO_SMALL",
                        message=f"Image resolution ({w}x{h}) is below minimum allowed ({MIN_DIMENSION}x{MIN_DIMENSION}).",
                        details=[
                            ErrorDetail(
                                field="dimensions",
                                issue=f"Minimum resolution is {MIN_DIMENSION}x{MIN_DIMENSION} px.",
                            )
                        ],
                    ).model_dump(),
                )
            if w > MAX_DIMENSION or h > MAX_DIMENSION:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=APIErrorResponse(
                        error_code="DIMENSION_TOO_LARGE",
                        message=f"Image resolution ({w}x{h}) exceeds maximum allowed ({MAX_DIMENSION}x{MAX_DIMENSION}).",
                        details=[
                            ErrorDetail(
                                field="dimensions",
                                issue=f"Maximum resolution is {MAX_DIMENSION}x{MAX_DIMENSION} px.",
                            )
                        ],
                    ).model_dump(),
                )
            return w, h
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=APIErrorResponse(
                error_code="CORRUPTED_IMAGE",
                message=f"Failed to decode image structure: {e!s}",
                details=[ErrorDetail(field="file", issue="Image stream is corrupted or truncated.")],
            ).model_dump(),
        ) from e


# --------------------------------------------------------------------------
# API Endpoints
# --------------------------------------------------------------------------
@router.post(
    "",
    response_model=ImageDetectionAPIResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": APIErrorResponse, "description": "Empty image payload"},
        413: {"model": APIErrorResponse, "description": "Payload exceeds 15 MB limit"},
        415: {"model": APIErrorResponse, "description": "Unsupported format or MIME"},
        422: {"model": APIErrorResponse, "description": "Corrupted or invalid dimensions"},
    },
)
async def detect_image(
    file: UploadFile = File(..., description="Multipart image upload (PNG, JPEG, WEBP)"),
) -> JSONResponse:
    """
    Execute image detection inference on a multipart form-data upload.
    Strictly validates MIME type, magic bytes, dimensions, and file size.
    """
    safe_filename = os.path.basename(file.filename or "uploaded.png")

    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=APIErrorResponse(
                error_code="READ_FAILURE",
                message=f"Failed to read upload stream: {e!s}",
                details=[ErrorDetail(field="file", issue="Stream read error.")],
            ).model_dump(),
        ) from e

    # Perform strict validation
    validate_image_payload(content, safe_filename, file.content_type)

    service = get_service()
    try:
        base_resp = service.predict(content, filename=safe_filename)
        api_resp = ImageDetectionAPIResponse(
            filename=base_resp.filename,
            verdict=base_resp.verdict,
            is_ai=base_resp.is_ai,
            confidence=base_resp.confidence,
            probabilities=base_resp.probabilities,
            image_metadata=base_resp.image_metadata,
            latency_ms=base_resp.latency_ms,
            model_version=base_resp.model_version,
            device=base_resp.device,
            request_id=str(uuid.uuid4()),
        )
        return JSONResponse(status_code=status.HTTP_200_OK, content=api_resp.model_dump())
    except (EmptyImageError, UnsupportedFormatError, CorruptedImageError) as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=APIErrorResponse(
                error_code="INFERENCE_PREPROCESSING_ERROR",
                message=str(e),
                details=[ErrorDetail(field="file", issue=str(e))],
            ).model_dump(),
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=APIErrorResponse(
                error_code="INTERNAL_ERROR",
                message=f"Unexpected error during inference: {e!s}",
            ).model_dump(),
        ) from e


@router.post(
    "/base64",
    response_model=ImageDetectionAPIResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": APIErrorResponse, "description": "Invalid Base64 payload"},
        413: {"model": APIErrorResponse, "description": "Payload exceeds 15 MB limit"},
        415: {"model": APIErrorResponse, "description": "Unsupported media format"},
        422: {"model": APIErrorResponse, "description": "Corrupted or invalid dimensions"},
    },
)
async def detect_image_base64(payload: ImageBase64Payload) -> JSONResponse:
    """
    Execute image detection inference on a Base64-encoded string or RFC 2397 Data URI.
    """
    raw_str = payload.image_data.strip()
    if not raw_str:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=APIErrorResponse(
                error_code="EMPTY_PAYLOAD",
                message="Base64 payload string cannot be empty.",
                details=[ErrorDetail(field="image_data", issue="Payload is empty.")],
            ).model_dump(),
        )

    # Strip Data URI prefix if present
    if ";base64," in raw_str:
        raw_b64 = raw_str.split(";base64,")[-1]
    else:
        raw_b64 = raw_str

    try:
        content = base64.b64decode(raw_b64, validate=True)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=APIErrorResponse(
                error_code="INVALID_BASE64",
                message=f"Invalid Base64 encoding: {e!s}",
                details=[ErrorDetail(field="image_data", issue="Failed base64 decoding.")],
            ).model_dump(),
        ) from e

    safe_filename = os.path.basename(payload.filename or "upload.png")
    validate_image_payload(content, safe_filename)

    service = get_service()
    try:
        base_resp = service.predict(content, filename=safe_filename)
        api_resp = ImageDetectionAPIResponse(
            filename=base_resp.filename,
            verdict=base_resp.verdict,
            is_ai=base_resp.is_ai,
            confidence=base_resp.confidence,
            probabilities=base_resp.probabilities,
            image_metadata=base_resp.image_metadata,
            latency_ms=base_resp.latency_ms,
            model_version=base_resp.model_version,
            device=base_resp.device,
            request_id=str(uuid.uuid4()),
        )
        return JSONResponse(status_code=status.HTTP_200_OK, content=api_resp.model_dump())
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=APIErrorResponse(
                error_code="INFERENCE_ERROR",
                message=f"Failed to process base64 image: {e!s}",
            ).model_dump(),
        ) from e


@router.get("/health", response_model=ServiceInfo, status_code=status.HTTP_200_OK)
async def health_check() -> ServiceInfo:
    """Returns runtime telemetry, model version, and acceleration hardware status."""
    service = get_service()
    return service.get_service_info()


def create_app() -> FastAPI:
    """Application factory for Image Detection microservice."""
    from fastapi.middleware.cors import CORSMiddleware

    app_instance = FastAPI(
        title="Image Detection Service",
        description="Deep Learning microservice for AI-generated synthetic image detection (Form-2 Sprint 2)",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app_instance.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app_instance.include_router(router)

    @app_instance.get("/internal/health", status_code=status.HTTP_200_OK, tags=["Internal"])
    async def internal_health():
        service = get_service()
        info = service.get_service_info()
        return {
            "status": "healthy",
            "service": info.service_name,
            "model": info.model_name,
            "device": info.device,
            "version": info.model_version,
        }

    @app_instance.get("/", status_code=status.HTTP_200_OK, tags=["Root"])
    async def root():
        return {
            "message": "Image Detection Microservice is active.",
            "docs_url": "/docs",
            "endpoints": {
                "detect_multipart": "POST /api/detect/image",
                "detect_base64": "POST /api/detect/image/base64",
                "health": "GET /api/detect/image/health",
            },
        }

    return app_instance

