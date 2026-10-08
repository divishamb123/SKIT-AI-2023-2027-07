import uuid
from typing import Any

import httpx
import magic
import structlog
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from shared.db.models.job import Job, JobStatus, ModalityType
from shared.db.session import get_async_session

from ..core.config import settings
from ..core.limiter import limiter
from ..core.queue import queue_service
from ..core.storage import storage
from ..schemas import TextDetectionRequest, TextDetectionResponse
from .deps import get_current_user

router = APIRouter()
logger = structlog.get_logger()

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}


@router.post("/image", status_code=status.HTTP_202_ACCEPTED)
@limiter.limit("20/minute")
async def detect_image(
    request: Request,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_async_session),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    # 1. Validate file size
    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Max size is {MAX_FILE_SIZE / 1024 / 1024} MB.",
        )

    # 2. Validate MIME type using python-magic
    mime_type = magic.from_buffer(file_bytes, mime=True)
    if mime_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{mime_type}'. Allowed: {', '.join(ALLOWED_MIME_TYPES)}",
        )

    # 3. Generate UUID-based MinIO object key
    object_key = f"{uuid.uuid4()}-{file.filename}"

    # 4. Upload to MinIO (if fails, DB record is NOT created)
    try:
        await storage.upload_file(object_key, file_bytes, mime_type)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload file to storage.",
        )

    # 5. Create Database Job Record
    new_job = Job(
        user_id=uuid.UUID(current_user["sub"]),
        status=JobStatus.QUEUED,
        modality=ModalityType.IMAGE,
        input_type="file",
        input_object_key=object_key,
        file_name=file.filename,
        file_size_bytes=len(file_bytes),
        file_mime_type=mime_type,
    )
    db.add(new_job)
    await db.commit()
    await db.refresh(new_job)

    # 6. Publish to RabbitMQ
    message_payload = {
        "job_id": str(new_job.id),
        "object_key": object_key,
        "modality": "image",
    }

    try:
        await queue_service.publish_message(
            settings.QUEUE_IMAGE_DETECTION, message_payload
        )
    except Exception as e:
        # If RabbitMQ fails, mark job as FAILED and store error
        new_job.status = JobStatus.FAILED
        new_job.error_message = f"Queue publish failed: {e!s}"
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to queue job for processing.",
        )

    return {
        "job_id": new_job.id,
        "status": new_job.status,
        "message": "Image successfully uploaded and queued for detection.",
    }


ALLOWED_AUDIO_MIME_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/flac",
    "audio/ogg",
    "audio/mpeg",
    "audio/mp3",
}


@router.post("/audio", status_code=status.HTTP_202_ACCEPTED)
@limiter.limit("20/minute")
async def detect_audio(
    request: Request,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_async_session),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    # 1. Validate file size
    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Max size is {MAX_FILE_SIZE / 1024 / 1024} MB.",
        )

    # 2. Validate MIME type using python-magic
    mime_type = magic.from_buffer(file_bytes, mime=True)
    if mime_type not in ALLOWED_AUDIO_MIME_TYPES and not mime_type.startswith("audio/"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{mime_type}'. Allowed audio formats only.",
        )

    # 3. Generate UUID-based MinIO object key
    object_key = f"{uuid.uuid4()}-{file.filename}"

    # 4. Upload to MinIO
    try:
        await storage.upload_file(object_key, file_bytes, mime_type)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload file to storage.",
        )

    # 5. Create Database Job Record
    new_job = Job(
        user_id=uuid.UUID(current_user["sub"]),
        status=JobStatus.QUEUED,
        modality=ModalityType.AUDIO,
        input_type="file",
        input_object_key=object_key,
        file_name=file.filename,
        file_size_bytes=len(file_bytes),
        file_mime_type=mime_type,
    )
    db.add(new_job)
    await db.commit()
    await db.refresh(new_job)

    # 6. Publish to RabbitMQ
    message_payload = {
        "job_id": str(new_job.id),
        "object_key": object_key,
        "modality": "audio",
    }

    try:
        await queue_service.publish_message(
            settings.QUEUE_AUDIO_DETECTION, message_payload
        )
    except Exception as e:
        new_job.status = JobStatus.FAILED
        new_job.error_message = f"Queue publish failed: {e!s}"
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to queue job for processing.",
        )

    return {
        "job_id": new_job.id,
        "status": new_job.status,
        "message": "Audio successfully uploaded and queued for detection.",
    }


@router.post("/text", response_model=TextDetectionResponse)
@limiter.limit("30/minute")
async def detect_text(
    request: Request,
    payload: TextDetectionRequest,
):
    if not payload.text or not payload.text.strip():
        logger.warning("empty_text_detection_request")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text cannot be empty.",
        )

    target_url = f"{settings.TEXT_SERVICE_URL.rstrip('/')}/detect"
    logger.info(
        "proxying_text_detection",
        target_url=target_url,
        text_len=len(payload.text),
    )

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(target_url, json={"text": payload.text})
            if resp.status_code >= 400:
                try:
                    error_detail = resp.json().get("detail", resp.text)
                except Exception:
                    error_detail = resp.text
                raise HTTPException(status_code=resp.status_code, detail=error_detail)

            result = resp.json()
            return TextDetectionResponse(**result)
        except HTTPException:
            raise
        except httpx.RequestError as exc:
            logger.error("text_service_unavailable", error=str(exc), url=target_url)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Text detection service unavailable: {exc}",
            )
        except Exception as exc:
            logger.error("text_detection_proxy_error", error=str(exc))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to process text detection: {exc}",
            )
