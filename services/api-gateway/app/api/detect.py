from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status, Request
from sqlalchemy.ext.asyncio import AsyncSession
import uuid
import magic
from typing import Dict, Any

from .deps import get_current_user
from shared.db.session import get_async_session
from shared.db.models.job import Job, JobStatus, ModalityType
from ..core.storage import storage
from ..core.queue import queue_service
from ..core.config import settings
from ..core.limiter import limiter

router = APIRouter()

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}


@router.post("/image", status_code=status.HTTP_202_ACCEPTED)
@limiter.limit("20/minute")
async def detect_image(
    request: Request,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_async_session),
    current_user: Dict[str, Any] = Depends(get_current_user),
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
        new_job.error_message = f"Queue publish failed: {str(e)}"
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
