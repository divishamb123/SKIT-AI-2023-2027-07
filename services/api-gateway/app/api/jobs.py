from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import uuid
from typing import Dict, Any

from .deps import get_current_user
from shared.db.session import get_async_session
from shared.db.models.job import Job

router = APIRouter()


@router.get("/{job_id}")
async def get_job(
    job_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    query = select(Job).where(
        Job.id == job_id, Job.user_id == uuid.UUID(current_user["sub"])
    )
    result = await db.execute(query)
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Job not found"
        )

    return {
        "id": job.id,
        "status": job.status,
        "modality": job.modality,
        "file_name": job.file_name,
        "created_at": job.created_at,
        "error_message": job.error_message,
    }
