from datetime import datetime, timezone

from app.core.config import settings
from fastapi import APIRouter

router = APIRouter()


@router.get("/health", tags=["health"])
async def health_check():
    return {
        "service": "auth-service",
        "status": "healthy",
        "version": settings.VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dependencies": {},
    }
