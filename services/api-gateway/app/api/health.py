from fastapi import APIRouter
from datetime import datetime, timezone
from ..core.config import settings

router = APIRouter()


@router.get("/health", tags=["health"])
async def health_check():
    return {
        "service": "api-gateway",
        "status": "healthy",
        "version": settings.VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dependencies": {},
    }
