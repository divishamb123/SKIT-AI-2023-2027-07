import structlog
from fastapi import FastAPI
from app.core.config import settings
from app.api import health, auth

# Configure structured logging
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger()

from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from app.core.limiter import limiter

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.include_router(health.router, prefix="/internal")
app.include_router(auth.router, prefix=settings.API_V1_STR)


@app.on_event("startup")
async def startup_event():
    logger.info("auth_service_starting", version=settings.VERSION)


@app.get("/")
async def root():
    return {"message": f"{settings.PROJECT_NAME} is running."}
