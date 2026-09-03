import structlog
from fastapi import FastAPI
from .core.config import settings
from .api import health, auth
from .api import detect, jobs

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
from .core.limiter import limiter

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/internal")
app.include_router(auth.router, prefix=f"{settings.API_V1_STR}/auth")
app.include_router(detect.router, prefix=f"{settings.API_V1_STR}/detect")
app.include_router(jobs.router, prefix=f"{settings.API_V1_STR}/jobs")

from .core.storage import storage
from .core.queue import queue_service


@app.on_event("startup")
async def startup_event():
    logger.info("api_gateway_starting", version=settings.VERSION)
    storage.initialize()
    await queue_service.connect()


@app.on_event("shutdown")
async def shutdown_event():
    await queue_service.close()


@app.get("/")
async def root():
    return {"message": f"{settings.PROJECT_NAME} is running."}
