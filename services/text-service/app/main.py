import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import detect, health

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

app = FastAPI(
    title="Text Detection Service",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/internal")
app.include_router(detect.router, prefix="/api/v1/detect")

@app.on_event("startup")
async def startup_event():
    logger.info("text_service_starting", version="1.0.0")

@app.get("/")
async def root():
    return {"message": "Text Detection Service is running."}
