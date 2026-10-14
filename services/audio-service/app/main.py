"""
Audio Detection Service — FastAPI application entry point.

Startup sequence
----------------
1. AASISTDetector.load() is called (graceful — starts in degraded mode if
   weights are absent; see audio_detector.py for details).
2. A persistent aio_pika connection is opened to RabbitMQ.
3. QueueConsumer begins consuming the forensics.audio.detect queue.

Shutdown sequence
-----------------
On SIGTERM / SIGINT the asynccontextmanager yields and the RabbitMQ
connection is closed cleanly.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import aio_pika
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.services.audio_detector import AASISTDetector
from app.services.queue_consumer import QueueConsumer

logger = logging.getLogger(__name__)

detector = AASISTDetector()
consumer: QueueConsumer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------
    logger.info("Audio service starting up…")

    # Load AASIST-L model (graceful — won't crash if weights are missing)
    detector.load()

    rabbitmq_url = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost/")
    connection = await aio_pika.connect_robust(rabbitmq_url)

    global consumer
    consumer = QueueConsumer(connection, "forensics.audio.detect", detector)
    await consumer.start()

    logger.info("Audio service ready.  model_loaded=%s", detector.is_ready())
    yield

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    logger.info("Audio service shutting down…")
    await connection.close()


app = FastAPI(
    title="Audio Detection Service",
    version="1.1.0",
    description=(
        "Background worker + health endpoint for AASIST-L audio deepfake detection. "
        "Starts in degraded mode when pretrained weights are absent."
    ),
    lifespan=lifespan,
)


@app.get("/internal/health", tags=["internal"])
def health() -> Any:
    """
    Liveness + readiness probe.

    Returns HTTP 200 with {"status": "healthy"} when the model is loaded,
    or HTTP 503 with {"status": "degraded"} when weights are missing.
    """
    info = detector.health()
    status_code = 200 if info["status"] == "healthy" else 503
    return JSONResponse(content=info, status_code=status_code)
