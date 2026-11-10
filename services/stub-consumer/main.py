"""
Image Detection Consumer
Dev Khandelwal (23ESKCA035) — Sprint 1 backend infrastructure

Consumes image detection jobs from the RabbitMQ queue:
  1. Downloads the uploaded image from MinIO
  2. POSTs it to the image-service for AI/real classification
  3. Writes verdict + confidence to the jobs table in PostgreSQL
  4. Marks the job COMPLETED or FAILED

Environment variables:
  RABBITMQ_URL           - amqp://user:pass@host:5672/
  QUEUE_IMAGE_DETECTION  - queue name (default: image-detection-queue)
  IMAGE_SERVICE_URL      - http://image-service:8004
  MINIO_ENDPOINT         - minio:9000
  MINIO_ACCESS_KEY       - access key
  MINIO_SECRET_KEY       - secret key
  MINIO_BUCKET           - bucket name (default: forensics-uploads)
  IMAGE_INFERENCE_TIMEOUT_S - per-job timeout in seconds (default: 30)
"""

import asyncio
import json
import os
import traceback
from datetime import datetime, timezone

import aio_pika
import httpx
import structlog
from minio import Minio

from shared.db.models.job import Job, JobStatus
from shared.db.session import async_session_maker

logger = structlog.get_logger()

RABBITMQ_URL = os.environ["RABBITMQ_URL"]
QUEUE_NAME = os.getenv("QUEUE_IMAGE_DETECTION", "image-detection-queue")
IMAGE_SERVICE_URL = os.getenv("IMAGE_SERVICE_URL", "http://image-service:8004")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ["MINIO_SECRET_KEY"]
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "forensics-uploads")
INFERENCE_TIMEOUT_S = float(os.getenv("IMAGE_INFERENCE_TIMEOUT_S", "30.0"))


def _download_from_minio(object_key: str) -> bytes:
    """Download object bytes from MinIO (runs in a thread)."""
    client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )
    response = client.get_object(MINIO_BUCKET, object_key)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


async def _run_inference(file_bytes: bytes, filename: str) -> dict:
    """POST image bytes to the image-service and return the JSON response."""
    url = f"{IMAGE_SERVICE_URL.rstrip('/')}/api/detect/image"
    async with httpx.AsyncClient(timeout=INFERENCE_TIMEOUT_S) as client:
        resp = await client.post(
            url,
            files={"file": (filename or "upload.png", file_bytes, "image/png")},
        )
        resp.raise_for_status()
        return resp.json()


async def process_message(message: aio_pika.IncomingMessage) -> None:
    async with message.process(ignore_processed=True):
        job_id: str | None = None
        try:
            payload = json.loads(message.body.decode())
            job_id = payload.get("job_id")
            object_key = payload.get("object_key")

            if not job_id or not object_key:
                logger.error("image_consumer_invalid_payload", payload=payload)
                await message.reject(requeue=False)
                return

            logger.info("image_job_start", job_id=job_id, object_key=object_key)

            # ── 1. Mark job as PROCESSING ──────────────────────────────────
            async with async_session_maker() as session:
                job = await session.get(Job, job_id)
                if not job:
                    logger.error("image_job_not_found", job_id=job_id)
                    await message.reject(requeue=False)
                    return
                filename = job.file_name or "upload.png"
                job.status = JobStatus.PROCESSING
                job.started_at = datetime.now(timezone.utc)
                await session.commit()

            # ── 2. Download image bytes from MinIO ─────────────────────────
            file_bytes = await asyncio.to_thread(_download_from_minio, object_key)
            logger.info(
                "image_minio_download_ok", job_id=job_id, size_bytes=len(file_bytes)
            )

            # ── 3. Run inference via image-service ─────────────────────────
            result = await _run_inference(file_bytes, filename)

            verdict = result.get("verdict", "UNKNOWN")
            confidence = result.get("confidence", 0.0)
            is_ai = result.get("is_ai", False)
            model_version = result.get("model_version", "unknown")
            latency_ms = result.get("latency_ms", 0.0)

            logger.info(
                "image_inference_ok",
                job_id=job_id,
                verdict=verdict,
                confidence=confidence,
                is_ai=is_ai,
                latency_ms=latency_ms,
                model_version=model_version,
            )

            # ── 4. Mark job COMPLETED and store verdict ────────────────────
            async with async_session_maker() as session:
                job = await session.get(Job, job_id)
                if job:
                    job.status = JobStatus.COMPLETED
                    job.completed_at = datetime.now(timezone.utc)
                    # Store verdict summary in error_message field (repurposed for result)
                    # TODO Sprint 3: write to reports table using Report model
                    job.error_message = json.dumps(
                        {
                            "verdict": verdict,
                            "confidence": confidence,
                            "is_ai": is_ai,
                            "model_version": model_version,
                            "latency_ms": latency_ms,
                        }
                    )
                    await session.commit()

            await message.ack()
            logger.info("image_job_complete", job_id=job_id, verdict=verdict)

        except httpx.HTTPStatusError as exc:
            err = f"image-service returned HTTP {exc.response.status_code}: {exc.response.text[:200]}"
            logger.error("image_inference_http_error", job_id=job_id, error=err)
            await _fail_job(job_id, err)
            await message.reject(requeue=False)

        except httpx.RequestError as exc:
            err = f"image-service unreachable: {exc}"
            logger.error("image_service_unavailable", job_id=job_id, error=err)
            await _fail_job(job_id, err)
            await message.reject(requeue=False)

        except Exception as exc:
            err = str(exc)
            logger.error(
                "image_job_failed",
                job_id=job_id,
                error=err,
                traceback=traceback.format_exc(),
            )
            await _fail_job(job_id, err)
            await message.reject(requeue=False)


async def _fail_job(job_id: str | None, error_message: str) -> None:
    """Mark a job as FAILED in the database."""
    if not job_id:
        return
    try:
        async with async_session_maker() as session:
            job = await session.get(Job, job_id)
            if job:
                job.status = JobStatus.FAILED
                job.error_message = error_message
                job.completed_at = datetime.now(timezone.utc)
                await session.commit()
    except Exception as db_err:
        logger.error("image_consumer_db_fail_update_error", error=str(db_err))


async def main() -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ]
    )

    logger.info(
        "image_consumer_starting",
        queue=QUEUE_NAME,
        image_service=IMAGE_SERVICE_URL,
        minio_endpoint=MINIO_ENDPOINT,
        inference_timeout_s=INFERENCE_TIMEOUT_S,
    )

    connection = await aio_pika.connect_robust(RABBITMQ_URL)
    async with connection:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=1)
        queue = await channel.declare_queue(QUEUE_NAME, durable=True)

        logger.info("image_consumer_ready", queue=QUEUE_NAME)
        await queue.consume(process_message)

        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            logger.info("image_consumer_stopped")


if __name__ == "__main__":
    asyncio.run(main())
