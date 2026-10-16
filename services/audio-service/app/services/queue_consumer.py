"""
QueueConsumer — background worker for audio jobs
=================================================
Consumes AASIST-L inference requests from the forensics.audio.detect queue.

Retry / timeout policy (Sprint 2 — Baseline Detection Services)
----------------------------------------------------------------
• Per-job retry:  each message carries an x-retry-count header.
  Up to MAX_RETRIES (3) transient failures are retried with
  exponential back-off (1 s → 2 s → 4 s) before the job is
  permanently marked FAILED.

• Inference timeout: predict_sync is given INFERENCE_TIMEOUT_S (30 s)
  inside asyncio.wait_for().  If it exceeds the budget the job is
  marked FAILED with a descriptive timeout message.

• MinIO download timeout: boto3 get_object is wrapped in
  asyncio.wait_for() with DOWNLOAD_TIMEOUT_S (20 s).

• Dead-letter: messages that exhaust all retries are nack'd so
  RabbitMQ routes them to forensics.dlx (already declared in start()).
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import aio_pika
import boto3
from aio_pika import IncomingMessage

from shared.db.models.detection_result import DetectionResult
from shared.db.models.explanation import Explanation
from shared.db.models.job import Job
from shared.db.session import async_session_maker

from app.services.anomaly_detector import detect_audio_anomalies
from app.services.audio_preprocessor import AudioPreprocessor
from app.services.spectrogram_visualizer import SpectrogramVisualizer

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Retry / timeout constants — read from env so they can be overridden
# in docker-compose.yml without rebuilding the image.
# ------------------------------------------------------------------
MAX_RETRIES: int = int(os.getenv("AUDIO_WORKER_MAX_RETRIES", "3"))
RETRY_BASE_DELAY_S: float = float(os.getenv("AUDIO_WORKER_RETRY_BASE_DELAY_S", "1.0"))
INFERENCE_TIMEOUT_S: float = float(
    os.getenv("AUDIO_WORKER_INFERENCE_TIMEOUT_S", "30.0")
)
DOWNLOAD_TIMEOUT_S: float = float(os.getenv("AUDIO_WORKER_DOWNLOAD_TIMEOUT_S", "20.0"))


class QueueConsumer:
    def __init__(
        self, connection: aio_pika.RobustConnection, queue_name: str, detector: Any
    ):
        self._connection = connection
        self._queue_name = queue_name
        self._detector = detector
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=os.getenv("MINIO_ENDPOINT", "http://minio:9000"),
            aws_access_key_id=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            aws_secret_access_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        )
        self.bucket = os.getenv("MINIO_BUCKET", "forensics-uploads")
        self.visualizer = SpectrogramVisualizer()

    async def start(self) -> None:
        channel = await self._connection.channel()
        await channel.set_qos(prefetch_count=1)
        queue = await channel.declare_queue(
            self._queue_name,
            durable=True,
            arguments={
                "x-dead-letter-exchange": "forensics.dlx",
                "x-dead-letter-routing-key": "dlx.failed",
            },
        )
        logger.info("Consuming from %s", self._queue_name)
        await queue.consume(self._process_message)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _retry_count(self, message: IncomingMessage) -> int:
        """Return how many times this message has been retried so far."""
        raw = (message.headers or {}).get("x-retry-count", 0)
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 0

    async def _mark_failed(self, job_id: str, reason: str) -> None:
        """Persist FAILED status to the database."""
        async with async_session_maker() as session:
            job = await session.get(Job, job_id)
            if job:
                job.status = "failed"
                job.error_message = reason
                job.completed_at = datetime.now(timezone.utc)
                await session.commit()
        logger.error("Audio job %s permanently failed: %s", job_id, reason)

    async def _download_audio(self, object_key: str) -> bytes:
        """Download audio bytes from MinIO with a configurable timeout."""
        loop = asyncio.get_running_loop()

        def _get():
            return self.s3_client.get_object(Bucket=self.bucket, Key=object_key)[
                "Body"
            ].read()

        return await asyncio.wait_for(
            loop.run_in_executor(None, _get),
            timeout=DOWNLOAD_TIMEOUT_S,
        )

    async def _run_inference(self, waveform: Any) -> dict:
        """Run AASIST-L inference with a configurable per-job timeout."""
        return await asyncio.wait_for(
            asyncio.to_thread(self._detector.predict_sync, waveform),
            timeout=INFERENCE_TIMEOUT_S,
        )

    # ------------------------------------------------------------------
    # Main message handler
    # ------------------------------------------------------------------

    async def _process_message(self, message: IncomingMessage) -> None:
        retry_count = self._retry_count(message)
        payload: dict = {}
        job_id: str = "unknown"

        # requeue=False — we handle requeue / nack ourselves below
        async with message.process(requeue=False):
            try:
                payload = json.loads(message.body.decode())
                job_id = payload["job_id"]
                logger.info(
                    "Processing audio job %s (attempt %d/%d)",
                    job_id,
                    retry_count + 1,
                    MAX_RETRIES + 1,
                )

                # ----------------------------------------------------------
                # Guard: degraded mode — model weights not loaded
                # ----------------------------------------------------------
                if not self._detector.is_ready():
                    degraded_reason = self._detector.health().get(
                        "error", "Model not loaded."
                    )
                    logger.error(
                        "Audio job %s rejected — detector not ready: %s",
                        job_id,
                        degraded_reason,
                    )
                    await self._mark_failed(
                        job_id, f"Model unavailable: {degraded_reason}"
                    )
                    return

                # ----------------------------------------------------------
                # Mark job as processing
                # ----------------------------------------------------------
                async with async_session_maker() as session:
                    job = await session.get(Job, job_id)
                    if job:
                        job.status = "processing"
                        job.started_at = datetime.now(timezone.utc)
                        await session.commit()

                # ----------------------------------------------------------
                # Download audio from MinIO (with timeout)
                # ----------------------------------------------------------
                try:
                    audio_bytes = await self._download_audio(payload["object_key"])
                except asyncio.TimeoutError:
                    raise RuntimeError(
                        f"MinIO download timed out after {DOWNLOAD_TIMEOUT_S}s "
                        f"for key {payload['object_key']!r}"
                    )

                # ----------------------------------------------------------
                # Preprocess waveform
                # ----------------------------------------------------------
                waveform = await asyncio.to_thread(
                    AudioPreprocessor.process_bytes, audio_bytes
                )

                # ----------------------------------------------------------
                # Run AASIST-L inference (with timeout)
                # ----------------------------------------------------------
                try:
                    result = await self._run_inference(waveform)
                except asyncio.TimeoutError:
                    raise RuntimeError(
                        f"AASIST-L inference timed out after {INFERENCE_TIMEOUT_S}s "
                        f"for job {job_id}"
                    )

                ai_prob = result["ai_probability"]

                # ----------------------------------------------------------
                # Spectrogram / anomaly explanation
                # ----------------------------------------------------------
                spectrogram_key = None
                flags = []
                if payload.get("options", {}).get("explain", True):
                    spec_bytes = await self.visualizer.generate(audio_bytes, ai_prob)
                    spectrogram_key = f"spectrograms/{job_id}/melspec.png"
                    self.s3_client.put_object(
                        Bucket=self.bucket,
                        Key=spectrogram_key,
                        Body=spec_bytes,
                        ContentType="image/png",
                    )
                    flags = await asyncio.to_thread(
                        detect_audio_anomalies, audio_bytes, ai_prob
                    )

                # ----------------------------------------------------------
                # Persist detection result + mark job completed
                # ----------------------------------------------------------
                async with async_session_maker() as session:
                    job = await session.get(Job, job_id)
                    if job:
                        det = DetectionResult(
                            job_id=job.id,
                            modality="audio",
                            ai_probability=result["ai_probability"],
                            confidence_score=result["confidence_score"],
                            verdict=result["verdict"],
                            model_name="AASIST-L",
                            processing_time_ms=result["processing_time_ms"],
                            device_used=result["device_used"],
                        )
                        session.add(det)
                        await session.flush()

                        if payload.get("options", {}).get("explain", True):
                            exp = Explanation(
                                result_id=det.id,
                                method="SPECTROGRAM",
                                artifact_type="image_png",
                                artifact_object_key=spectrogram_key,
                                artifact_data={"anomaly_flags": flags},
                            )
                            session.add(exp)

                        job.status = "completed"
                        job.completed_at = datetime.now(timezone.utc)
                        await session.commit()

                logger.info("Audio job %s completed successfully.", job_id)

            except Exception as exc:
                logger.warning(
                    "Audio job %s failed on attempt %d: %s",
                    job_id,
                    retry_count + 1,
                    exc,
                )

                if retry_count < MAX_RETRIES:
                    # --------------------------------------------------
                    # Exponential back-off before requeueing
                    # --------------------------------------------------
                    delay = RETRY_BASE_DELAY_S * (2**retry_count)
                    logger.info(
                        "Retrying audio job %s in %.1fs (attempt %d/%d)…",
                        job_id,
                        delay,
                        retry_count + 1,
                        MAX_RETRIES,
                    )
                    await asyncio.sleep(delay)

                    # Republish with incremented retry counter
                    channel = await self._connection.channel()
                    await channel.default_exchange.publish(
                        aio_pika.Message(
                            body=message.body,
                            headers={
                                **(message.headers or {}),
                                "x-retry-count": retry_count + 1,
                            },
                            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                        ),
                        routing_key=self._queue_name,
                    )
                    logger.info(
                        "Audio job %s requeued (retry %d).", job_id, retry_count + 1
                    )
                else:
                    # --------------------------------------------------
                    # All retries exhausted — permanently mark as failed
                    # --------------------------------------------------
                    await self._mark_failed(
                        job_id,
                        f"Permanently failed after {MAX_RETRIES} retries. "
                        f"Last error: {exc}",
                    )
