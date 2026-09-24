import asyncio
import json
import logging
from typing import Any
import aio_pika
from aio_pika import IncomingMessage
from shared.db.session import async_session_maker
from shared.db.models.job import Job
from shared.db.models.detection_result import DetectionResult
from shared.db.models.explanation import Explanation
from app.services.audio_preprocessor import AudioPreprocessor
from app.services.spectrogram_visualizer import SpectrogramVisualizer
from app.services.anomaly_detector import detect_audio_anomalies
import os
import boto3
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class QueueConsumer:
    def __init__(self, connection: aio_pika.RobustConnection, queue_name: str, detector: Any):
        self._connection = connection
        self._queue_name = queue_name
        self._detector = detector
        self.s3_client = boto3.client(
            's3',
            endpoint_url=os.getenv("MINIO_ENDPOINT", "http://minio:9000"),
            aws_access_key_id=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            aws_secret_access_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        )
        self.bucket = os.getenv("MINIO_BUCKET", "forensics-bucket")
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
            }
        )
        logger.info(f"Consuming from {self._queue_name}")
        await queue.consume(self._process_message)

    async def _process_message(self, message: IncomingMessage) -> None:
        async with message.process(requeue=False):
            try:
                payload = json.loads(message.body.decode())
                job_id = payload['job_id']
                logger.info(f"Processing audio job {job_id}")

                async with async_session_maker() as session:
                    job = await session.get(Job, job_id)
                    if job:
                        job.status = "processing"
                        job.started_at = datetime.now(timezone.utc)
                        await session.commit()
                
                response = self.s3_client.get_object(Bucket=self.bucket, Key=payload["input_object_key"])
                audio_bytes = response['Body'].read()

                waveform, meta = await asyncio.to_thread(AudioPreprocessor.process_bytes, audio_bytes)
                
                result = await asyncio.to_thread(self._detector.predict_sync, waveform)
                
                ai_prob = result["ai_probability"]
                
                # Explanation / Visualization
                spectrogram_key = None
                flags = []
                if payload.get("options", {}).get("explain", True):
                    spec_bytes = await self.visualizer.generate(audio_bytes, ai_prob)
                    spectrogram_key = f"spectrograms/{job_id}/melspec.png"
                    self.s3_client.put_object(
                        Bucket=self.bucket,
                        Key=spectrogram_key,
                        Body=spec_bytes,
                        ContentType="image/png"
                    )
                    flags = await asyncio.to_thread(detect_audio_anomalies, audio_bytes, ai_prob)

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
                            device_used=result["device_used"]
                        )
                        session.add(det)
                        await session.flush()
                        
                        if payload.get("options", {}).get("explain", True):
                            exp = Explanation(
                                result_id=det.id,
                                method="SPECTROGRAM",
                                artifact_type="image_png",
                                artifact_object_key=spectrogram_key,
                                artifact_data={"anomaly_flags": flags}
                            )
                            session.add(exp)
                            
                        job.status = "completed"
                        job.completed_at = datetime.now(timezone.utc)
                        await session.commit()
                        
            except Exception as exc:
                logger.error(f"Handler failed: {exc}")
                raise
