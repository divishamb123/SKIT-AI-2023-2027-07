import asyncio
import json
import os
import aio_pika
import structlog
from datetime import datetime, timezone
import traceback

from shared.db.session import async_session_maker
from shared.db.models.job import Job

logger = structlog.get_logger()

RABBITMQ_URL = os.environ["RABBITMQ_URL"]
QUEUE_NAME = os.getenv("QUEUE_IMAGE_DETECTION", "image-detection-queue")


async def process_message(message: aio_pika.IncomingMessage):
    async with message.process(ignore_processed=True):
        try:
            payload = json.loads(message.body.decode())
            job_id = payload.get("job_id")
            if not job_id:
                logger.error("missing_job_id", payload=payload)
                await message.reject(requeue=False)
                return

            logger.info("processing_job_start", job_id=job_id)

            # Mark as PROCESSING
            async with async_session_maker() as session:
                job = await session.get(Job, job_id)
                if not job:
                    logger.error("job_not_found", job_id=job_id)
                    await message.reject(requeue=False)
                    return

                job.status = "processing"
                job.started_at = datetime.now(timezone.utc)
                await session.commit()

            # Simulate processing delay
            await asyncio.sleep(3)

            # Mark as COMPLETED
            async with async_session_maker() as session:
                job = await session.get(Job, job_id)
                if job:
                    job.status = "completed"
                    job.completed_at = datetime.now(timezone.utc)
                    await session.commit()
                    logger.info("processing_job_complete", job_id=job_id)

            # Acknowledge after successful DB update
            await message.ack()

        except Exception as e:
            logger.error(
                "job_processing_error", error=str(e), traceback=traceback.format_exc()
            )

            # If we know the job_id, mark it as FAILED
            if "job_id" in locals() and job_id:
                try:
                    async with async_session_maker() as session:
                        job = await session.get(Job, job_id)
                        if job:
                            job.status = "failed"
                            job.error_message = str(e)
                            job.completed_at = datetime.now(timezone.utc)
                            await session.commit()
                except Exception as db_err:
                    logger.error("failed_to_update_job_error", error=str(db_err))

            # Acknowledge the message so it doesn't loop infinitely (or we could reject without requeue)
            await message.reject(requeue=False)


async def main():
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ]
    )

    connection = await aio_pika.connect_robust(RABBITMQ_URL)
    async with connection:
        channel = await connection.channel()
        # Set prefetch count to 1 for fair dispatch
        await channel.set_qos(prefetch_count=1)

        queue = await channel.declare_queue(QUEUE_NAME, durable=True)

        logger.info("stub_consumer_started", queue=QUEUE_NAME)
        await queue.consume(process_message)

        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            logger.info("stub_consumer_stopped")


if __name__ == "__main__":
    asyncio.run(main())
