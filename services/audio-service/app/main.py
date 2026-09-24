from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.services.audio_detector import AASISTDetector
from app.services.queue_consumer import QueueConsumer
import aio_pika

detector = AASISTDetector()
consumer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    detector.load()
    
    # We would normally connect to RabbitMQ here, but for this mock setup, we connect in the consumer
    import os
    rabbitmq_url = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost/")
    connection = await aio_pika.connect_robust(rabbitmq_url)
    
    global consumer
    consumer = QueueConsumer(connection, "forensics.audio.detect", detector)
    await consumer.start()
    
    yield
    # Shutdown
    await connection.close()

app = FastAPI(title="Audio Detection Service", version="1.0.0", lifespan=lifespan)

@app.get("/internal/health")
def health():
    return {"status": "healthy"}
