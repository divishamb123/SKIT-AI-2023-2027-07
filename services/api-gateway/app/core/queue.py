import structlog
import aio_pika
import json
from typing import Dict, Any

from .config import settings

logger = structlog.get_logger()


class QueueService:
    def __init__(self):
        self.connection = None
        self.channel = None

    async def connect(self):
        """Establish connection to RabbitMQ."""
        try:
            self.connection = await aio_pika.connect_robust(settings.RABBITMQ_URL)
            self.channel = await self.connection.channel()
            logger.info("rabbitmq_connected")
        except Exception as e:
            logger.error("rabbitmq_connection_error", error=str(e))
            raise

    async def publish_message(self, routing_key: str, message_body: Dict[str, Any]):
        """Publish a JSON message to a specific routing key/queue."""
        if not self.channel:
            await self.connect()

        try:
            message = aio_pika.Message(
                body=json.dumps(message_body).encode("utf-8"),
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                content_type="application/json",
            )
            await self.channel.default_exchange.publish(
                message, routing_key=routing_key
            )
            logger.info("rabbitmq_message_published", routing_key=routing_key)
        except Exception as e:
            logger.error(
                "rabbitmq_publish_error", routing_key=routing_key, error=str(e)
            )
            raise

    async def close(self):
        """Close connection."""
        if self.connection:
            await self.connection.close()
            logger.info("rabbitmq_connection_closed")


queue_service = QueueService()
