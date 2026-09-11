import asyncio
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "services", "api-gateway"))

from app.core.storage import storage
from app.core.queue import queue_service


async def main():
    print("Initializing Storage (MinIO)...")
    storage.initialize()

    test_object_name = "test_upload.txt"
    test_content = b"Hello from Phase A Verification!"

    print(f"Uploading {test_object_name}...")
    uploaded_name = await storage.upload_file(
        test_object_name, test_content, "text/plain"
    )
    print(f"Storage Upload Success: {uploaded_name}")

    print("Connecting to RabbitMQ...")
    await queue_service.connect()

    test_message = {"job_id": "test-1234", "object_key": uploaded_name, "type": "image"}

    print("Publishing message to queue...")
    await queue_service.publish_message("image-detection-queue", test_message)
    print("Message Published Successfully!")

    await queue_service.close()

    print("\nALL PHASE A RUNTIME VERIFICATIONS PASSED!")


if __name__ == "__main__":
    asyncio.run(main())
