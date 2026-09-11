import structlog
from minio import Minio
from minio.error import S3Error
import io

from .config import settings

logger = structlog.get_logger()


class StorageService:
    def __init__(self):
        self.client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE,
        )
        self.bucket_name = settings.MINIO_BUCKET_UPLOADS

    def initialize(self):
        """Ensure the bucket exists."""
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                logger.info("minio_bucket_created", bucket=self.bucket_name)
            else:
                logger.info("minio_bucket_exists", bucket=self.bucket_name)
        except S3Error as e:
            logger.error("minio_initialization_error", error=str(e))
            raise

    async def upload_file(
        self,
        object_name: str,
        file_data: bytes,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Upload a file to MinIO."""
        try:
            file_stream = io.BytesIO(file_data)
            self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                data=file_stream,
                length=len(file_data),
                content_type=content_type,
            )
            logger.info("minio_upload_success", object_name=object_name)
            return object_name
        except S3Error as e:
            logger.error("minio_upload_error", object_name=object_name, error=str(e))
            raise


storage = StorageService()
