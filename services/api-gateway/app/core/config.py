from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "API Gateway"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    AUTH_SERVICE_URL: str = "http://localhost:8000/api/v1/auth"

    # Storage Config
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str
    MINIO_SECURE: bool = False
    MINIO_BUCKET_UPLOADS: str = "forensics-uploads"

    # Queue Config
    RABBITMQ_URL: str
    QUEUE_IMAGE_DETECTION: str = "image-detection-queue"
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
