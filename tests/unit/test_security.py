from datetime import timedelta

# We need to set env vars before importing settings
import os

os.environ["JWT_SECRET_KEY"] = "test_secret_key"
os.environ["POSTGRES_PASSWORD"] = "test"
os.environ["RABBITMQ_PASSWORD"] = "test"
os.environ["MINIO_SECRET_KEY"] = "test"
os.environ["RABBITMQ_URL"] = "amqp://test"

from app.core import security


def test_password_hashing():
    password = "MySecurePassword123"
    hashed = security.get_password_hash(password)
    assert hashed != password
    assert len(hashed) > 10


def test_password_verification():
    password = "MySecurePassword123"
    hashed = security.get_password_hash(password)
    assert security.verify_password(password, hashed) is True
    assert security.verify_password("WrongPassword", hashed) is False


def test_access_token_creation():
    data = {"sub": "user_id_123", "email": "test@test.com"}
    token = security.create_access_token(data)
    assert isinstance(token, str)
    assert len(token) > 20


def test_refresh_token_creation():
    data = {"sub": "user_id_123"}
    token = security.create_refresh_token(data)
    assert isinstance(token, str)


def test_token_payload_decoding():
    data = {"sub": "user_id_123", "email": "test@test.com"}
    token = security.create_access_token(data)
    decoded = security.verify_token(token)
    assert decoded is not None
    assert decoded["sub"] == "user_id_123"
    assert decoded["email"] == "test@test.com"
    assert "exp" in decoded


def test_expired_token_rejection():
    data = {"sub": "user_id_123"}
    # Create token that expired 5 minutes ago
    token = security.create_access_token(data, expires_delta=timedelta(minutes=-5))
    decoded = security.verify_token(token)
    assert decoded is None


def test_invalid_token_rejection():
    token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.invalid.payload"
    decoded = security.verify_token(token)
    assert decoded is None
