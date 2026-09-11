import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient
import uuid
import os

# We need to set env vars before importing anything that uses them
os.environ["JWT_SECRET_KEY"] = "test_secret_key"
os.environ["POSTGRES_PASSWORD"] = "test"
os.environ["RABBITMQ_PASSWORD"] = "test"
os.environ["MINIO_SECRET_KEY"] = "test"
os.environ["RABBITMQ_URL"] = "amqp://test"
os.environ["REDIS_URI"] = "redis://localhost:6379/1"

from app.main import app
from shared.db.session import get_async_session
from app.core import security
from app.core import redis
from shared.db.models.user import User

# Mocks
mock_session = AsyncMock()


async def override_get_async_session():
    yield mock_session


app.dependency_overrides[get_async_session] = override_get_async_session

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_mocks():
    mock_session.reset_mock()
    # Also patch redis locally
    redis.store_refresh_token = AsyncMock()
    redis.delete_refresh_token = AsyncMock()
    redis.verify_and_delete_refresh_token = AsyncMock(return_value="user_id_123")


def test_user_registration_success(monkeypatch):
    # Mocking db query
    mock_result = MagicMock()
    mock_result.scalars().first.return_value = None
    mock_session.execute.return_value = mock_result

    response = client.post(
        "/api/v1/auth/register",
        json={"name": "Test User", "email": "new@test.com", "password": "password123"},
    )

    assert response.status_code == 201
    assert response.json()["message"] == "User created successfully"
    assert "user_id" in response.json()
    assert mock_session.add.called
    assert mock_session.commit.called


def test_duplicate_email_rejection():
    # Mock existing user
    mock_user = User(email="existing@test.com")
    mock_result = MagicMock()
    mock_result.scalars().first.return_value = mock_user
    mock_session.execute.return_value = mock_result

    response = client.post(
        "/api/v1/auth/register",
        json={
            "name": "Test User",
            "email": "existing@test.com",
            "password": "password123",
        },
    )

    assert response.status_code == 400
    assert "Email already registered" in response.json()["detail"]


def test_login_success():
    hashed = security.get_password_hash("correct_password")
    mock_user = User(
        id=uuid.uuid4(),
        email="test@test.com",
        password_hash=hashed,
        role="user",
        is_active=True,
    )
    mock_result = MagicMock()
    mock_result.scalars().first.return_value = mock_user
    mock_session.execute.return_value = mock_result

    response = client.post(
        "/api/v1/auth/login",
        json={"email": "test@test.com", "password": "correct_password"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "message": "Successfully logged in",
        "token_type": "cookie"
    }
    
    cookies = [val for key, val in response.headers.items() if key.lower() == "set-cookie"]
    assert any("access_token=" in c and "HttpOnly" in c for c in cookies)
    assert any("refresh_token=" in c and "HttpOnly" in c for c in cookies)


def test_login_invalid_password():
    hashed = security.get_password_hash("correct_password")
    mock_user = User(
        id=uuid.uuid4(),
        email="test@test.com",
        password_hash=hashed,
        role="user",
        is_active=True,
    )
    mock_result = MagicMock()
    mock_result.scalars().first.return_value = mock_user
    mock_session.execute.return_value = mock_result

    response = client.post(
        "/api/v1/auth/login",
        json={"email": "test@test.com", "password": "wrong_password"},
    )

    assert response.status_code == 401
    assert "Incorrect email or password" in response.json()["detail"]


def test_logout_token_invalidation():
    # Calling logout
    response = client.post(
        "/api/v1/auth/logout", json={"refresh_token": "some_token_here"}
    )

    assert response.status_code == 200
    assert response.json()["message"] == "Successfully logged out"
    assert redis.delete_refresh_token.called
