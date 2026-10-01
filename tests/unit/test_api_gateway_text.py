"""
test_api_gateway_text.py — Unit tests for API Gateway /detect/text proxy route
"""

import os
from unittest.mock import AsyncMock, patch

from fastapi import status
from fastapi.testclient import TestClient
import httpx
import pytest

# Configure minimal test environment variables before importing settings
os.environ["JWT_SECRET_KEY"] = "test_secret_key"
os.environ["RABBITMQ_URL"] = "amqp://guest:guest@localhost:5672/"
os.environ["MINIO_SECRET_KEY"] = "test_minio_secret"
os.environ["TEXT_SERVICE_URL"] = "http://mock-text-service:8007"

from pathlib import Path
import sys

GATEWAY_DIR = Path(__file__).resolve().parent.parent.parent / "services" / "api-gateway"
sys.path.insert(0, str(GATEWAY_DIR))

# Clear any cached 'app' modules from other services
for mod in list(sys.modules.keys()):
    if mod == "app" or mod.startswith("app."):
        del sys.modules[mod]

from app.main import app
from app.schemas import TextDetectionResponse


@pytest.fixture
def client():
    return TestClient(app)


def test_detect_text_proxy_success(client: TestClient):
    """Test successful proxying of text detection request to text-service."""
    mock_response_data = {
        "label": "ai",
        "confidence": 0.9421,
        "model_version": "microsoft/deberta-v3-base-pretrained",
    }

    mock_resp = httpx.Response(
        status_code=status.HTTP_200_OK,
        json=mock_response_data,
        request=httpx.Request("POST", "http://mock-text-service:8007/detect"),
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp

        response = client.post(
            "/api/v1/detect/text",
            json={"text": "Artificial intelligence model generated this paragraph."},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["label"] == "ai"
        assert data["confidence"] == 0.9421
        assert data["model_version"] == "microsoft/deberta-v3-base-pretrained"

        # Assert contract parses via TextDetectionResponse
        parsed = TextDetectionResponse(**data)
        assert parsed.label == "ai"


def test_detect_text_empty_input_rejected(client: TestClient):
    """Test that empty or whitespace text is rejected before proxying."""
    response = client.post("/api/v1/detect/text", json={"text": "   "})
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "Text cannot be empty" in response.json().get("detail", "")


def test_detect_text_service_unavailable(client: TestClient):
    """Test 503 response when text-service cannot be reached."""
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = httpx.ConnectError("Connection refused")

        response = client.post(
            "/api/v1/detect/text",
            json={"text": "This is a legitimate test sentence for connection errors."},
        )

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert "Text detection service unavailable" in response.json().get("detail", "")
