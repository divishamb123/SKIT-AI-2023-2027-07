"""
test_image_detection_api.py — Unit & Integration Tests for /api/detect/image Endpoint

Sprint 2: Baseline Detection Service (Member 1: Divisha Manak Bohra - 23ESKCA038)
Week 2: I/O API Specification and /api/detect/image endpoint with strict validation
"""

import base64
import io

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from src.image_detector.api import create_app

app = create_app()
client = TestClient(app)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
@pytest.fixture
def valid_png_bytes():
    buf = io.BytesIO()
    img = Image.new("RGB", (64, 64), color=(60, 120, 200))
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def valid_jpeg_bytes():
    buf = io.BytesIO()
    img = Image.new("RGB", (64, 64), color=(200, 100, 50))
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def valid_webp_bytes():
    buf = io.BytesIO()
    img = Image.new("RGB", (64, 64), color=(50, 180, 90))
    img.save(buf, format="WEBP")
    return buf.getvalue()


# --------------------------------------------------------------------------
# Health & Status Tests
# --------------------------------------------------------------------------
def test_api_health_endpoint():
    """Verify public health endpoint reports operational metadata."""
    response = client.get("/api/detect/image/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "ResNet" in data["model_name"]
    assert "PNG" in data["supported_formats"]


def test_internal_health_endpoint():
    """Verify internal microservice Docker healthcheck endpoint."""
    response = client.get("/internal/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "device" in data


def test_root_endpoint():
    """Verify service root returns welcome banner and endpoints."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "endpoints" in data


# --------------------------------------------------------------------------
# Happy Path Tests (Multipart Uploads)
# --------------------------------------------------------------------------
def test_detect_image_valid_png(valid_png_bytes):
    """Verify successful inference on valid PNG multipart upload."""
    response = client.post(
        "/api/detect/image",
        files={"file": ("sample.png", valid_png_bytes, "image/png")},
    )
    assert response.status_code == 200
    data = response.json()

    # Validate schema compliance
    assert data["filename"] == "sample.png"
    assert data["verdict"] in ["REAL", "AI_GENERATED"]
    assert isinstance(data["is_ai"], bool)
    assert 0.0 <= data["confidence"] <= 1.0
    assert 0.0 <= data["probabilities"]["real"] <= 1.0
    assert 0.0 <= data["probabilities"]["ai_generated"] <= 1.0
    assert data["image_metadata"]["width"] == 64
    assert data["image_metadata"]["height"] == 64
    assert data["image_metadata"]["format"] == "PNG"
    assert data["latency_ms"] > 0
    assert "request_id" in data
    assert "timestamp" in data
    assert data["status"] == "success"


def test_detect_image_valid_jpeg(valid_jpeg_bytes):
    """Verify successful inference on valid JPEG multipart upload."""
    response = client.post(
        "/api/detect/image",
        files={"file": ("photo.jpg", valid_jpeg_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["image_metadata"]["format"] == "JPEG"
    assert data["verdict"] in ["REAL", "AI_GENERATED"]


def test_detect_image_valid_webp(valid_webp_bytes):
    """Verify successful inference on valid WEBP multipart upload."""
    response = client.post(
        "/api/detect/image",
        files={"file": ("graphic.webp", valid_webp_bytes, "image/webp")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["image_metadata"]["format"] == "WEBP"


# --------------------------------------------------------------------------
# Happy Path Tests (Base64 JSON Endpoint)
# --------------------------------------------------------------------------
def test_detect_image_base64_valid(valid_png_bytes):
    """Verify successful inference on Base64 Data URI JSON payload."""
    b64_raw = base64.b64encode(valid_png_bytes).decode("ascii")
    data_uri = f"data:image/png;base64,{b64_raw}"

    response = client.post(
        "/api/detect/image/base64",
        json={"image_data": data_uri, "filename": "b64_sample.png"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "b64_sample.png"
    assert data["verdict"] in ["REAL", "AI_GENERATED"]
    assert data["status"] == "success"


# --------------------------------------------------------------------------
# Strict Validation Failure Tests (Status Codes & Error Schemas)
# --------------------------------------------------------------------------
def test_detect_image_empty_payload():
    """Rule: 0-byte file must be rejected with HTTP 400 Bad Request."""
    response = client.post(
        "/api/detect/image",
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert response.status_code == 400
    data = response.json()["detail"]
    assert data["error_code"] == "EMPTY_PAYLOAD"


def test_detect_image_unsupported_extension(valid_png_bytes):
    """Rule: Unsupported extension must be rejected with HTTP 415."""
    response = client.post(
        "/api/detect/image",
        files={"file": ("document.pdf", valid_png_bytes, "application/pdf")},
    )
    assert response.status_code == 415
    data = response.json()["detail"]
    assert data["error_code"] == "UNSUPPORTED_EXTENSION"


def test_detect_image_spoofed_magic_bytes():
    """Rule: Spoofed extension with invalid binary header rejected with HTTP 415."""
    fake_png = b"Plain text content that is masquerading as a png file."
    response = client.post(
        "/api/detect/image",
        files={"file": ("spoofed.png", fake_png, "image/png")},
    )
    assert response.status_code == 415
    data = response.json()["detail"]
    assert data["error_code"] == "INVALID_MAGIC_BYTES"


def test_detect_image_corrupted_stream():
    """Rule: Truncated/corrupted stream rejected with HTTP 422 Unprocessable Entity."""
    corrupted_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDRcorrupted_body_data"
    response = client.post(
        "/api/detect/image",
        files={"file": ("corrupt.png", corrupted_png, "image/png")},
    )
    assert response.status_code == 422
    data = response.json()["detail"]
    assert data["error_code"] == "CORRUPTED_IMAGE"


def test_detect_image_dimension_too_small():
    """Rule: Image below 16x16 minimum resolution rejected with HTTP 422."""
    buf = io.BytesIO()
    tiny_img = Image.new("RGB", (8, 8), color=(255, 0, 0))
    tiny_img.save(buf, format="PNG")
    tiny_bytes = buf.getvalue()

    response = client.post(
        "/api/detect/image",
        files={"file": ("tiny.png", tiny_bytes, "image/png")},
    )
    assert response.status_code == 422
    data = response.json()["detail"]
    assert data["error_code"] == "DIMENSION_TOO_SMALL"


def test_detect_image_oversized_payload():
    """Rule: Image exceeding 15 MB limit rejected with HTTP 413."""
    oversized_bytes = b"0" * (15 * 1024 * 1024 + 1024)  # 15 MB + 1 KB
    response = client.post(
        "/api/detect/image",
        files={"file": ("huge.png", oversized_bytes, "image/png")},
    )
    assert response.status_code == 413
    data = response.json()["detail"]
    assert data["error_code"] == "PAYLOAD_TOO_LARGE"


def test_detect_image_base64_empty():
    """Rule: Empty Base64 string rejected with HTTP 400 Bad Request."""
    response = client.post(
        "/api/detect/image/base64",
        json={"image_data": "", "filename": "empty.png"},
    )
    assert response.status_code == 400
    data = response.json()["detail"]
    assert data["error_code"] == "EMPTY_PAYLOAD"


def test_detect_image_base64_invalid_encoding():
    """Rule: Malformed Base64 string rejected with HTTP 400."""
    response = client.post(
        "/api/detect/image/base64",
        json={"image_data": "not_valid_base64_!@#$%", "filename": "bad.png"},
    )
    assert response.status_code == 400
    data = response.json()["detail"]
    assert data["error_code"] == "INVALID_BASE64"
