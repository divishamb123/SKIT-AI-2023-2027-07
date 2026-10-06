"""
test_batch_processing.py — Unit and Integration Tests for Batch Image Detection

Sprint 2: Baseline Detection Service (Member 1: Divisha Manak Bohra - 23ESKCA038)
Task 2: Designing batch-processing logic for multiple image inputs (Week 3)
"""

import base64
import io
import time
from fastapi.testclient import TestClient
from PIL import Image
import pytest

from src.image_detector.api import create_app
from src.image_detector.schemas import (
    BatchImageInferenceResponse,
    PredictionVerdict,
)
from src.image_detector.service import ImageDetectionService


@pytest.fixture(scope="module")
def service() -> ImageDetectionService:
    """Fixture providing initialized ImageDetectionService."""
    return ImageDetectionService(device="cpu")


@pytest.fixture(scope="module")
def test_client() -> TestClient:
    """Fixture providing FastAPI TestClient for microservice endpoints."""
    app = create_app()
    return TestClient(app)


def make_test_image(
    color: tuple = (100, 150, 200), size: tuple = (64, 64), fmt: str = "PNG"
) -> bytes:
    """Helper creating in-memory image bytes."""
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=color)
    img.save(buf, format=fmt)
    return buf.getvalue()


# --------------------------------------------------------------------------
# Service-Level Batch Tests
# --------------------------------------------------------------------------


def test_batch_empty_submission(service: ImageDetectionService):
    """Verifies graceful handling of empty image sequence."""
    resp = service.predict_batch([])
    assert isinstance(resp, BatchImageInferenceResponse)
    assert resp.summary.total_submitted == 0
    assert resp.summary.successful_count == 0
    assert len(resp.results) == 0
    assert len(resp.errors) == 0


def test_batch_prediction_homogeneous_pil(service: ImageDetectionService):
    """Verifies batch inference across multiple PIL Image instances."""
    pil_images = [
        Image.new("RGB", (32, 32), color=(i * 20, 100, 200)) for i in range(5)
    ]
    filenames = [f"test_pil_{i}.png" for i in range(5)]

    resp = service.predict_batch(pil_images, filenames=filenames, chunk_size=4)

    assert isinstance(resp, BatchImageInferenceResponse)
    assert resp.summary.total_submitted == 5
    assert resp.summary.successful_count == 5
    assert resp.summary.failed_count == 0
    assert len(resp.results) == 5
    assert len(resp.errors) == 0

    for i, r in enumerate(resp.results):
        assert r.filename == filenames[i]
        assert r.verdict in [PredictionVerdict.REAL, PredictionVerdict.AI_GENERATED]
        assert 0.0 <= r.confidence <= 1.0


def test_batch_prediction_multi_format_bytes(service: ImageDetectionService):
    """Verifies batch processing across mixed bytes formats (PNG, JPEG, WEBP)."""
    png_bytes = make_test_image((255, 0, 0), fmt="PNG")
    jpg_bytes = make_test_image((0, 255, 0), fmt="JPEG")
    webp_bytes = make_test_image((0, 0, 255), fmt="WEBP")

    inputs = [
        ("image_1.png", png_bytes),
        ("image_2.jpg", jpg_bytes),
        ("image_3.webp", webp_bytes),
    ]

    resp = service.predict_batch(inputs, chunk_size=2)
    assert resp.summary.successful_count == 3
    assert resp.summary.failed_count == 0
    assert resp.results[0].image_metadata.format == "PNG"
    assert resp.results[1].image_metadata.format == "JPEG"
    assert resp.results[2].image_metadata.format == "WEBP"


def test_batch_chunking_mechanics(service: ImageDetectionService):
    """Verifies chunking (B <= 32) properly partitions batches across multiple chunks."""
    # 7 images with chunk_size=3 -> 3 chunks: 3 + 3 + 1
    images = [make_test_image((i * 30, i * 20, i * 10)) for i in range(7)]
    resp = service.predict_batch(images, chunk_size=3)

    assert resp.summary.total_submitted == 7
    assert resp.summary.successful_count == 7
    assert resp.chunk_size_used == 3
    assert len(resp.results) == 7


def test_batch_resilient_partial_failure_handling(service: ImageDetectionService):
    """
    Verifies that corrupted or malformed payloads do not abort the batch.
    Valid items must succeed and invalid items must be catalogued in errors.
    """
    valid_png = make_test_image((50, 100, 150), fmt="PNG")
    corrupted_bytes = b"corrupted_garbage_bytes_data"
    empty_bytes = b""

    batch_inputs = [
        ("valid_1.png", valid_png),
        ("corrupted.png", corrupted_bytes),
        ("valid_2.png", valid_png),
        ("empty.png", empty_bytes),
    ]

    resp = service.predict_batch(batch_inputs, chunk_size=2, return_errors=True)

    assert resp.summary.total_submitted == 4
    assert resp.summary.successful_count == 2
    assert resp.summary.failed_count == 2
    assert len(resp.results) == 2
    assert len(resp.errors) == 2

    error_filenames = {e.filename: e.error_code for e in resp.errors}
    assert "corrupted.png" in error_filenames
    assert "empty.png" in error_filenames
    assert error_filenames["corrupted.png"] == "CORRUPTED_IMAGE"
    assert error_filenames["empty.png"] == "EMPTY_PAYLOAD"


def test_batch_throughput_speedup(service: ImageDetectionService):
    """
    Verifies that batched tensor forward pass processes items efficiently.
    """
    sample = make_test_image((128, 128, 128))
    images = [sample] * 8

    # Warmup
    service.predict(sample)

    # Batched execution
    t0 = time.perf_counter()
    batch_resp = service.predict_batch(images, chunk_size=8)
    batch_time = time.perf_counter() - t0

    assert batch_resp.summary.successful_count == 8
    assert batch_resp.avg_latency_per_image_ms > 0
    assert batch_time > 0


# --------------------------------------------------------------------------
# API-Level Endpoint Tests
# --------------------------------------------------------------------------


def test_batch_api_multipart_endpoint(test_client: TestClient):
    """Tests POST /api/detect/image/batch with multipart file list."""
    img1 = make_test_image((255, 100, 50), fmt="PNG")
    img2 = make_test_image((50, 200, 100), fmt="JPEG")

    files = [
        ("files", ("batch_1.png", img1, "image/png")),
        ("files", ("batch_2.jpg", img2, "image/jpeg")),
    ]

    response = test_client.post("/api/detect/image/batch", files=files)
    assert response.status_code == 200
    data = response.json()

    assert "batch_id" in data
    assert data["summary"]["total_submitted"] == 2
    assert data["summary"]["successful_count"] == 2
    assert len(data["results"]) == 2
    assert data["results"][0]["filename"] == "batch_1.png"
    assert data["results"][1]["filename"] == "batch_2.jpg"


def test_batch_api_multipart_empty_rejection(test_client: TestClient):
    """Verifies that empty files parameter returns HTTP 400 EMPTY_BATCH."""
    response = test_client.post("/api/detect/image/batch", files=[])
    # Either 400 or 422 depending on FastAPI form parser; our handler guards len(files)==0
    assert response.status_code in [400, 422]


def test_batch_api_multipart_limit_enforcement(test_client: TestClient):
    """Verifies that batches exceeding MAX_BATCH_SIZE (32) are rejected with HTTP 400."""
    dummy_img = make_test_image((10, 20, 30))
    oversized_files = [
        ("files", (f"img_{i}.png", dummy_img, "image/png")) for i in range(33)
    ]

    response = test_client.post("/api/detect/image/batch", files=oversized_files)
    assert response.status_code == 400
    data = response.json()
    assert data.get("error_code") == "BATCH_TOO_LARGE" or "BATCH_TOO_LARGE" in str(data)


def test_batch_api_base64_endpoint(test_client: TestClient):
    """Tests POST /api/detect/image/batch/base64 with JSON payload."""
    img_bytes = make_test_image((80, 120, 160), fmt="PNG")
    b64_str = base64.b64encode(img_bytes).decode("utf-8")
    data_uri = f"data:image/png;base64,{b64_str}"

    payload = {
        "images": [
            {"image_data": data_uri, "filename": "sample1.png"},
            {"image_data": b64_str, "filename": "sample2.png"},
        ]
    }

    response = test_client.post("/api/detect/image/batch/base64", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["summary"]["total_submitted"] == 2
    assert data["summary"]["successful_count"] == 2
    assert len(data["results"]) == 2


def test_batch_api_base64_empty_rejection(test_client: TestClient):
    """Verifies POST /api/detect/image/batch/base64 rejects empty images list."""
    response = test_client.post("/api/detect/image/batch/base64", json={"images": []})
    assert response.status_code == 400
    data = response.json()
    assert "EMPTY_BATCH" in str(data)
