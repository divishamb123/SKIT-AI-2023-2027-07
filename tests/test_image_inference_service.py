"""
test_image_inference_service.py — Unit Tests for Sprint 2 Task 1 (Image Inference Service)

Member: Divisha Manak Bohra (23ESKCA038)
"""

import base64
import io

from PIL import Image
import pytest

from src.image_detector import (
    CorruptedImageError,
    EmptyImageError,
    ImageDetectionService,
    ImageInferenceResponse,
    PredictionVerdict,
)


@pytest.fixture(scope="module")
def inference_service():
    """Initializes a shared instance of ImageDetectionService."""
    return ImageDetectionService(device="cpu")


@pytest.fixture
def sample_rgb_image():
    """Generates a synthetic 64x64 RGB PIL Image."""
    return Image.new("RGB", (64, 64), color=(120, 180, 240))


@pytest.fixture
def sample_png_bytes(sample_rgb_image):
    """Encodes sample image into in-memory PNG bytes."""
    buf = io.BytesIO()
    sample_rgb_image.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def sample_base64_data_uri(sample_png_bytes):
    """Encodes sample image into a Base64 Data URI."""
    b64_str = base64.b64encode(sample_png_bytes).decode("ascii")
    return f"data:image/png;base64,{b64_str}"


def test_service_initialization(inference_service):
    """Verify service boots properly, loads model, and exposes metadata."""
    assert inference_service.model is not None
    info = inference_service.get_service_info()
    assert info.status == "healthy"
    assert "ResNet" in info.model_name
    assert "PNG" in info.supported_formats


def test_prediction_with_pil_image(inference_service, sample_rgb_image):
    """Verify inference pipeline accepting PIL Image input."""
    response = inference_service.predict(sample_rgb_image, filename="pil_test.png")
    assert isinstance(response, ImageInferenceResponse)
    assert response.filename == "pil_test.png"
    assert response.verdict in [PredictionVerdict.REAL, PredictionVerdict.AI_GENERATED]
    assert isinstance(response.is_ai, bool)
    assert 0.0 <= response.confidence <= 1.0
    assert 0.0 <= response.probabilities.real <= 1.0
    assert 0.0 <= response.probabilities.ai_generated <= 1.0
    assert (
        abs((response.probabilities.real + response.probabilities.ai_generated) - 1.0)
        < 0.01
    )
    assert response.latency_ms > 0.0
    assert response.image_metadata.width == 64
    assert response.image_metadata.height == 64


def test_prediction_with_bytes(inference_service, sample_png_bytes):
    """Verify inference pipeline accepting raw binary image bytes."""
    response = inference_service.predict(sample_png_bytes, filename="bytes_test.png")
    assert isinstance(response, ImageInferenceResponse)
    assert response.image_metadata.format == "PNG"
    assert response.image_metadata.channels == 3


def test_prediction_with_base64_uri(inference_service, sample_base64_data_uri):
    """Verify inference pipeline accepting data:image/...;base64 URI string."""
    response = inference_service.predict(
        sample_base64_data_uri, filename="b64_test.png"
    )
    assert isinstance(response, ImageInferenceResponse)
    assert response.verdict in [PredictionVerdict.REAL, PredictionVerdict.AI_GENERATED]


def test_prediction_color_space_conversions(inference_service):
    """Verify RGBA and Grayscale inputs automatically normalize to 3-channel RGB."""
    # RGBA image
    rgba_img = Image.new("RGBA", (32, 32), color=(255, 0, 0, 128))
    resp_rgba = inference_service.predict(rgba_img, filename="rgba.png")
    assert resp_rgba.image_metadata.channels == 4  # original reported channels

    # Grayscale image
    gray_img = Image.new("L", (32, 32), color=128)
    resp_gray = inference_service.predict(gray_img, filename="gray.png")
    assert resp_gray.image_metadata.channels == 1  # original reported channels


def test_error_handling_empty_payload(inference_service):
    """Verify EmptyImageError is raised when given 0-byte or empty input."""
    with pytest.raises(EmptyImageError):
        inference_service.predict(b"", filename="empty.png")


def test_error_handling_corrupted_payload(inference_service):
    """Verify CorruptedImageError is raised for non-image or corrupted bytes."""
    corrupted_data = b"RIFF\x00\x00\x00\x00WEBPVP8 \x00\x00corrupt_payload_data"
    with pytest.raises(CorruptedImageError):
        inference_service.predict(corrupted_data, filename="corrupt.webp")


def test_inference_determinism(inference_service, sample_rgb_image):
    """Verify that repeated inference runs on the identical image produce identical results."""
    resp1 = inference_service.predict(sample_rgb_image, filename="det_test.png")
    resp2 = inference_service.predict(sample_rgb_image, filename="det_test.png")

    assert resp1.verdict == resp2.verdict
    assert resp1.confidence == resp2.confidence
    assert resp1.probabilities.ai_generated == resp2.probabilities.ai_generated
    assert resp1.probabilities.real == resp2.probabilities.real


def test_response_json_serialization(inference_service, sample_rgb_image):
    """Verify standard Pydantic JSON dump and dictionary export adhere to schema."""
    resp = inference_service.predict(sample_rgb_image, filename="export_test.png")
    json_str = resp.model_dump_json()
    assert '"verdict":' in json_str
    assert '"probabilities":' in json_str
    assert '"latency_ms":' in json_str
