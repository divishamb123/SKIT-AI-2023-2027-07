"""
test_image_ensemble.py — Automated Unit & Integration Tests for Image Ensemble Module

Sprint 3 (Accuracy Optimisation) — Week 1 Test Suite
Member: Divisha Manak Bohra (23ESKCA038)
"""

import base64
import io

from PIL import Image
import pytest
import torch

from src.image_detector.ensemble import (
    DEFAULT_WEIGHTS,
    ENSEMBLE_VERSION,
    ImageEnsembleDetector,
    MobileNetV3Classifier,
)
from src.image_detector.schemas import (
    BatchImageInferenceResponse,
    ImageInferenceResponse,
    PredictionVerdict,
)


@pytest.fixture(scope="module")
def ensemble_detector():
    """Initializes a shared ImageEnsembleDetector instance."""
    return ImageEnsembleDetector(weights=DEFAULT_WEIGHTS)


@pytest.fixture
def sample_pil_image():
    """Creates a sample 224x224 RGB image for testing."""
    img = Image.new("RGB", (224, 224), color=(70, 130, 180))
    return img


@pytest.fixture
def sample_image_bytes(sample_pil_image):
    """Creates PNG bytes from sample PIL image."""
    buf = io.BytesIO()
    sample_pil_image.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def sample_base64_uri(sample_image_bytes):
    """Creates an RFC 2397 Data URI from sample image bytes."""
    encoded = base64.b64encode(sample_image_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


# --------------------------------------------------------------------------
# 1. Architecture & Weight Initialization Tests
# --------------------------------------------------------------------------
def test_ensemble_initialization_and_weights(ensemble_detector):
    assert ensemble_detector.ensemble_version == ENSEMBLE_VERSION
    assert ensemble_detector.resnet is not None
    assert ensemble_detector.mobilenet is not None

    weights = ensemble_detector.weights
    assert "resnet18" in weights and "mobilenet_v3" in weights
    assert pytest.approx(sum(weights.values()), rel=1e-3) == 1.0


def test_ensemble_weight_normalization():
    # Pass arbitrary un-normalized weights (e.g. 70 and 30)
    detector = ImageEnsembleDetector(weights={"resnet18": 7.0, "mobilenet_v3": 3.0})
    assert detector.weights["resnet18"] == 0.7
    assert detector.weights["mobilenet_v3"] == 0.3
    assert pytest.approx(sum(detector.weights.values()), rel=1e-3) == 1.0


def test_mobilenet_v3_classifier_forward():
    classifier = MobileNetV3Classifier(num_classes=2)
    dummy_input = torch.randn(2, 3, 224, 224)
    logits = classifier(dummy_input)
    assert logits.shape == (2, 2)
    assert not torch.isnan(logits).any()


# --------------------------------------------------------------------------
# 2. Single Image Inference & Schema Tests
# --------------------------------------------------------------------------
def test_single_prediction_with_pil(ensemble_detector, sample_pil_image):
    response = ensemble_detector.predict(sample_pil_image, filename="test_sample.png")

    assert isinstance(response, ImageInferenceResponse)
    assert response.filename == "test_sample.png"
    assert response.verdict in (PredictionVerdict.REAL, PredictionVerdict.AI_GENERATED)
    assert 0.0 <= response.confidence <= 1.0
    assert 0.0 <= response.probabilities.real <= 1.0
    assert 0.0 <= response.probabilities.ai_generated <= 1.0
    assert pytest.approx(response.probabilities.real + response.probabilities.ai_generated, rel=1e-3) == 1.0

    # Verify Ensemble Metadata
    assert response.ensemble_metadata is not None
    assert response.ensemble_metadata.ensemble_strategy == "weighted_soft_voting"
    assert len(response.ensemble_metadata.submodels) == 2
    assert 0.0 <= response.ensemble_metadata.consensus_score <= 1.0
    assert response.ensemble_metadata.disagreement_variance >= 0.0


def test_single_prediction_with_bytes(ensemble_detector, sample_image_bytes):
    response = ensemble_detector.predict(sample_image_bytes, filename="bytes_test.png")
    assert response.verdict in (PredictionVerdict.REAL, PredictionVerdict.AI_GENERATED)
    assert response.image_metadata.channels == 3
    assert response.latency_ms > 0.0


def test_single_prediction_with_base64_uri(ensemble_detector, sample_base64_uri):
    response = ensemble_detector.predict(sample_base64_uri, filename="b64_test.png")
    assert response.verdict in (PredictionVerdict.REAL, PredictionVerdict.AI_GENERATED)
    assert response.image_metadata.format == "PNG"


# --------------------------------------------------------------------------
# 3. Mathematical Fusion & Consensus Verification
# --------------------------------------------------------------------------
def test_soft_voting_arithmetic(ensemble_detector, sample_pil_image):
    resp = ensemble_detector.predict(sample_pil_image)
    meta = resp.ensemble_metadata

    sub_res = next(s for s in meta.submodels if s.model_name == "resnet18")
    sub_mob = next(s for s in meta.submodels if s.model_name == "mobilenet_v3")

    expected_ai = (sub_res.weight * sub_res.probabilities.ai_generated) + (
        sub_mob.weight * sub_mob.probabilities.ai_generated
    )
    assert pytest.approx(resp.probabilities.ai_generated, abs=1e-3) == expected_ai


def test_decision_threshold_tuning(ensemble_detector, sample_pil_image):
    # Test strict threshold behavior
    resp_low_tau = ensemble_detector.predict(sample_pil_image, threshold=0.01)
    resp_high_tau = ensemble_detector.predict(sample_pil_image, threshold=0.99)

    # When tau = 0.01, almost anything is classified as AI
    assert resp_low_tau.verdict == PredictionVerdict.AI_GENERATED
    # When tau = 0.99, almost anything is classified as REAL
    assert resp_high_tau.verdict == PredictionVerdict.REAL


# --------------------------------------------------------------------------
# 4. Batch Collation & Error Handling Tests
# --------------------------------------------------------------------------
def test_batch_prediction_collation(ensemble_detector, sample_pil_image):
    images = [sample_pil_image for _ in range(6)]
    filenames = [f"sample_{i}.png" for i in range(6)]

    batch_resp = ensemble_detector.predict_batch(images, filenames=filenames, chunk_size=4)

    assert isinstance(batch_resp, BatchImageInferenceResponse)
    assert batch_resp.summary.total_submitted == 6
    assert batch_resp.summary.successful_count == 6
    assert batch_resp.summary.failed_count == 0
    assert len(batch_resp.results) == 6
    assert batch_resp.batch_latency_ms > 0.0


def test_batch_prediction_fault_isolation(ensemble_detector, sample_pil_image):
    # Create mixed batch: 2 valid images, 1 corrupt byte payload, 1 valid image
    mixed_batch = [
        sample_pil_image,
        sample_pil_image,
        b"CORRUPTED_NOT_AN_IMAGE_PAYLOAD",
        sample_pil_image,
    ]
    filenames = ["valid_1.png", "valid_2.png", "corrupt.png", "valid_3.png"]

    batch_resp = ensemble_detector.predict_batch(mixed_batch, filenames=filenames)

    assert batch_resp.summary.total_submitted == 4
    assert batch_resp.summary.successful_count == 3
    assert batch_resp.summary.failed_count == 1
    assert len(batch_resp.errors) == 1
    assert batch_resp.errors[0].filename == "corrupt.png"
    assert batch_resp.errors[0].index == 2


def test_dynamic_chunking_large_batch(ensemble_detector, sample_pil_image):
    # 35 items partitioned into chunks of 16 (16 + 16 + 3)
    large_batch = [sample_pil_image for _ in range(35)]
    batch_resp = ensemble_detector.predict_batch(large_batch, chunk_size=16)

    assert batch_resp.summary.total_submitted == 35
    assert batch_resp.summary.successful_count == 35
    assert batch_resp.summary.failed_count == 0
    assert len(batch_resp.results) == 35
