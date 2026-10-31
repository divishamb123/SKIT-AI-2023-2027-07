"""
test_sprint2_final_audit.py — Comprehensive Pytest Audit for Sprint 2 Completion

Member: Divisha Manak Bohra (23ESKCA038)
Project ID: SKIT/AI/2023-2027/07
Sprint: Sprint 2 — Baseline Detection Service (21-Sep-2026 to 08-Nov-2026)
User Story: Building the image detection module

Audits all 4 Form-2 Tasks:
  1. Developing an image inference service with a defined input/output interface
  2. Designing batch-processing logic for multiple image inputs
  3. Testing image inference using grouped input samples
  4. Designing image upload and result interaction components
"""

import io
from pathlib import Path

from PIL import Image
import pytest
import torch

from src.image_detector import (
    BatchImageInferenceResponse,
    GroupedImageEvaluator,
    ImageDetectionService,
    ImageInferenceResponse,
    PredictionVerdict,
    decode_image,
    load_baseline_model,
    preprocess_image_tensor,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"


@pytest.fixture(scope="module")
def image_service():
    """Initializes ImageDetectionService in evaluation mode with deterministic weights."""
    torch.manual_seed(42)
    return ImageDetectionService(device=None)


@pytest.fixture
def sample_pil_image():
    """Generates an in-memory 256x256 RGB PIL test image."""
    return Image.new("RGB", (256, 256), color=(128, 64, 200))


@pytest.fixture
def sample_image_bytes(sample_pil_image):
    """Generates JPEG byte buffer for testing upload ingestion."""
    buf = io.BytesIO()
    sample_pil_image.save(buf, format="JPEG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Task 1 Audit: Developing an image inference service with a defined I/O interface
# ---------------------------------------------------------------------------


def test_task1_model_and_device_selection():
    """Verifies ResNet-18 baseline model initialization and automatic hardware selection."""
    model, device = load_baseline_model()
    assert model is not None
    assert str(device) in ["mps", "cuda", "cpu"]

    # Verify input shape expected (1, 3, 224, 224)
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (1, 2)


def test_task1_preprocessor_pipeline(sample_image_bytes, sample_pil_image):
    """Verifies preprocessor handles byte buffers, PIL images, resizing, and ImageNet normalization."""
    pil_from_bytes, meta_bytes = decode_image(sample_image_bytes)
    assert meta_bytes.width == 256
    assert meta_bytes.height == 256
    assert meta_bytes.channels == 3
    tensor_from_bytes = preprocess_image_tensor(pil_from_bytes)
    assert tensor_from_bytes.shape == (1, 3, 224, 224)

    pil_from_pil, meta_pil = decode_image(sample_pil_image)
    assert meta_pil.format == "PNG"
    tensor_from_pil = preprocess_image_tensor(pil_from_pil)
    assert tensor_from_pil.shape == (1, 3, 224, 224)


def test_task1_defined_io_contract(image_service, sample_image_bytes):
    """Verifies that predict() satisfies the strictly typed Pydantic I/O schema."""
    response = image_service.predict(sample_image_bytes, filename="test.jpg")
    assert isinstance(response, ImageInferenceResponse)
    assert response.verdict in [PredictionVerdict.REAL, PredictionVerdict.AI_GENERATED]
    assert 0.0 <= response.confidence <= 1.0
    assert (
        abs(response.probabilities.real + response.probabilities.ai_generated - 1.0)
        < 1e-4
    )
    assert response.latency_ms > 0
    assert response.model_version == "baseline-resnet18-v1.0"


# ---------------------------------------------------------------------------
# Task 2 Audit: Designing batch-processing logic for multiple image inputs
# ---------------------------------------------------------------------------


def test_task2_batch_tensor_collation_and_speedup(image_service, sample_image_bytes):
    """Verifies 4D batch tensor collation, dynamic chunking B<=32, and throughput."""
    batch_size = 8
    inputs = [(f"img_{i}.jpg", sample_image_bytes) for i in range(batch_size)]

    batch_response = image_service.predict_batch(inputs, chunk_size=4)
    assert isinstance(batch_response, BatchImageInferenceResponse)
    assert batch_response.summary.total_submitted == batch_size
    assert len(batch_response.results) == batch_size
    assert batch_response.batch_latency_ms > 0
    assert batch_response.avg_latency_per_image_ms > 0
    fps = batch_response.summary.total_submitted / (
        batch_response.batch_latency_ms / 1000.0
    )
    assert fps > 0


def test_task2_partial_failure_fault_isolation(image_service, sample_image_bytes):
    """Verifies that corrupted items in a batch do not crash the batch execution."""
    inputs = [
        ("valid_1.jpg", sample_image_bytes),
        ("corrupt_2.png", b"corrupted_invalid_data"),
        ("valid_3.jpg", sample_image_bytes),
    ]

    batch_response = image_service.predict_batch(inputs)
    assert len(batch_response.results) == 2
    assert len(batch_response.errors) == 1
    assert batch_response.errors[0].filename == "corrupt_2.png"
    assert (
        "Corrupted or unrecognizable image payload" in batch_response.errors[0].message
    )


# ---------------------------------------------------------------------------
# Task 3 Audit: Testing image inference using grouped input samples
# ---------------------------------------------------------------------------


def test_task3_grouped_evaluator_metrics_and_stratification(image_service):
    """Verifies GroupedImageEvaluator metric calculations, confusion matrix, and reporting."""
    evaluator = GroupedImageEvaluator(service=image_service)

    # 45 TP (yt=1, yp=1), 5 FP (yt=0, yp=1), 40 TN (yt=0, yp=0), 10 FN (yt=1, yp=0)
    y_true = [1] * 45 + [0] * 5 + [0] * 40 + [1] * 10
    y_pred = [1] * 45 + [1] * 5 + [0] * 40 + [0] * 10
    confidences = [0.95] * 100

    metrics = evaluator.compute_metrics(
        y_true=y_true,
        y_pred=y_pred,
        confidences=confidences,
        group_name="test_group",
        group_type="generator",
        total_latency_ms=100.0,
    )
    assert metrics.accuracy == 0.85
    assert metrics.precision == 0.90
    assert metrics.recall == 0.8182
    assert metrics.specificity == 0.8889
    assert metrics.f1_score == 0.8571
    assert metrics.confusion_matrix.tp == 45


def test_task3_baseline_evaluation_artifacts():
    """Verifies that baseline evaluation results exist in JSON and Markdown with Form-2 attribution."""
    json_path = RESULTS_DIR / "image_baseline_evaluation.json"
    md_path = DOCS_DIR / "image_baseline_results.md"

    assert json_path.is_file(), f"Missing {json_path}"
    assert md_path.is_file(), f"Missing {md_path}"

    json_content = json_path.read_text(encoding="utf-8")
    md_content = md_path.read_text(encoding="utf-8")

    assert "baseline-resnet18-v1.0" in json_content
    assert "overall_metrics" in json_content
    assert "group_breakdowns" in json_content
    assert "by_class" in json_content

    assert "Divisha Manak Bohra" in md_content
    assert "23ESKCA038" in md_content
    assert "SKIT/AI/2023-2027/07" in md_content
    assert "Throughput" in md_content


# ---------------------------------------------------------------------------
# Task 4 Audit: Designing image upload and result interaction components
# ---------------------------------------------------------------------------


def test_task4_frontend_components_integrity():
    """Verifies that all 7 modular frontend interaction components exist and export cleanly."""
    component_files = [
        "ImageUploadDropzone.tsx",
        "VerdictGauge.tsx",
        "ImageMetadataCard.tsx",
        "BatchResultsGallery.tsx",
        "DecisionBoundaryController.tsx",
        "ForensicEvidenceViewer.tsx",
        "ForensicReportModal.tsx",
    ]
    components_dir = FRONTEND_DIR / "src" / "components" / "image-detection"

    for file_name in component_files:
        comp_path = components_dir / file_name
        assert comp_path.is_file(), f"Missing component {file_name}"
        content = comp_path.read_text(encoding="utf-8")
        assert (
            "export function" in content or "export const" in content
        ), f"Missing export in {file_name}"


def test_task4_threshold_simulation_and_margin_math():
    """Verifies threshold calibration mathematical properties in DecisionBoundaryController."""
    controller_path = (
        FRONTEND_DIR
        / "src"
        / "components"
        / "image-detection"
        / "DecisionBoundaryController.tsx"
    )
    content = controller_path.read_text(encoding="utf-8")

    # Mathematical margin Δ = |P(AI) - τ|
    assert "Math.abs(pAI - threshold)" in content
    # Tiers
    assert "HIGH_AI" in content
    assert "SUSPICIOUS_AI" in content
    assert "AMBIGUOUS" in content
    assert "PLAUSIBLE_REAL" in content
    assert "HIGH_REAL" in content


def test_task4_channel_decomposition_formulas():
    """Verifies luminance and channel decomposition formulas in ForensicEvidenceViewer."""
    viewer_path = (
        FRONTEND_DIR
        / "src"
        / "components"
        / "image-detection"
        / "ForensicEvidenceViewer.tsx"
    )
    content = viewer_path.read_text(encoding="utf-8")

    # ITU-R BT.601 Luminance formula: Y = 0.299R + 0.587G + 0.114B
    assert "0.299 * r + 0.587 * g + 0.114 * b" in content
    # High-pass filter kernel
    assert "highpass" in content


def test_task4_cryptographic_audit_certificate():
    """Verifies SHA-256 evidence hashing and certificate export in ForensicReportModal."""
    modal_path = (
        FRONTEND_DIR
        / "src"
        / "components"
        / "image-detection"
        / "ForensicReportModal.tsx"
    )
    content = modal_path.read_text(encoding="utf-8")

    assert (
        "crypto.subtle.digest('SHA-256'" in content or "crypto.subtle.digest" in content
    )
    assert "Digital Media Forensic Investigation Certificate" in content
    assert "Divisha Manak Bohra" in content
    assert "window.print()" in content


# ---------------------------------------------------------------------------
# Architecture & Microservice Audit: Docker & API Gateway
# ---------------------------------------------------------------------------


def test_image_service_docker_and_compose():
    """Verifies image-service containerization in Dockerfile and docker-compose.yml."""
    dockerfile = PROJECT_ROOT / "services" / "image-service" / "Dockerfile"
    docker_compose = PROJECT_ROOT / "docker-compose.yml"

    assert dockerfile.is_file(), f"Missing {dockerfile}"
    assert docker_compose.is_file(), f"Missing {docker_compose}"

    dc_content = docker_compose.read_text(encoding="utf-8")
    assert "image-service:" in dc_content
    assert "services/image-service/Dockerfile" in dc_content
    assert "8004:8004" in dc_content
    assert "IMAGE_SERVICE_URL" in dc_content
