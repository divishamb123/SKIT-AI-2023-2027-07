"""
test_grouped_image_evaluation.py — Test Suite for Grouped Image Inference and Evaluation

Sprint 2: Baseline Detection Service (Member 1: Divisha Manak Bohra - 23ESKCA038)
Task 3: Testing image inference using grouped input samples (Week 4)
"""

import io
from pathlib import Path
from PIL import Image
import pytest

from src.image_detector.evaluator import GroupedImageEvaluator
from src.image_detector.schemas import (
    GroupedEvaluationReport,
)
from src.image_detector.service import ImageDetectionService

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "datasets" / "splits" / "manifest.csv"


@pytest.fixture(scope="module")
def service() -> ImageDetectionService:
    """Fixture providing CPU-bound ImageDetectionService for fast deterministic tests."""
    return ImageDetectionService(device="cpu")


@pytest.fixture(scope="module")
def evaluator(service: ImageDetectionService) -> GroupedImageEvaluator:
    """Fixture providing initialized GroupedImageEvaluator."""
    return GroupedImageEvaluator(service=service, default_chunk_size=8)


def create_in_memory_image(
    color: tuple = (120, 180, 240), size: tuple = (64, 64), fmt: str = "PNG"
) -> bytes:
    """Helper creating in-memory image bytes."""
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=color)
    img.save(buf, format=fmt)
    return buf.getvalue()


# --------------------------------------------------------------------------
# Grouped Evaluation Unit Tests
# --------------------------------------------------------------------------


def test_evaluator_initialization(evaluator: GroupedImageEvaluator):
    """Verifies evaluator service linkage and default configuration."""
    assert evaluator.service is not None
    assert evaluator.default_chunk_size == 8
    assert evaluator.service.model_version == "baseline-resnet18-v1.0"


def test_compute_metrics_perfect_predictions(evaluator: GroupedImageEvaluator):
    """Tests metric calculation when all predictions are correct."""
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    confs = [0.95, 0.90, 0.99, 0.88]

    metrics = evaluator.compute_metrics(
        y_true=y_true,
        y_pred=y_pred,
        confidences=confs,
        group_name="Perfect Group",
        group_type="unit_test",
        total_latency_ms=100.0,
    )

    assert metrics.accuracy == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.specificity == 1.0
    assert metrics.f1_score == 1.0
    assert metrics.confusion_matrix.tp == 2
    assert metrics.confusion_matrix.tn == 2
    assert metrics.confusion_matrix.fp == 0
    assert metrics.confusion_matrix.fn == 0
    assert metrics.sample_count == 4
    assert metrics.throughput_fps == 40.0


def test_compute_metrics_skewed_predictions(evaluator: GroupedImageEvaluator):
    """Tests metric calculations under imperfect / false positive predictions."""
    y_true = [0, 0, 1, 1]
    y_pred = [1, 0, 1, 0]  # 1 FP, 1 FN, 1 TP, 1 TN
    confs = [0.80, 0.85, 0.90, 0.70]

    metrics = evaluator.compute_metrics(
        y_true=y_true,
        y_pred=y_pred,
        confidences=confs,
        group_name="Mixed Group",
        group_type="unit_test",
        total_latency_ms=200.0,
    )

    assert metrics.accuracy == 0.5
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.specificity == 0.5
    assert metrics.f1_score == 0.5
    assert metrics.confusion_matrix.tp == 1
    assert metrics.confusion_matrix.tn == 1
    assert metrics.confusion_matrix.fp == 1
    assert metrics.confusion_matrix.fn == 1


def test_load_manifest_samples(evaluator: GroupedImageEvaluator):
    """Tests reading, filtering, and class-balancing from manifest.csv."""
    assert MANIFEST_PATH.is_file(), "manifest.csv must exist"

    samples = evaluator.load_manifest_samples(
        MANIFEST_PATH, split="test", limit_per_class=10, balance_classes=True
    )

    assert len(samples) == 20
    reals = [s for s in samples if s["label"] == 0]
    ais = [s for s in samples if s["label"] == 1]
    assert len(reals) == 10
    assert len(ais) == 10
    assert all("path" in s and "generator" in s for s in samples)


def test_load_manifest_stratified_generators(evaluator: GroupedImageEvaluator):
    """Tests loading holdout split stratified across all unseen generators."""
    samples = evaluator.load_manifest_samples(
        MANIFEST_PATH, split="holdout", limit_per_generator=5
    )

    generators = set(s["generator"] for s in samples)
    assert "midjourney" in generators
    assert "biggan" in generators
    assert "stable-diffusion" in generators
    assert "real" in generators


def test_group_samples_by_key(evaluator: GroupedImageEvaluator):
    """Verifies sample partitioning by arbitrary metadata keys."""
    sample_data = [
        {"generator": "midjourney", "label": 1},
        {"generator": "biggan", "label": 1},
        {"generator": "midjourney", "label": 1},
        {"generator": "cifar10", "label": 0},
    ]

    grouped_gen = evaluator.group_samples_by(sample_data, "generator")
    assert len(grouped_gen["midjourney"]) == 2
    assert len(grouped_gen["biggan"]) == 1
    assert len(grouped_gen["cifar10"]) == 1

    grouped_label = evaluator.group_samples_by(sample_data, "label")
    assert len(grouped_label["1"]) == 3
    assert len(grouped_label["0"]) == 1


def test_evaluate_group_on_manifest_images(evaluator: GroupedImageEvaluator):
    """Evaluates real image files sampled from CIFAKE test split."""
    samples = evaluator.load_manifest_samples(
        MANIFEST_PATH, split="test", limit_per_class=6, balance_classes=True
    )

    metrics = evaluator.evaluate_group(
        samples, group_name="CIFAKE Test Sample", group_type="split", chunk_size=4
    )

    assert metrics.sample_count == 12
    assert metrics.successful_count == 12
    assert metrics.failed_count == 0
    assert metrics.throughput_fps > 0
    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.mean_confidence <= 1.0


def test_evaluate_grouped_dataset_breakdowns(evaluator: GroupedImageEvaluator):
    """Verifies multi-dimensional breakdown report generation."""
    samples = evaluator.load_manifest_samples(
        MANIFEST_PATH, split="test", limit_per_class=8, balance_classes=True
    )

    report = evaluator.evaluate_grouped_dataset(
        samples, grouping_keys=["generator", "split"], chunk_size=4
    )

    assert isinstance(report, GroupedEvaluationReport)
    assert report.overall_metrics.successful_count == 16
    assert "by_class" in report.group_breakdowns
    assert "by_generator" in report.group_breakdowns
    assert "by_split" in report.group_breakdowns
    assert len(report.group_breakdowns["by_class"]) == 2


def test_benchmark_group_sizes(evaluator: GroupedImageEvaluator):
    """Tests throughput and latency profiling across different group sizes."""
    samples = evaluator.load_manifest_samples(
        MANIFEST_PATH, split="test", limit_per_class=8, balance_classes=True
    )

    scaling_results = evaluator.benchmark_group_sizes(samples, group_sizes=[1, 4, 8])
    assert len(scaling_results) == 3
    for res in scaling_results:
        assert res.successful_count == 16
        assert res.throughput_fps > 0
        assert res.total_latency_ms > 0


def test_fault_tolerant_grouped_evaluation(
    evaluator: GroupedImageEvaluator, tmp_path: Path
):
    """Verifies that inaccessible / corrupt images in a group do not crash evaluation."""
    # Create 2 valid images and 1 corrupted file
    img1_path = tmp_path / "valid1.png"
    img2_path = tmp_path / "valid2.png"
    corrupt_path = tmp_path / "corrupt.png"

    img1_path.write_bytes(create_in_memory_image((255, 0, 0)))
    img2_path.write_bytes(create_in_memory_image((0, 255, 0)))
    corrupt_path.write_bytes(b"NOT_A_VALID_IMAGE_HEADER")

    mixed_samples = [
        {"path": str(img1_path), "filename": "valid1.png", "label": 0},
        {"path": str(corrupt_path), "filename": "corrupt.png", "label": 1},
        {"path": str(img2_path), "filename": "valid2.png", "label": 0},
    ]

    metrics = evaluator.evaluate_group(
        mixed_samples,
        group_name="Fault Tolerance Test",
        group_type="robustness",
        chunk_size=2,
    )

    assert metrics.sample_count == 3
    assert metrics.successful_count == 2
    assert metrics.failed_count == 1
    assert metrics.confusion_matrix.tn + metrics.confusion_matrix.fp == 2


def test_report_serialization_and_markdown(
    evaluator: GroupedImageEvaluator, tmp_path: Path
):
    """Verifies report Markdown formatting and JSON saving."""
    samples = evaluator.load_manifest_samples(
        MANIFEST_PATH, split="test", limit_per_class=4, balance_classes=True
    )
    report = evaluator.evaluate_grouped_dataset(samples, grouping_keys=["generator"])

    md_path = tmp_path / "test_report.md"
    json_path = tmp_path / "test_report.json"

    evaluator.save_report(report, json_path=json_path, md_path=md_path)

    assert json_path.is_file()
    assert md_path.is_file()
    md_content = md_path.read_text(encoding="utf-8")
    assert "Image Detection Baseline Evaluation Report" in md_content
    assert "Divisha Manak Bohra" in md_content
    assert "ResNet-18" in md_content
