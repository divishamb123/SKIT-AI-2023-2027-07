"""
verify_sprint2_week4.py — Standalone Verification Suite for Sprint 2 Week 4 Deliverables

Student: Divisha Manak Bohra (23ESKCA038)
Sprint: Sprint 2 — Baseline Detection Service
User Story: Building the image detection module
Task 3: Testing image inference using grouped input samples (Week 4)
"""

import io
from pathlib import Path
import sys

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.image_detector.evaluator import GroupedImageEvaluator
from src.image_detector.schemas import (
    ConfusionMatrixMetrics,
    GroupEvaluationMetrics,
    GroupedEvaluationReport,
)
from src.image_detector.service import ImageDetectionService

MANIFEST_PATH = PROJECT_ROOT / "datasets" / "splits" / "manifest.csv"
RESULTS_JSON_PATH = PROJECT_ROOT / "results" / "image_baseline_evaluation.json"
RESULTS_MD_PATH = PROJECT_ROOT / "docs" / "image_baseline_results.md"


def print_banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def create_synthetic_image(color: tuple = (100, 150, 200), size: tuple = (64, 64), fmt: str = "PNG") -> bytes:
    """Helper creating in-memory image bytes."""
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=color)
    img.save(buf, format=fmt)
    return buf.getvalue()


def main() -> int:
    print_banner("SPRINT 2 WEEK 4 — GROUPED INPUT SAMPLES EVALUATION VERIFICATION")
    print("Member:    Divisha Manak Bohra (23ESKCA038)")
    print("Story:     Building the image detection module")
    print("Task 3:    Testing image inference using grouped input samples (Week 4)")

    passed_checks = 0
    total_checks = 8

    # ----------------------------------------------------------------------
    # Check 1: Grouped Evaluation Schemas & Schema Integrity
    # ----------------------------------------------------------------------
    print("\n[Check 1/8] Verifying Grouped Evaluation Schemas & Contract Models...")
    try:
        cm = ConfusionMatrixMetrics(tp=10, fp=2, tn=12, fn=1)
        assert cm.tp == 10 and cm.fp == 2 and cm.tn == 12 and cm.fn == 1

        gm = GroupEvaluationMetrics(
            group_name="Test Generator",
            group_type="generator",
            sample_count=25,
            successful_count=25,
            failed_count=0,
            accuracy=0.88,
            precision=0.8333,
            recall=0.9091,
            specificity=0.8571,
            f1_score=0.8696,
            mean_confidence=0.9412,
            confusion_matrix=cm,
            total_latency_ms=125.4,
            avg_latency_ms=5.02,
            throughput_fps=199.36,
        )
        assert gm.accuracy == 0.88
        assert gm.sample_count == 25

        report = GroupedEvaluationReport(
            model_version="baseline-resnet18-v1.0",
            device="cpu",
            overall_metrics=gm,
            group_breakdowns={"by_generator": [gm]},
        )
        assert report.report_id is not None
        assert report.overall_metrics.f1_score == 0.8696

        print("  ✓ ConfusionMatrixMetrics schema verified")
        print("  ✓ GroupEvaluationMetrics schema verified")
        print("  ✓ GroupedEvaluationReport schema verified")
        passed_checks += 1
    except Exception as e:
        print(f"  ✗ FAILED Check 1: {e}")
        return 1

    # ----------------------------------------------------------------------
    # Check 2: Core Evaluator Engine Initialization & Metric Math
    # ----------------------------------------------------------------------
    print("\n[Check 2/8] Verifying Evaluator Initialization & Mathematical Invariants...")
    try:
        service = ImageDetectionService(device="cpu")
        evaluator = GroupedImageEvaluator(service=service)

        # Metric calculation invariant check:
        # 10 TP, 5 FP, 20 TN, 5 FN -> Total = 40
        # Accuracy = (10+20)/40 = 0.75
        # Precision = 10/(10+5) = 0.6667
        # Recall = 10/(10+5) = 0.6667
        # Specificity = 20/(20+5) = 0.8000
        # F1 = 0.6667
        y_true = [1] * 15 + [0] * 25
        y_pred = [1] * 10 + [0] * 5 + [1] * 5 + [0] * 20
        confs = [0.9] * 40
        computed = evaluator.compute_metrics(
            y_true=y_true,
            y_pred=y_pred,
            confidences=confs,
            group_name="Math Test",
            group_type="validation",
            total_latency_ms=200.0,
        )

        assert computed.accuracy == 0.75, f"Expected acc 0.75, got {computed.accuracy}"
        assert computed.specificity == 0.8, f"Expected spec 0.8, got {computed.specificity}"
        assert computed.confusion_matrix.tp == 10
        assert computed.confusion_matrix.fp == 5
        assert computed.confusion_matrix.tn == 20
        assert computed.confusion_matrix.fn == 5
        print("  ✓ Evaluator bound to service model: " + service.model_version)
        print("  ✓ Classification metric invariants mathematically confirmed")
        passed_checks += 1
    except Exception as e:
        print(f"  ✗ FAILED Check 2: {e}")
        return 1

    # ----------------------------------------------------------------------
    # Check 3: Grouped In-Domain Test Partition Ingestion (CIFAKE)
    # ----------------------------------------------------------------------
    print("\n[Check 3/8] Loading and Grouping In-Domain Test Samples (CIFAKE)...")
    try:
        evaluator = GroupedImageEvaluator()
        test_samples = evaluator.load_manifest_samples(
            MANIFEST_PATH, split="test", limit_per_class=25, balance_classes=True
        )
        assert len(test_samples) == 50, f"Expected 50 samples, got {len(test_samples)}"
        real_count = sum(1 for s in test_samples if s["label"] == 0)
        ai_count = sum(1 for s in test_samples if s["label"] == 1)
        assert real_count == 25 and ai_count == 25, "Class balance invariant violated"

        # Check path existence
        for s in test_samples[:5]:
            assert Path(s["path"]).exists(), f"Path does not exist: {s['path']}"

        print(f"  ✓ In-Domain test partition successfully sampled: {len(test_samples)} items")
        print("  ✓ Exact 50/50 class balance confirmed (25 Real : 25 AI)")
        passed_checks += 1
    except Exception as e:
        print(f"  ✗ FAILED Check 3: {e}")
        return 1

    # ----------------------------------------------------------------------
    # Check 4: Out-of-Domain Holdout Partition Stratification (GenImage)
    # ----------------------------------------------------------------------
    print("\n[Check 4/8] Loading and Stratifying Out-of-Domain Holdout Samples (GenImage)...")
    try:
        evaluator = GroupedImageEvaluator()
        holdout_samples = evaluator.load_manifest_samples(
            MANIFEST_PATH, split="holdout", limit_per_generator=15
        )
        generators = set(s["generator"] for s in holdout_samples)
        assert "midjourney" in generators, "Missing midjourney in holdout"
        assert "biggan" in generators, "Missing biggan in holdout"
        assert "stable-diffusion" in generators, "Missing stable-diffusion in holdout"
        assert "real" in generators, "Missing real in holdout"

        print(f"  ✓ Out-of-domain holdout partition loaded: {len(holdout_samples)} items")
        print(f"  ✓ Stratified generators present: {sorted(list(generators))}")
        passed_checks += 1
    except Exception as e:
        print(f"  ✗ FAILED Check 4: {e}")
        return 1

    # ----------------------------------------------------------------------
    # Check 5: Grouped Inference Execution & Multi-Dimensional Breakdowns
    # ----------------------------------------------------------------------
    print("\n[Check 5/8] Executing Grouped Inference & Generating Multi-Dimensional Breakdowns...")
    try:
        evaluator = GroupedImageEvaluator()
        pool = test_samples[:20] + holdout_samples[:20]
        report = evaluator.evaluate_grouped_dataset(
            pool, grouping_keys=["generator", "split"], chunk_size=8
        )

        assert report.overall_metrics.successful_count == 40
        assert "by_class" in report.group_breakdowns
        assert "by_generator" in report.group_breakdowns
        assert "by_split" in report.group_breakdowns
        print(f"  ✓ Evaluated total samples:     {report.overall_metrics.successful_count}")
        print(f"  ✓ Overall Throughput:           {report.overall_metrics.throughput_fps:.1f} FPS")
        print(f"  ✓ Generator groups evaluated:   {len(report.group_breakdowns['by_generator'])}")
        print(f"  ✓ Split groups evaluated:       {len(report.group_breakdowns['by_split'])}")
        passed_checks += 1
    except Exception as e:
        print(f"  ✗ FAILED Check 5: {e}")
        return 1

    # ----------------------------------------------------------------------
    # Check 6: Group Batch Scaling & Throughput Profiling
    # ----------------------------------------------------------------------
    print("\n[Check 6/8] Profiling Batch Collation Scaling Across Group Sizes (1, 4, 8, 16)...")
    try:
        evaluator = GroupedImageEvaluator()
        scaling_pool = test_samples[:16]
        scaling_metrics = evaluator.benchmark_group_sizes(scaling_pool, group_sizes=[1, 4, 8, 16])

        assert len(scaling_metrics) == 4
        print("  ✓ Group Size 1:  Latency = " + f"{scaling_metrics[0].total_latency_ms:.1f} ms | Throughput = {scaling_metrics[0].throughput_fps:.1f} FPS")
        print("  ✓ Group Size 4:  Latency = " + f"{scaling_metrics[1].total_latency_ms:.1f} ms | Throughput = {scaling_metrics[1].throughput_fps:.1f} FPS")
        print("  ✓ Group Size 8:  Latency = " + f"{scaling_metrics[2].total_latency_ms:.1f} ms | Throughput = {scaling_metrics[2].throughput_fps:.1f} FPS")
        print("  ✓ Group Size 16: Latency = " + f"{scaling_metrics[3].total_latency_ms:.1f} ms | Throughput = {scaling_metrics[3].throughput_fps:.1f} FPS")
        passed_checks += 1
    except Exception as e:
        print(f"  ✗ FAILED Check 6: {e}")
        return 1

    # ----------------------------------------------------------------------
    # Check 7: Fault-Isolated Group Evaluation (Corrupted Image Resilience)
    # ----------------------------------------------------------------------
    print("\n[Check 7/8] Verifying Group Fault Isolation with Mixed Corrupted Inputs...")
    try:
        evaluator = GroupedImageEvaluator()
        scratch_dir = PROJECT_ROOT / "scratch_test_group"
        scratch_dir.mkdir(exist_ok=True)
        valid_img = scratch_dir / "valid_group_img.png"
        corrupt_img = scratch_dir / "corrupt_group_img.png"

        valid_img.write_bytes(create_synthetic_image())
        corrupt_img.write_bytes(b"INVALID_HEADER_GARBAGE")

        mixed_group = [
            {"path": str(valid_img), "filename": "valid_group_img.png", "label": 0},
            {"path": str(corrupt_img), "filename": "corrupt_group_img.png", "label": 1},
        ]

        group_metrics = evaluator.evaluate_group(
            mixed_group, group_name="Resilience Group", group_type="fault_test", chunk_size=2
        )

        assert group_metrics.sample_count == 2
        assert group_metrics.successful_count == 1
        assert group_metrics.failed_count == 1
        print("  ✓ Group completed without unhandled exceptions")
        print("  ✓ 1 valid sample scored, 1 corrupt sample safely isolated in failed_count")

        # Cleanup scratch files
        valid_img.unlink(missing_ok=True)
        corrupt_img.unlink(missing_ok=True)
        scratch_dir.rmdir()

        passed_checks += 1
    except Exception as e:
        print(f"  ✗ FAILED Check 7: {e}")
        return 1

    # ----------------------------------------------------------------------
    # Check 8: Artifact Generation & Persistence
    # ----------------------------------------------------------------------
    print("\n[Check 8/8] Verifying Evaluation Artifacts (JSON & Markdown Documentation)...")
    try:
        assert RESULTS_JSON_PATH.is_file(), f"Missing {RESULTS_JSON_PATH}"
        assert RESULTS_MD_PATH.is_file(), f"Missing {RESULTS_MD_PATH}"

        json_bytes = RESULTS_JSON_PATH.stat().st_size
        md_bytes = RESULTS_MD_PATH.stat().st_size
        assert json_bytes > 500, f"JSON report too small ({json_bytes} bytes)"
        assert md_bytes > 500, f"Markdown report too small ({md_bytes} bytes)"

        md_text = RESULTS_MD_PATH.read_text(encoding="utf-8")
        assert "Divisha Manak Bohra" in md_text
        assert "Grouped Breakdown Analysis" in md_text
        assert "Overall Accuracy" in md_text

        print(f"  ✓ Verified results JSON: {RESULTS_JSON_PATH.name} ({json_bytes:,} bytes)")
        print(f"  ✓ Verified results Markdown: {RESULTS_MD_PATH.name} ({md_bytes:,} bytes)")
        passed_checks += 1
    except Exception as e:
        print(f"  ✗ FAILED Check 8: {e}")
        return 1

    # ----------------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------------
    print_banner(f"VERIFICATION COMPLETE: {passed_checks}/{total_checks} CHECKS PASSED (100%)")
    print("  Sprint:     Sprint 2 — Baseline Detection Service")
    print("  Week:       Week 4")
    print("  Member:     Divisha Manak Bohra (23ESKCA038)")
    print("  Task 3:     Testing image inference using grouped input samples")
    print("  Status:     ALL CHECKS PASSED — READY FOR FORM-3 REPORTING\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
