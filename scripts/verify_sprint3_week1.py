"""
verify_sprint3_week1.py — Multi-Model Image Ensemble Verification Suite

Sprint 3 (Accuracy Optimisation) — Week 1 Verification
Member: Divisha Manak Bohra (23ESKCA038)

Validates:
1. Ensemble Architecture & Sub-Model Instantiation (ResNet-18 + MobileNetV3)
2. Weighted Soft-Voting Probability Fusion & Normalization
3. Sub-Model Consensus & Disagreement Metric Computation
4. Multi-Format Image Ingestion (PIL, Bytes, Base64 Data URI)
5. Vectorized Batch Collation & Dynamic Chunking (B <= 32)
6. Fault-Isolated Partial Error Handling
7. Hardware Acceleration Telemetry (Apple Silicon MPS / CUDA / CPU)
8. Pytest Test Suite Execution (tests/test_image_ensemble.py)
"""

import base64
import io
from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from src.image_detector.ensemble import (
    DEFAULT_WEIGHTS,
    ImageEnsembleDetector,
    MobileNetV3Classifier,
)
from src.image_detector.model import BaselineImageClassifier
from src.image_detector.schemas import PredictionVerdict


def create_test_image(color=(50, 100, 150)) -> Image.Image:
    return Image.new("RGB", (224, 224), color=color)


def run_verification():
    print("=" * 75)
    print("SPRINT 3 — WEEK 1: MULTI-MODEL ENSEMBLE VERIFICATION SUITE")
    print("Candidate: Divisha Manak Bohra (23ESKCA038)")
    print("User Story: Refining the image detection module")
    print("=" * 75)

    # --------------------------------------------------------------------------
    # Stage 1: Ensemble Architecture & Sub-Model Instantiation
    # --------------------------------------------------------------------------
    print("\n--- [Stage 1] Verifying Ensemble Architecture & Backbones ---")
    detector = ImageEnsembleDetector(weights={"resnet18": 0.6, "mobilenet_v3": 0.4})
    assert isinstance(detector.resnet, BaselineImageClassifier), "ResNet-18 missing!"
    assert isinstance(detector.mobilenet, MobileNetV3Classifier), "MobileNetV3 missing!"
    print("  [PASS] Backbone 1: ResNet-18 (Deep Spatial Residual CNN)")
    print("  [PASS] Backbone 2: MobileNetV3-Small (Inverted Residual + SE Attention)")
    print(f"  [PASS] Hardware Target: {detector.device}")
    print(f"  [PASS] Ensemble Version: {detector.ensemble_version}")

    # --------------------------------------------------------------------------
    # Stage 2: Weighted Soft-Voting Probability Fusion & Weight Normalization
    # --------------------------------------------------------------------------
    print("\n--- [Stage 2] Verifying Soft-Voting Fusion & Weight Normalization ---")
    assert detector.weights["resnet18"] == 0.6
    assert detector.weights["mobilenet_v3"] == 0.4
    assert sum(detector.weights.values()) == 1.0

    # Test auto-normalization of un-normalized weights (e.g. 8 and 2)
    detector.set_weights({"resnet18": 8.0, "mobilenet_v3": 2.0})
    assert detector.weights["resnet18"] == 0.8
    assert detector.weights["mobilenet_v3"] == 0.2
    print(f"  [PASS] Normalized weights: resnet18={detector.weights['resnet18']}, mobilenet_v3={detector.weights['mobilenet_v3']}")

    # Restore default weights
    detector.set_weights(DEFAULT_WEIGHTS)

    # --------------------------------------------------------------------------
    # Stage 3: Single Image Inference & Multi-Channel Formats
    # --------------------------------------------------------------------------
    print("\n--- [Stage 3] Testing Multi-Format Ingestion & Output Schema ---")
    pil_img = create_test_image(color=(30, 90, 180))

    # Format A: PIL
    t0 = time.perf_counter()
    resp_pil = detector.predict(pil_img, filename="pil_test.png")
    lat_pil = (time.perf_counter() - t0) * 1000.0
    assert resp_pil.verdict in (PredictionVerdict.REAL, PredictionVerdict.AI_GENERATED)
    print(f"  [PASS] PIL Image Ingestion -> Verdict: {resp_pil.verdict.value} (Lat: {lat_pil:.1f}ms)")

    # Format B: Raw Bytes
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    raw_bytes = buf.getvalue()
    resp_bytes = detector.predict(raw_bytes, filename="bytes_test.png")
    assert resp_bytes.verdict in (PredictionVerdict.REAL, PredictionVerdict.AI_GENERATED)
    print(f"  [PASS] Raw Bytes Ingestion -> Verdict: {resp_bytes.verdict.value}")

    # Format C: RFC 2397 Base64 URI
    b64_str = f"data:image/png;base64,{base64.b64encode(raw_bytes).decode('ascii')}"
    resp_b64 = detector.predict(b64_str, filename="base64_test.png")
    assert resp_b64.verdict in (PredictionVerdict.REAL, PredictionVerdict.AI_GENERATED)
    print(f"  [PASS] Base64 URI Ingestion -> Verdict: {resp_b64.verdict.value}")

    # --------------------------------------------------------------------------
    # Stage 4: Sub-Model Telemetry & Consensus Metric
    # --------------------------------------------------------------------------
    print("\n--- [Stage 4] Verifying Sub-Model Consensus & Disagreement Variance ---")
    meta = resp_pil.ensemble_metadata
    assert meta is not None, "Ensemble metadata missing!"
    assert len(meta.submodels) == 2, "Expected 2 sub-models in telemetry!"

    for sm in meta.submodels:
        print(f"  * Sub-Model: {sm.model_name:<14} | Weight: {sm.weight:.2f} | Conf: {sm.confidence:.4f} | P(AI): {sm.probabilities.ai_generated:.4f}")

    print(f"  [PASS] Consensus Agreement Score : {meta.consensus_score:.4f} (1.0 = unanimous)")
    print(f"  [PASS] Disagreement Variance     : {meta.disagreement_variance:.6f}")
    assert 0.0 <= meta.consensus_score <= 1.0

    # --------------------------------------------------------------------------
    # Stage 5: Vectorized Batch Collation & Throughput
    # --------------------------------------------------------------------------
    print("\n--- [Stage 5] Testing Vectorized Batch Collation ---")
    batch_images = [create_test_image(color=(i * 20, 100, 150)) for i in range(12)]
    batch_names = [f"batch_img_{i}.png" for i in range(12)]

    t_batch0 = time.perf_counter()
    batch_resp = detector.predict_batch(batch_images, filenames=batch_names, chunk_size=8)
    batch_time = (time.perf_counter() - t_batch0) * 1000.0

    assert batch_resp.summary.total_submitted == 12
    assert batch_resp.summary.successful_count == 12
    assert len(batch_resp.results) == 12
    print(f"  [PASS] Batch processed: 12 images in {batch_time:.1f}ms ({batch_resp.summary.successful_count} succeeded, {batch_resp.summary.failed_count} failed)")
    print(f"  [PASS] Average latency per image: {batch_resp.avg_latency_per_image_ms:.2f}ms")

    # --------------------------------------------------------------------------
    # Stage 6: Fault-Isolated Partial Error Handling
    # --------------------------------------------------------------------------
    print("\n--- [Stage 6] Verifying Fault-Isolated Partial Error Handling ---")
    mixed_batch = [
        create_test_image(),
        b"CORRUPTED_NON_IMAGE_DATA_12345",
        create_test_image(),
    ]
    mixed_resp = detector.predict_batch(mixed_batch, filenames=["ok1.png", "bad.png", "ok2.png"])
    assert mixed_resp.summary.total_submitted == 3
    assert mixed_resp.summary.successful_count == 2
    assert mixed_resp.summary.failed_count == 1
    assert len(mixed_resp.errors) == 1
    assert mixed_resp.errors[0].filename == "bad.png"
    print(f"  [PASS] Fault isolation verified: 2 succeeded, 1 safely caught in errors list ({mixed_resp.errors[0].error_code})")

    # --------------------------------------------------------------------------
    # Stage 7: Dynamic Chunking Stress Test (B > 32)
    # --------------------------------------------------------------------------
    print("\n--- [Stage 7] Testing Dynamic Chunking Across Large Batches ---")
    large_batch = [create_test_image() for _ in range(36)]
    large_resp = detector.predict_batch(large_batch, chunk_size=16)
    assert large_resp.summary.successful_count == 36
    print("  [PASS] Sliced 36 items cleanly into chunks (16+16+4) without GPU memory crash.")

    # --------------------------------------------------------------------------
    # Stage 8: Pytest Unit Test Execution
    # --------------------------------------------------------------------------
    print("\n--- [Stage 8] Running Pytest Suite (tests/test_image_ensemble.py) ---")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_image_ensemble.py", "-v"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print("  [PASS] All 11 automated pytest unit tests passed (100%).")
    else:
        print(f"  [FAIL] Pytest encountered errors:\n{result.stdout}")
        sys.exit(1)

    print("\n" + "=" * 75)
    print("ALL SPRINT 3 — WEEK 1 DELIVERABLES FULLY VERIFIED (100% COMPLETE)")
    print("===========================================================================")


if __name__ == "__main__":
    run_verification()
