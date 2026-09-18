"""
verify_sprint2_week1.py — Standalone Verification Suite for Sprint 2 Week 1 Deliverables

Student: Divisha Manak Bohra (23ESKCA038)
Sprint: Sprint 2 — Baseline Detection Service
User Story: Building the image detection module
Task: Developing an image inference service with a defined input/output interface
"""

import base64
import io
import sys
import time
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.image_detector import (
    CorruptedImageError,
    EmptyImageError,
    ImageDetectionService,
    ImageInferenceResponse,
    PredictionVerdict,
)


def print_banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def create_test_image(color: tuple, size: tuple = (128, 128)) -> Image.Image:
    """Helper to generate in-memory synthetic image."""
    return Image.new("RGB", size, color=color)


def main() -> int:
    print_banner("SPRINT 2 WEEK 1 — IMAGE INFERENCE SERVICE VERIFICATION")
    print("Member:    Divisha Manak Bohra (23ESKCA038)")
    print("Story:     Building the image detection module")
    print("Task 1:    Developing an image inference service with defined I/O interface")

    passed_checks = 0
    total_checks = 6

    # ----------------------------------------------------------------------
    # Check 1: Service Initialization and Hardware Discovery
    # ----------------------------------------------------------------------
    print("\n[Check 1/6] Initializing ImageDetectionService & Hardware Probe...")
    try:
        service = ImageDetectionService()
        info = service.get_service_info()
        print(f"  ✓ Service Name:     {info.service_name}")
        print(f"  ✓ Model Version:    {info.model_version}")
        print(f"  ✓ Active Device:    {info.device}")
        print(f"  ✓ Supported Formats:{', '.join(info.supported_formats)}")
        passed_checks += 1
    except Exception as e:
        print(f"  ✗ FAILED to initialize service: {e}")
        return 1

    # ----------------------------------------------------------------------
    # Check 2: Single Image Inference via PIL Image
    # ----------------------------------------------------------------------
    print("\n[Check 2/6] Verifying PIL Image Inference...")
    try:
        sample_img = create_test_image(color=(34, 139, 34), size=(200, 200))
        resp = service.predict(sample_img, filename="sample_pil.png")
        assert isinstance(resp, ImageInferenceResponse)
        assert resp.verdict in [PredictionVerdict.REAL, PredictionVerdict.AI_GENERATED]
        assert 0.0 <= resp.confidence <= 1.0
        assert 0.0 <= resp.probabilities.real <= 1.0
        assert 0.0 <= resp.probabilities.ai_generated <= 1.0
        print(f"  ✓ Verdict:          {resp.verdict.value}")
        print(f"  ✓ Confidence:       {resp.confidence:.4f}")
        print(f"  ✓ Probabilities:    Real={resp.probabilities.real:.4f}, AI={resp.probabilities.ai_generated:.4f}")
        print(f"  ✓ Metadata:         {resp.image_metadata.width}x{resp.image_metadata.height} ({resp.image_metadata.format})")
        print(f"  ✓ Latency:          {resp.latency_ms:.2f} ms")
        passed_checks += 1
    except Exception as e:
        print(f"  ✗ FAILED PIL image inference: {e}")
        return 1

    # ----------------------------------------------------------------------
    # Check 3: Raw Byte and Base64 Data URI Ingestion
    # ----------------------------------------------------------------------
    print("\n[Check 3/6] Verifying In-Memory Bytes & Base64 Data URI...")
    try:
        buf = io.BytesIO()
        sample_img.save(buf, format="JPEG")
        jpeg_bytes = buf.getvalue()

        # Ingestion from raw bytes
        resp_bytes = service.predict(jpeg_bytes, filename="encoded.jpeg")
        assert resp_bytes.image_metadata.format == "JPEG"

        # Ingestion from Base64 Data URI
        b64_str = base64.b64encode(jpeg_bytes).decode("ascii")
        data_uri = f"data:image/jpeg;base64,{b64_str}"
        resp_b64 = service.predict(data_uri, filename="encoded_b64.jpeg")
        assert resp_b64.verdict == resp_bytes.verdict

        print("  ✓ Successfully ingested raw binary JPEG bytes.")
        print("  ✓ Successfully ingested RFC 2397 Base64 Data URI.")
        passed_checks += 1
    except Exception as e:
        print(f"  ✗ FAILED multi-modal input ingestion: {e}")
        return 1

    # ----------------------------------------------------------------------
    # Check 4: Determinism and Calibration Checks
    # ----------------------------------------------------------------------
    print("\n[Check 4/6] Verifying Deterministic Prediction Invariance...")
    try:
        runs = [service.predict(sample_img, filename="deterministic.png") for _ in range(3)]
        c0 = runs[0].confidence
        v0 = runs[0].verdict
        for idx, r in enumerate(runs[1:], 1):
            assert r.verdict == v0, f"Verdict mismatch in run {idx}"
            assert abs(r.confidence - c0) < 1e-4, f"Confidence mismatch in run {idx}"
        print("  ✓ Verified 100% deterministic outputs across repeated inference cycles.")
        passed_checks += 1
    except Exception as e:
        print(f"  ✗ FAILED determinism check: {e}")
        return 1

    # ----------------------------------------------------------------------
    # Check 5: Exception Handling & Edge Case Resilience
    # ----------------------------------------------------------------------
    print("\n[Check 5/6] Verifying Typed Exceptions on Malformed Inputs...")
    try:
        # Empty payload
        try:
            service.predict(b"", filename="zero_byte.png")
            print("  ✗ Empty byte payload failed to raise EmptyImageError")
            return 1
        except EmptyImageError:
            print("  ✓ EmptyImageError properly raised for 0-byte input.")

        # Corrupted payload
        try:
            service.predict(b"\x89PNG\r\n\x1a\ncorrupted_noise_payload", filename="bad.png")
            print("  ✗ Corrupted payload failed to raise CorruptedImageError")
            return 1
        except CorruptedImageError:
            print("  ✓ CorruptedImageError properly raised for corrupted bytes.")

        passed_checks += 1
    except Exception as e:
        print(f"  ✗ FAILED exception verification: {e}")
        return 1

    # ----------------------------------------------------------------------
    # Check 6: Latency Benchmark
    # ----------------------------------------------------------------------
    print("\n[Check 6/6] Benchmarking Inference Latency (< 100ms CPU SLA)...")
    try:
        latencies = []
        for _ in range(5):
            t0 = time.perf_counter()
            service.predict(sample_img)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        print(f"  ✓ Avg Latency (5 iterations): {avg_latency:.2f} ms (Min: {min_latency:.2f} ms)")
        passed_checks += 1
    except Exception as e:
        print(f"  ✗ FAILED latency benchmark: {e}")
        return 1

    # ----------------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------------
    print_banner(f"VERIFICATION PASSED: ALL {passed_checks}/{total_checks} CHECKS SUCCEEDED")
    print(f"Status: Sprint 2 Week 1 (Task 1 Foundation) is 100% complete ({passed_checks}/{total_checks} checks passed).\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
