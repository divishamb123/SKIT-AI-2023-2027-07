"""
verify_sprint2_week3.py — Standalone Verification Suite for Sprint 2 Week 3 Deliverables

Student: Divisha Manak Bohra (23ESKCA038)
Sprint: Sprint 2 — Baseline Detection Service
User Story: Building the image detection module
Task 2: Designing batch-processing logic for multiple image inputs (Week 3)
"""

import base64
import io
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.image_detector.api import create_app
from src.image_detector.service import ImageDetectionService

app = create_app()
client = TestClient(app)


def print_banner(title: str) -> None:
    print("\n" + "=" * 75)
    print(f"  {title}")
    print("=" * 75)


def create_test_image(color: tuple, size: tuple = (64, 64), fmt: str = "PNG") -> bytes:
    """Helper to generate in-memory synthetic image bytes."""
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=color)
    img.save(buf, format=fmt)
    return buf.getvalue()


def main() -> int:
    print_banner(
        "SPRINT 2 WEEK 3 — HIGH-THROUGHPUT BATCH-PROCESSING LOGIC VERIFICATION"
    )
    print("Member:    Divisha Manak Bohra (23ESKCA038)")
    print("Story:     Building the image detection module")
    print(
        "Task 2:    Designing batch-processing logic for multiple image inputs (Week 3)"
    )

    passed_checks = 0
    total_checks = 8

    # ----------------------------------------------------------------------
    # Check 1: OpenAPI Schema & Batch Routes Discovery
    # ----------------------------------------------------------------------
    print("\n[Check 1/8] Verifying OpenAPI Schema and Batch Routes Discovery...")
    try:
        resp = client.get("/openapi.json")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        schema = resp.json()
        paths = schema.get("paths", {})
        assert (
            "/api/detect/image/batch" in paths
        ), "Missing /api/detect/image/batch route"
        assert (
            "/api/detect/image/batch/base64" in paths
        ), "Missing /api/detect/image/batch/base64 route"
        print("  ✓ Documented route found: POST /api/detect/image/batch")
        print("  ✓ Documented route found: POST /api/detect/image/batch/base64")
        passed_checks += 1
    except Exception as e:
        print(f"  ✗ FAILED OpenAPI check: {e}")
        return 1

    # ----------------------------------------------------------------------
    # Check 2: Core Batch Ingestion & Tensor Collation
    # ----------------------------------------------------------------------
    print(
        "\n[Check 2/8] Testing Core Batch Ingestion & Tensor Collation (PNG, JPEG, WEBP)..."
    )
    try:
        service = ImageDetectionService(device="cpu")
        images = [
            ("img_red.png", create_test_image((255, 0, 0), fmt="PNG")),
            ("img_green.jpg", create_test_image((0, 255, 0), fmt="JPEG")),
            ("img_blue.webp", create_test_image((0, 0, 255), fmt="WEBP")),
            ("img_yellow.png", create_test_image((255, 255, 0), fmt="PNG")),
            ("img_cyan.jpg", create_test_image((0, 255, 255), fmt="JPEG")),
            ("img_magenta.webp", create_test_image((255, 0, 255), fmt="WEBP")),
        ]

        batch_resp = service.predict_batch(images, chunk_size=4)
        assert batch_resp.summary.total_submitted == 6, "Total submitted mismatch"
        assert batch_resp.summary.successful_count == 6, "Successful count mismatch"
        assert batch_resp.summary.failed_count == 0, "Failed count mismatch"
        assert len(batch_resp.results) == 6, "Results count mismatch"
        print(f"  ✓ Collation Device:   {batch_resp.device}")
        print(f"  ✓ Total Submitted:    {batch_resp.summary.total_submitted}")
        print(f"  ✓ Successful Count:   {batch_resp.summary.successful_count}")
        print(f"  ✓ Mean Confidence:    {batch_resp.summary.mean_confidence:.4f}")
        passed_checks += 1
    except Exception as e:
        print(f"  ✗ FAILED Batch Ingestion & Collation check: {e}")
        return 1

    # ----------------------------------------------------------------------
    # Check 3: Dynamic Chunking (B <= 32) Partitioning Logic
    # ----------------------------------------------------------------------
    print("\n[Check 3/8] Testing Dynamic Chunking (B <= 32) Partitioning Logic...")
    try:
        sample_img = create_test_image((120, 120, 120))
        # 9 images with chunk_size=4 -> chunks: 4 + 4 + 1
        nine_images = [sample_img] * 9
        resp_chunked = service.predict_batch(nine_images, chunk_size=4)
        assert resp_chunked.summary.successful_count == 9, "Expected 9 successful"
        assert resp_chunked.chunk_size_used == 4, "Expected chunk_size_used == 4"
        print(
            "  ✓ Successfully verified chunk partitioning (4 + 4 + 1 items) without drops."
        )
        passed_checks += 1
    except Exception as e:
        print(f"  ✗ FAILED Dynamic Chunking check: {e}")
        return 1

    # ----------------------------------------------------------------------
    # Check 4: Vectorized Acceleration & Throughput Benchmarking
    # ----------------------------------------------------------------------
    print("\n[Check 4/8] Benchmarking Vectorized Batch Acceleration vs Sequential...")
    try:
        benchmark_images = [create_test_image((i * 20, 80, 160)) for i in range(8)]
        # Measure batch time
        t0 = time.perf_counter()
        batch_out = service.predict_batch(benchmark_images, chunk_size=8)
        batch_duration_ms = (time.perf_counter() - t0) * 1000.0
        assert batch_out.summary.successful_count == 8

        # Measure sequential time
        t0 = time.perf_counter()
        for img in benchmark_images:
            service.predict(img)
        seq_duration_ms = (time.perf_counter() - t0) * 1000.0

        print(
            f"  ✓ Sequential 8-image duration: {seq_duration_ms:.2f} ms ({seq_duration_ms / 8:.2f} ms/img)"
        )
        print(
            f"  ✓ Vectorized 8-image batch:    {batch_duration_ms:.2f} ms ({batch_duration_ms / 8:.2f} ms/img)"
        )
        print(
            f"  ✓ Acceleration factor:         {seq_duration_ms / max(batch_duration_ms, 0.001):.2f}x throughput"
        )
        passed_checks += 1
    except Exception as e:
        print(f"  ✗ FAILED Throughput Benchmark check: {e}")
        return 1

    # ----------------------------------------------------------------------
    # Check 5: Multipart Multi-File API Batch Endpoint
    # ----------------------------------------------------------------------
    print(
        "\n[Check 5/8] Testing Multipart Multi-File API Batch Endpoint (/api/detect/image/batch)..."
    )
    try:
        f1 = create_test_image((200, 50, 50), fmt="PNG")
        f2 = create_test_image((50, 200, 50), fmt="JPEG")
        f3 = create_test_image((50, 50, 200), fmt="WEBP")

        files = [
            ("files", ("upload_1.png", f1, "image/png")),
            ("files", ("upload_2.jpg", f2, "image/jpeg")),
            ("files", ("upload_3.webp", f3, "image/webp")),
        ]

        resp = client.post("/api/detect/image/batch", files=files)
        assert (
            resp.status_code == 200
        ), f"Expected 200, got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert "batch_id" in data, "Missing batch_id in response"
        assert data["summary"]["successful_count"] == 3
        print(f"  ✓ Received HTTP 200 with Batch ID: {data['batch_id']}")
        print(f"  ✓ Successfully verified {len(data['results'])} result objects.")
        passed_checks += 1
    except Exception as e:
        print(f"  ✗ FAILED Multipart API Batch check: {e}")
        return 1

    # ----------------------------------------------------------------------
    # Check 6: Base64 JSON API Batch Endpoint
    # ----------------------------------------------------------------------
    print(
        "\n[Check 6/8] Testing Base64 JSON API Batch Endpoint (/api/detect/image/batch/base64)..."
    )
    try:
        b64_1 = base64.b64encode(create_test_image((10, 20, 30))).decode("utf-8")
        b64_2 = "data:image/png;base64," + base64.b64encode(
            create_test_image((40, 50, 60))
        ).decode("utf-8")

        payload = {
            "images": [
                {"image_data": b64_1, "filename": "b64_1.png"},
                {"image_data": b64_2, "filename": "b64_2.png"},
            ]
        }

        resp = client.post("/api/detect/image/batch/base64", json=payload)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        data = resp.json()
        assert data["summary"]["successful_count"] == 2
        print("  ✓ Base64 batch successfully parsed and evaluated.")
        passed_checks += 1
    except Exception as e:
        print(f"  ✗ FAILED Base64 API Batch check: {e}")
        return 1

    # ----------------------------------------------------------------------
    # Check 7: Fault-Isolated Partial Failure Resilience
    # ----------------------------------------------------------------------
    print(
        "\n[Check 7/8] Testing Fault-Isolated Partial Failure Resilience (Mixed Valid + Corrupted)..."
    )
    try:
        valid_img = create_test_image((77, 88, 99))
        corrupt_img = b"random_corrupted_payload"
        empty_img = b""

        mixed_inputs = [
            ("valid_alpha.png", valid_img),
            ("corrupt_beta.png", corrupt_img),
            ("valid_gamma.png", valid_img),
            ("empty_delta.png", empty_img),
        ]

        batch_mixed = service.predict_batch(mixed_inputs, return_errors=True)
        assert batch_mixed.summary.total_submitted == 4
        assert batch_mixed.summary.successful_count == 2
        assert batch_mixed.summary.failed_count == 2
        assert len(batch_mixed.errors) == 2
        err_dict = {e.filename: e.error_code for e in batch_mixed.errors}
        assert err_dict["corrupt_beta.png"] == "CORRUPTED_IMAGE"
        assert err_dict["empty_delta.png"] == "EMPTY_PAYLOAD"
        print(
            "  ✓ Successfully completed valid items while isolating bad items into errors list."
        )
        print(
            f"  ✓ Errors isolated: {list(err_dict.keys())} -> {list(err_dict.values())}"
        )
        passed_checks += 1
    except Exception as e:
        print(f"  ✗ FAILED Fault Isolation check: {e}")
        return 1

    # ----------------------------------------------------------------------
    # Check 8: Boundary Limits Enforcement (Empty & Max Limit)
    # ----------------------------------------------------------------------
    print("\n[Check 8/8] Testing Boundary Limits Enforcement (Empty & >32 Limit)...")
    try:
        # Test > 32 images limit
        oversized = [
            ("files", (f"file_{i}.png", create_test_image((1, 2, 3)), "image/png"))
            for i in range(33)
        ]
        resp_over = client.post("/api/detect/image/batch", files=oversized)
        assert (
            resp_over.status_code == 400
        ), f"Expected 400, got {resp_over.status_code}"
        assert "BATCH_TOO_LARGE" in resp_over.text

        # Test empty base64 list
        resp_empty = client.post("/api/detect/image/batch/base64", json={"images": []})
        assert (
            resp_empty.status_code == 400
        ), f"Expected 400, got {resp_empty.status_code}"
        assert "EMPTY_BATCH" in resp_empty.text

        print("  ✓ Properly rejected >32 batch size with HTTP 400 (BATCH_TOO_LARGE).")
        print("  ✓ Properly rejected empty batch with HTTP 400 (EMPTY_BATCH).")
        passed_checks += 1
    except Exception as e:
        print(f"  ✗ FAILED Boundary Limits check: {e}")
        return 1

    # Summary
    print_banner(
        f"VERIFICATION PASSED: ALL {passed_checks}/{total_checks} CHECKS SUCCEEDED"
    )
    print(
        "Status: Sprint 2 Week 3 (Batch-Processing Logic) is 100% complete (8/8 checks passed).\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
