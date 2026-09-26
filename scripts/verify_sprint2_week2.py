"""
verify_sprint2_week2.py — Standalone Verification Suite for Sprint 2 Week 2 Deliverables

Student: Divisha Manak Bohra (23ESKCA038)
Sprint: Sprint 2 — Baseline Detection Service
User Story: Building the image detection module
Task: I/O API Specification: Define request/response schemas, implement /api/detect/image endpoint with strict validation
"""

import base64
import io
import sys
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.image_detector.api import create_app

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
        "SPRINT 2 WEEK 2 — I/O API SPECIFICATION & STRICT VALIDATION VERIFICATION"
    )
    print("Member:    Divisha Manak Bohra (23ESKCA038)")
    print("Story:     Building the image detection module")
    print(
        "Task 1:    I/O API Specification & /api/detect/image Endpoint Implementation"
    )

    passed_checks = 0
    total_checks = 8

    # ----------------------------------------------------------------------
    # Check 1: OpenAPI Schema Generation & Endpoints Discovery
    # ----------------------------------------------------------------------
    print("\n[Check 1/8] Verifying OpenAPI Schema and Route Discovery...")
    try:
        resp = client.get("/openapi.json")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        schema = resp.json()
        paths = schema.get("paths", {})
        assert "/api/detect/image" in paths, "Missing /api/detect/image route"
        assert (
            "/api/detect/image/base64" in paths
        ), "Missing /api/detect/image/base64 route"
        assert (
            "/api/detect/image/health" in paths
        ), "Missing /api/detect/image/health route"
        print(
            f"  ✓ OpenAPI Title:      {schema['info']['title']} v{schema['info']['version']}"
        )
        print(f"  ✓ Total Routes:       {len(paths)} documented routes found.")
        passed_checks += 1
    except Exception as e:
        print(f"  ✗ FAILED OpenAPI check: {e}")
        return 1

    # ----------------------------------------------------------------------
    # Check 2: Healthcheck Telemetry
    # ----------------------------------------------------------------------
    print("\n[Check 2/8] Testing /api/detect/image/health Diagnostic...")
    try:
        resp = client.get("/api/detect/image/health")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        data = resp.json()
        assert data["status"] == "healthy"
        print(f"  ✓ Service Status:     {data['status']}")
        print(f"  ✓ Active Device:      {data['device']}")
        print(f"  ✓ Model Version:      {data['model_version']}")
        passed_checks += 1
    except Exception as e:
        print(f"  ✗ FAILED health endpoint check: {e}")
        return 1

    # ----------------------------------------------------------------------
    # Check 3: Multipart Form-Data Detection (PNG)
    # ----------------------------------------------------------------------
    print("\n[Check 3/8] Testing Multipart Image Detection (PNG)...")
    try:
        png_bytes = create_test_image(color=(30, 144, 255), size=(128, 128), fmt="PNG")
        resp = client.post(
            "/api/detect/image",
            files={"file": ("test_sample.png", png_bytes, "image/png")},
        )
        assert (
            resp.status_code == 200
        ), f"Expected 200, got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert data["filename"] == "test_sample.png"
        assert data["verdict"] in ["REAL", "AI_GENERATED"]
        assert 0.0 <= data["confidence"] <= 1.0
        assert "request_id" in data
        assert "timestamp" in data
        print(f"  ✓ Verdict:            {data['verdict']}")
        print(f"  ✓ Confidence:         {data['confidence']:.4f}")
        print(f"  ✓ Request ID:         {data['request_id']}")
        print(f"  ✓ Latency:            {data['latency_ms']:.2f} ms")
        passed_checks += 1
    except Exception as e:
        print(f"  ✗ FAILED PNG multipart detection: {e}")
        return 1

    # ----------------------------------------------------------------------
    # Check 4: Multi-Format Ingestion (JPEG & WEBP)
    # ----------------------------------------------------------------------
    print("\n[Check 4/8] Testing Multi-Format Ingestion (JPEG & WEBP)...")
    try:
        jpeg_bytes = create_test_image(color=(255, 69, 0), size=(100, 100), fmt="JPEG")
        resp_jpeg = client.post(
            "/api/detect/image",
            files={"file": ("photo.jpeg", jpeg_bytes, "image/jpeg")},
        )
        assert resp_jpeg.status_code == 200
        assert resp_jpeg.json()["image_metadata"]["format"] == "JPEG"

        webp_bytes = create_test_image(color=(50, 205, 50), size=(100, 100), fmt="WEBP")
        resp_webp = client.post(
            "/api/detect/image",
            files={"file": ("graphic.webp", webp_bytes, "image/webp")},
        )
        assert resp_webp.status_code == 200
        assert resp_webp.json()["image_metadata"]["format"] == "WEBP"

        print("  ✓ Successfully processed image/jpeg upload.")
        print("  ✓ Successfully processed image/webp upload.")
        passed_checks += 1
    except Exception as e:
        print(f"  ✗ FAILED multi-format check: {e}")
        return 1

    # ----------------------------------------------------------------------
    # Check 5: Base64 JSON Payload Detection
    # ----------------------------------------------------------------------
    print(
        "\n[Check 5/8] Testing Base64 Payload Ingestion (/api/detect/image/base64)..."
    )
    try:
        b64_raw = base64.b64encode(png_bytes).decode("ascii")
        data_uri = f"data:image/png;base64,{b64_raw}"
        resp = client.post(
            "/api/detect/image/base64",
            json={"image_data": data_uri, "filename": "base64_upload.png"},
        )
        assert (
            resp.status_code == 200
        ), f"Expected 200, got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert data["filename"] == "base64_upload.png"
        assert data["status"] == "success"
        print(f"  ✓ Base64 Verdict:     {data['verdict']} ({data['confidence']:.4f})")
        passed_checks += 1
    except Exception as e:
        print(f"  ✗ FAILED Base64 detection check: {e}")
        return 1

    # ----------------------------------------------------------------------
    # Check 6: Strict Rejection of 0-Byte Payload (HTTP 400)
    # ----------------------------------------------------------------------
    print("\n[Check 6/8] Testing Empty Payload Boundary (HTTP 400 Bad Request)...")
    try:
        resp = client.post(
            "/api/detect/image",
            files={"file": ("zero_byte.png", b"", "image/png")},
        )
        assert resp.status_code == 400, f"Expected 400, got {resp.status_code}"
        detail = resp.json()["detail"]
        assert detail["error_code"] == "EMPTY_PAYLOAD"
        print(f"  ✓ Caught HTTP 400:     {detail['message']}")
        passed_checks += 1
    except Exception as e:
        print(f"  ✗ FAILED empty payload check: {e}")
        return 1

    # ----------------------------------------------------------------------
    # Check 7: Strict Rejection of Unsupported Media & Spoofed Headers (HTTP 415)
    # ----------------------------------------------------------------------
    print("\n[Check 7/8] Testing Media Type & Magic Byte Guardrails (HTTP 415)...")
    try:
        # Unsupported extension
        resp_ext = client.post(
            "/api/detect/image",
            files={
                "file": (
                    "malicious.exe",
                    b"MZ\x90\x00fake_executable",
                    "application/octet-stream",
                )
            },
        )
        assert resp_ext.status_code == 415, f"Expected 415, got {resp_ext.status_code}"
        assert resp_ext.json()["detail"]["error_code"] == "UNSUPPORTED_EXTENSION"

        # Spoofed magic bytes (.png filename with text content)
        resp_spoof = client.post(
            "/api/detect/image",
            files={"file": ("spoofed.png", b"Not a PNG image header", "image/png")},
        )
        assert (
            resp_spoof.status_code == 415
        ), f"Expected 415, got {resp_spoof.status_code}"
        assert resp_spoof.json()["detail"]["error_code"] == "INVALID_MAGIC_BYTES"

        print("  ✓ Properly rejected unsupported file extension with HTTP 415.")
        print("  ✓ Properly caught spoofed extension via magic bytes with HTTP 415.")
        passed_checks += 1
    except Exception as e:
        print(f"  ✗ FAILED media type verification: {e}")
        return 1

    # ----------------------------------------------------------------------
    # Check 8: Corrupted Image & Dimension Boundary Guardrails (HTTP 422)
    # ----------------------------------------------------------------------
    print("\n[Check 8/8] Testing Corrupted Stream & Dimension Guardrails (HTTP 422)...")
    try:
        # Corrupted stream
        corrupt_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDRcorrupted_payload_data"
        resp_corrupt = client.post(
            "/api/detect/image",
            files={"file": ("broken.png", corrupt_bytes, "image/png")},
        )
        assert (
            resp_corrupt.status_code == 422
        ), f"Expected 422, got {resp_corrupt.status_code}"
        assert resp_corrupt.json()["detail"]["error_code"] == "CORRUPTED_IMAGE"

        # Resolution too small (< 16x16)
        tiny_bytes = create_test_image(color=(255, 255, 0), size=(10, 10), fmt="PNG")
        resp_tiny = client.post(
            "/api/detect/image",
            files={"file": ("tiny.png", tiny_bytes, "image/png")},
        )
        assert (
            resp_tiny.status_code == 422
        ), f"Expected 422, got {resp_tiny.status_code}"
        assert resp_tiny.json()["detail"]["error_code"] == "DIMENSION_TOO_SMALL"

        print("  ✓ Successfully rejected corrupted image stream with HTTP 422.")
        print("  ✓ Successfully rejected sub-minimum resolution with HTTP 422.")
        passed_checks += 1
    except Exception as e:
        print(f"  ✗ FAILED corruption/dimension verification: {e}")
        return 1

    # ----------------------------------------------------------------------
    # Final Summary
    # ----------------------------------------------------------------------
    print_banner(
        f"VERIFICATION PASSED: ALL {passed_checks}/{total_checks} CHECKS SUCCEEDED"
    )
    print(
        f"Status: Sprint 2 Week 2 (I/O API Specification) is 100% complete ({passed_checks}/{total_checks} checks passed).\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
