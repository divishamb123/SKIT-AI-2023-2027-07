"""
test_image_interaction_components.py — Pytest Suite for Sprint 2 Week 5 Deliverables

Student: Divisha Manak Bohra (23ESKCA038)
Sprint: Sprint 2 — Baseline Detection Service
User Story: Building the image detection module
Task 4: Designing image upload and result interaction components (Week 5)
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
TYPES_PATH = FRONTEND_DIR / "src" / "types" / "image-detection.ts"
API_LIB_PATH = FRONTEND_DIR / "src" / "lib" / "api.ts"
DROPZONE_PATH = (
    FRONTEND_DIR / "src" / "components" / "image-detection" / "ImageUploadDropzone.tsx"
)
GAUGE_PATH = (
    FRONTEND_DIR / "src" / "components" / "image-detection" / "VerdictGauge.tsx"
)
METADATA_PATH = (
    FRONTEND_DIR / "src" / "components" / "image-detection" / "ImageMetadataCard.tsx"
)
GALLERY_PATH = (
    FRONTEND_DIR / "src" / "components" / "image-detection" / "BatchResultsGallery.tsx"
)
PAGE_PATH = FRONTEND_DIR / "src" / "app" / "image-detection" / "page.tsx"
NAVBAR_PATH = FRONTEND_DIR / "src" / "components" / "layout" / "Navbar.tsx"


def test_typescript_schemas_exist_and_complete():
    """Verifies that all required TypeScript schemas for image detection exist."""
    assert TYPES_PATH.is_file(), f"Missing {TYPES_PATH}"
    content = TYPES_PATH.read_text(encoding="utf-8")

    expected_types = [
        "ImageMetadata",
        "ImageProbabilities",
        "ImageDetectionResponse",
        "BatchImageItem",
        "BatchImageErrorItem",
        "BatchImageDetectionResponse",
        "SelectedImageFile",
    ]
    for type_name in expected_types:
        assert f"interface {type_name}" in content, f"Missing interface {type_name}"


def test_api_client_functions():
    """Verifies that api.ts exports detectImage, detectImageBase64, and detectImageBatch."""
    assert API_LIB_PATH.is_file(), f"Missing {API_LIB_PATH}"
    content = API_LIB_PATH.read_text(encoding="utf-8")

    assert "export async function detectImage(" in content
    assert "export async function detectImageBase64(" in content
    assert "export async function detectImageBatch(" in content
    assert "multipart/form-data" in content
    assert "/api/detect/image/batch" in content


def test_dropzone_component_validation_guardrails():
    """Verifies that ImageUploadDropzone.tsx enforces format, size, and batch ceilings."""
    assert DROPZONE_PATH.is_file(), f"Missing {DROPZONE_PATH}"
    content = DROPZONE_PATH.read_text(encoding="utf-8")

    assert "ALLOWED_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.webp']" in content
    assert "15 * 1024 * 1024" in content  # 15 MB limit
    assert "DEFAULT_MAX_FILES = 32" in content  # 32 batch limit
    assert "onSampleLoad" in content
    assert "onFilesSelected" in content


def test_verdict_gauge_visual_elements():
    """Verifies that VerdictGauge.tsx renders SVG gauge, colors, and percentage math."""
    assert GAUGE_PATH.is_file(), f"Missing {GAUGE_PATH}"
    content = GAUGE_PATH.read_text(encoding="utf-8")

    assert "strokeDasharray" in content
    assert "strokeDashoffset" in content
    assert "#f43f5e" in content  # Rose for AI
    assert "#10b981" in content  # Emerald for Real
    assert "AUTHENTIC PHOTOGRAPH" in content
    assert "AI-GENERATED MEDIA" in content


def test_metadata_card_attributes():
    """Verifies that ImageMetadataCard.tsx includes resolution, latency, and audit JSON."""
    assert METADATA_PATH.is_file(), f"Missing {METADATA_PATH}"
    content = METADATA_PATH.read_text(encoding="utf-8")

    assert "Resolution" in content
    assert "Color Channels" in content
    assert "Inference Latency" in content
    assert "JSON.stringify" in content
    assert "navigator.clipboard.writeText" in content


def test_batch_gallery_kpi_and_filtering():
    """Verifies that BatchResultsGallery.tsx renders summary KPIs and filter buttons."""
    assert GALLERY_PATH.is_file(), f"Missing {GALLERY_PATH}"
    content = GALLERY_PATH.read_text(encoding="utf-8")

    assert "throughput_fps" in content
    assert "handleExportJson" in content
    assert "AI-Generated" in content
    assert "Authentic Real" in content
    assert "Isolated Batch Errors" in content


def test_page_component_wiring():
    """Verifies that /image-detection/page.tsx coordinates components and modes."""
    assert PAGE_PATH.is_file(), f"Missing {PAGE_PATH}"
    content = PAGE_PATH.read_text(encoding="utf-8")

    assert "Divisha Manak Bohra" in content
    assert "Single Image Analysis" in content
    assert "Batch Processing (B ≤ 32)" in content
    assert "VerdictGauge" in content
    assert "BatchResultsGallery" in content
    assert "detectImage" in content
    assert "detectImageBatch" in content


def test_navigation_bar_links():
    """Verifies that Navbar includes /image-detection in the primary menu."""
    assert NAVBAR_PATH.is_file(), f"Missing {NAVBAR_PATH}"
    content = NAVBAR_PATH.read_text(encoding="utf-8")

    assert "{ name: 'Image Detection', href: '/image-detection' }" in content
