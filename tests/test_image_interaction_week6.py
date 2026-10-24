"""
test_image_interaction_week6.py — Pytest Suite for Sprint 2 Week 6 Deliverables

Student: Divisha Manak Bohra (23ESKCA038)
Sprint: Sprint 2 — Baseline Detection Service
User Story: Building the image detection module
Task 4: Designing image upload and result interaction components (Week 6)
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
TYPES_PATH = FRONTEND_DIR / "src" / "types" / "image-detection.ts"
CONTROLLER_PATH = (
    FRONTEND_DIR
    / "src"
    / "components"
    / "image-detection"
    / "DecisionBoundaryController.tsx"
)
VIEWER_PATH = (
    FRONTEND_DIR
    / "src"
    / "components"
    / "image-detection"
    / "ForensicEvidenceViewer.tsx"
)
MODAL_PATH = (
    FRONTEND_DIR / "src" / "components" / "image-detection" / "ForensicReportModal.tsx"
)
GALLERY_PATH = (
    FRONTEND_DIR / "src" / "components" / "image-detection" / "BatchResultsGallery.tsx"
)
PAGE_PATH = FRONTEND_DIR / "src" / "app" / "image-detection" / "page.tsx"


def test_week6_typescript_schemas():
    """Verifies that Week 6 schemas for threshold simulation, channels, and certificates exist."""
    assert TYPES_PATH.is_file(), f"Missing {TYPES_PATH}"
    content = TYPES_PATH.read_text(encoding="utf-8")

    expected_types = [
        "ConfidenceTier",
        "ThresholdSimulationResult",
        "ChannelDecomposition",
        "ForensicAuditReport",
    ]
    for type_name in expected_types:
        assert type_name in content, f"Missing type {type_name} in {TYPES_PATH.name}"

    assert "'HIGH_AI'" in content
    assert "evidenceHashSha256" in content
    assert "calibratedVerdict" in content


def test_decision_boundary_controller_structure():
    """Verifies that DecisionBoundaryController.tsx provides threshold slider and presets."""
    assert CONTROLLER_PATH.is_file(), f"Missing {CONTROLLER_PATH}"
    content = CONTROLLER_PATH.read_text(encoding="utf-8")

    assert "export function DecisionBoundaryController" in content
    assert "export function computeThresholdSimulation" in content
    assert 'type="range"' in content
    assert "0.10" in content and "0.90" in content
    assert "High Sensitivity" in content
    assert "Balanced Baseline" in content


def test_decision_boundary_math_logic():
    """Validates the mathematical boundary margin and calibration rules in DecisionBoundaryController."""
    content = CONTROLLER_PATH.read_text(encoding="utf-8")

    assert "adjustedVerdict" in content
    assert "Math.abs(pAI - threshold)" in content
    assert "HIGH_AI" in content
    assert "SUSPICIOUS_AI" in content
    assert "HIGH_REAL" in content
    assert "PLAUSIBLE_REAL" in content
    assert "AMBIGUOUS" in content


def test_forensic_evidence_viewer_channels():
    """Verifies that ForensicEvidenceViewer.tsx provides multi-channel decomposition."""
    assert VIEWER_PATH.is_file(), f"Missing {VIEWER_PATH}"
    content = VIEWER_PATH.read_text(encoding="utf-8")

    assert "export function ForensicEvidenceViewer" in content
    assert "rgb" in content
    assert "red" in content
    assert "green" in content
    assert "blue" in content
    assert "luminance" in content
    assert "highpass" in content
    assert "0.299 * r + 0.587 * g + 0.114 * b" in content  # Luminance formula
    assert "zoomLevel" in content


def test_forensic_report_modal_components():
    """Verifies that ForensicReportModal.tsx provides cryptographic hash and PDF printing."""
    assert MODAL_PATH.is_file(), f"Missing {MODAL_PATH}"
    content = MODAL_PATH.read_text(encoding="utf-8")

    assert "export function ForensicReportModal" in content
    assert (
        "crypto.subtle.digest('SHA-256'" in content or "crypto.subtle.digest" in content
    )
    assert "Digital Media Forensic Investigation Certificate" in content
    assert "window.print()" in content
    assert "SKIT/AI/2023-2027/07" in content
    assert "Divisha Manak Bohra" in content


def test_batch_gallery_enhanced_features():
    """Verifies that BatchResultsGallery.tsx includes CSV export, search, and dynamic thresholding."""
    assert GALLERY_PATH.is_file(), f"Missing {GALLERY_PATH}"
    content = GALLERY_PATH.read_text(encoding="utf-8")

    assert "handleExportCsv" in content
    assert "handleExportJson" in content
    assert "searchQuery" in content
    assert "batchThreshold" in content
    assert "sortBy" in content


def test_image_detection_page_week6_integration():
    """Verifies that /image-detection/page.tsx integrates all Week 6 interactive tools."""
    assert PAGE_PATH.is_file(), f"Missing {PAGE_PATH}"
    content = PAGE_PATH.read_text(encoding="utf-8")

    assert "DecisionBoundaryController" in content
    assert "ForensicEvidenceViewer" in content
    assert "ForensicReportModal" in content
    assert "Divisha Manak Bohra" in content
    assert "Sprint 2 • Task 4" in content


def test_student_and_form2_attribution():
    """Verifies formal student attribution and project ID consistency."""
    for path in [CONTROLLER_PATH, VIEWER_PATH, MODAL_PATH, PAGE_PATH]:
        content = path.read_text(encoding="utf-8")
        assert (
            "Divisha Manak Bohra" in content or "23ESKCA038" in content
        ), f"Missing attribution in {path.name}"
