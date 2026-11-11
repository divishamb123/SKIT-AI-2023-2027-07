#!/usr/bin/env python3
"""
Sprint 3 Week 1 — Audio Threshold Calibration Verification
===========================================================
Dev Khandelwal (23ESKCA035) — Sprint 3 audio accuracy optimisation

Deliverable:
  Audio decision threshold calibration engine implementing 3 operating
  profiles (balanced, high_security, forensic) with temperature-scaled
  sigmoid probability calibration.

  - Balanced   τ = 1.7633  EER operating point from ASVspoof 2019 LA eval
  - High-sec   τ = 2.4     Reduced false acceptance
  - Forensic   τ = 0.8     Reduced missed detections

Run from repository root:
    python scripts/verify_sprint3_week1_audio.py

Exit 0 = all checks pass. Exit 1 = any check fails.
"""

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_SERVICE_PATH = os.path.join(REPO_ROOT, "services", "audio-service")
sys.path.insert(0, AUDIO_SERVICE_PATH)

PASS = "PASS"
FAIL = "FAIL"
results = []


def check(label, status, detail=""):
    results.append((label, status, detail))
    icon = "[OK]" if status == PASS else "[FAIL]"
    print(f"  {icon} [{status}] {label}")
    if detail:
        for line in detail.strip().splitlines():
            print(f"       {line}")


# =====================================================================
# Check 1: Module exists on disk
# =====================================================================
print("\n" + "=" * 70)
print("Check 1 - threshold_calibrator.py present on disk")
print("=" * 70)

calibrator_path = os.path.join(
    AUDIO_SERVICE_PATH, "app", "services", "threshold_calibrator.py"
)

if not os.path.isfile(calibrator_path):
    check("threshold_calibrator.py exists", FAIL, f"Not found: {calibrator_path}")
    sys.exit(1)
check("threshold_calibrator.py exists", PASS, calibrator_path)

# =====================================================================
# Check 2: Module imports cleanly
# =====================================================================
print("\n" + "=" * 70)
print("Check 2 - Module imports and public API")
print("=" * 70)

try:
    from app.services.threshold_calibrator import (
        CalibrationResult,  # noqa: F401 — imported to verify it is importable
        ConfidenceTier,
        ThresholdCalibrator,
        ThresholdMode,
        Verdict,
    )

    check("ThresholdCalibrator class importable", PASS)
    check("ThresholdMode enum importable", PASS)
    check("Verdict enum importable", PASS)
    check("ConfidenceTier enum importable", PASS)
    check("CalibrationResult dataclass importable", PASS)
except ImportError as exc:
    check("Module import", FAIL, str(exc))
    sys.exit(1)

# =====================================================================
# Check 3: EER threshold matches actual evaluation results
# =====================================================================
print("\n" + "=" * 70)
print("Check 3 - EER threshold matches asvspoof_baseline_metrics.json")
print("=" * 70)

metrics_path = os.path.join(REPO_ROOT, "results", "asvspoof_baseline_metrics.json")

if not os.path.isfile(metrics_path):
    check(
        "asvspoof_baseline_metrics.json exists",
        FAIL,
        f"Not found: {metrics_path}",
    )
else:
    check("asvspoof_baseline_metrics.json exists", PASS)
    try:
        with open(metrics_path, encoding="utf-8") as fh:
            metrics = json.load(fh)

        actual_eer = metrics.get("eer_percent")
        actual_threshold = metrics.get("eer_threshold")

        cal = ThresholdCalibrator(ThresholdMode.BALANCED)

        if (
            actual_threshold is not None
            and abs(cal.active_threshold - actual_threshold) < 1e-3
        ):
            check(
                f"Balanced threshold = {cal.active_threshold} matches JSON ({actual_threshold})",
                PASS,
            )
        else:
            check(
                "Balanced threshold matches JSON",
                FAIL,
                f"Calibrator: {cal.active_threshold}, JSON: {actual_threshold}",
            )

        if actual_eer is not None and abs(actual_eer - 0.992) < 0.01:
            check(
                f"EER = {actual_eer:.3f}% (target: <= 1.0%)",
                PASS,
            )
        else:
            check("EER value in expected range", FAIL, f"Got: {actual_eer}")

    except Exception as exc:
        check("Read baseline metrics JSON", FAIL, str(exc))

# =====================================================================
# Check 4: All 3 operating modes produce correct verdicts
# =====================================================================
print("\n" + "=" * 70)
print("Check 4 - All 3 threshold modes produce correct verdicts")
print("=" * 70)

for mode in ThresholdMode:
    try:
        cal = ThresholdCalibrator(mode)
        r_high = cal.calibrate(10.0)
        r_low = cal.calibrate(-10.0)

        assert r_high.verdict == Verdict.BONAFIDE, f"Score 10.0 not BONAFIDE in {mode}"
        assert r_low.verdict == Verdict.SPOOF, f"Score -10.0 not SPOOF in {mode}"
        assert 0.0 <= r_high.calibrated_probability <= 1.0
        assert 0.0 <= r_low.calibrated_probability <= 1.0

        check(
            f"Mode '{mode.value}' (threshold={cal.active_threshold:.4f})",
            PASS,
            f"BONAFIDE@+10: P={r_high.calibrated_probability:.4f} [{r_high.confidence_tier.value}]  "
            f"SPOOF@-10:  P={r_low.calibrated_probability:.4f} [{r_low.confidence_tier.value}]",
        )
    except Exception as exc:
        check(f"Mode '{mode.value}'", FAIL, str(exc))

# =====================================================================
# Check 5: Temperature-scaled sigmoid math
# =====================================================================
print("\n" + "=" * 70)
print("Check 5 - Temperature-scaled sigmoid calibration math")
print("=" * 70)

try:
    cal = ThresholdCalibrator(ThresholdMode.BALANCED)

    # At the threshold, sigmoid(0) = 0.5 exactly.
    at_boundary = cal.calibrate(cal.active_threshold)
    if abs(at_boundary.calibrated_probability - 0.5) < 1e-5:
        check(
            "sigmoid((tau - tau) / T) = 0.5 at decision boundary",
            PASS,
            f"P = {at_boundary.calibrated_probability:.8f}",
        )
    else:
        check(
            "sigmoid at boundary = 0.5",
            FAIL,
            f"P = {at_boundary.calibrated_probability}",
        )

    # Monotonicity over a range of scores.
    test_scores = [-5.0, -2.0, 0.0, 1.0, 1.7633, 2.5, 5.0]
    probs = [cal.calibrate(s).calibrated_probability for s in test_scores]
    monotone = all(probs[i] < probs[i + 1] for i in range(len(probs) - 1))
    if monotone:
        check(
            "P(bonafide) monotonically increasing across 7 score points",
            PASS,
            " ".join(f"{p:.4f}" for p in probs),
        )
    else:
        check(
            "P(bonafide) monotonically increasing",
            FAIL,
            str(list(zip(test_scores, probs))),
        )

    # Probability bounds for extreme inputs.
    edge_scores = [-1000.0, -100.0, 0.0, cal.active_threshold, 100.0, 1000.0]
    bounded = all(
        0.0 <= cal.calibrate(s).calibrated_probability <= 1.0 for s in edge_scores
    )
    if bounded:
        check(
            "P(bonafide) in [0.0, 1.0] for extreme inputs (+-1000)",
            PASS,
        )
    else:
        check("P(bonafide) bounds check", FAIL, "Out-of-bounds probability detected")

except Exception as exc:
    check("Calibration math verification", FAIL, str(exc))

# =====================================================================
# Check 6: Confidence tier classification
# =====================================================================
print("\n" + "=" * 70)
print("Check 6 - Confidence tier classification")
print("=" * 70)

try:
    cal = ThresholdCalibrator(ThresholdMode.BALANCED)

    r_extreme = cal.calibrate(10.0)
    r_boundary = cal.calibrate(cal.active_threshold)

    if r_extreme.confidence_tier == ConfidenceTier.HIGH:
        check(
            "Extreme score (10.0) → HIGH confidence tier",
            PASS,
            f"P = {r_extreme.calibrated_probability:.4f}",
        )
    else:
        check(
            "Extreme score → HIGH confidence tier",
            FAIL,
            f"Got: {r_extreme.confidence_tier}",
        )

    if r_boundary.confidence_tier == ConfidenceTier.LOW:
        check(
            "Boundary score (tau) → LOW confidence tier",
            PASS,
            f"P = {r_boundary.calibrated_probability:.4f}",
        )
    else:
        check(
            "Boundary score → LOW confidence tier",
            FAIL,
            f"Got: {r_boundary.confidence_tier}",
        )

except Exception as exc:
    check("Confidence tier check", FAIL, str(exc))

# =====================================================================
# Check 7: Batch calibration
# =====================================================================
print("\n" + "=" * 70)
print("Check 7 - Batch calibration consistency")
print("=" * 70)

try:
    cal = ThresholdCalibrator(ThresholdMode.BALANCED)
    scores = [-3.0, -1.0, 0.0, 1.7633, 2.5, 5.0]
    batch = cal.calibrate_batch(scores)
    individual = [cal.calibrate(s) for s in scores]

    if len(batch) == len(scores):
        check(
            f"Batch returns {len(scores)} results for {len(scores)} inputs",
            PASS,
        )
    else:
        check("Batch length matches input", FAIL, f"Got {len(batch)}")

    all_match = all(b == i for b, i in zip(batch, individual))
    if all_match:
        check(
            "Batch results match individual calibrate() calls (6 scores)",
            PASS,
        )
    else:
        check("Batch matches individual results", FAIL)

except Exception as exc:
    check("Batch calibration check", FAIL, str(exc))

# =====================================================================
# Check 8: threshold_info() metadata
# =====================================================================
print("\n" + "=" * 70)
print("Check 8 - threshold_info() metadata completeness")
print("=" * 70)

try:
    cal = ThresholdCalibrator(ThresholdMode.BALANCED)
    info = cal.threshold_info()
    required_keys = (
        "mode",
        "threshold",
        "temperature",
        "description",
        "target_metric",
        "eer_reference",
    )
    missing = [k for k in required_keys if k not in info]
    if not missing:
        check("threshold_info() has all required keys", PASS, str(list(required_keys)))
    else:
        check("threshold_info() missing keys", FAIL, str(missing))

    eer_ref = info.get("eer_reference", {})
    if abs(eer_ref.get("eer_percent", -1) - 0.992) < 0.001:
        check(
            f"EER reference = {eer_ref.get('eer_percent')}% (expected 0.992%)",
            PASS,
        )
    else:
        check("EER reference value", FAIL, str(eer_ref))

    if eer_ref.get("total_samples") == 71237:
        check("EER reference sample count = 71,237", PASS)
    else:
        check("EER reference sample count", FAIL, str(eer_ref.get("total_samples")))

except Exception as exc:
    check("threshold_info() check", FAIL, str(exc))

# =====================================================================
# Summary
# =====================================================================
print("\n" + "=" * 70)
print("SPRINT 3 WEEK 1 VERIFICATION SUMMARY — Dev Khandelwal (23ESKCA035)")
print("=" * 70)

total = len(results)
passed = sum(1 for _, s, _ in results if s == PASS)
failed = sum(1 for _, s, _ in results if s == FAIL)

print(f"  Total checks : {total}")
print(f"  PASS         : {passed}")
print(f"  FAIL         : {failed}")
print()

if failed == 0:
    print(f"  *** ALL {passed}/{total} SPRINT 3 WEEK 1 CHECKS PASSED ***")
    print("  Sprint 3 progress: 14% (Week 1 complete)")
    print()
    sys.exit(0)
else:
    print(f"  [FAIL] {failed} check(s) failed. Review above and fix before submitting.")
    sys.exit(1)
