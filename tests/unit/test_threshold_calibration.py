"""
Unit tests for ThresholdCalibrator — Sprint 3 Week 1.

Tests cover:
  - All three operating profiles (balanced, high_security, forensic)
  - Hard verdict decisions (BONAFIDE / SPOOF)
  - Calibration math: sigmoid boundary, monotonicity, probability bounds
  - Confidence tier classification (HIGH / MEDIUM / LOW)
  - Determinism across repeated calls
  - Mode switching at runtime (set_mode)
  - Batch calibration correctness and length
  - threshold_info() metadata and EER reference values
  - Edge cases: extreme scores, zero temperature guard

No real model weights or external services required — pure unit tests.

Sprint 3 — Accuracy Optimisation
Dev Khandelwal (23ESKCA035)
"""

import sys
import os
import unittest

# ---------------------------------------------------------------------------
# Path setup — mirrors the pattern used in test_audio_detector.py
# so the audio-service package is importable from tests/unit/.
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_AUDIO_SERVICE = os.path.join(_REPO_ROOT, "services", "audio-service")
if _AUDIO_SERVICE not in sys.path:
    sys.path.insert(0, _AUDIO_SERVICE)

from app.services.threshold_calibrator import (  # noqa: E402
    CalibrationResult,
    ConfidenceTier,
    ThresholdCalibrator,
    ThresholdMode,
    Verdict,
)


class TestThresholdProfiles(unittest.TestCase):
    """Verify that each operating profile has sensible threshold values."""

    def test_balanced_threshold_matches_eer(self):
        """Balanced threshold must equal the EER threshold from evaluation."""
        cal = ThresholdCalibrator(ThresholdMode.BALANCED)
        self.assertAlmostEqual(cal.active_threshold, 1.7633, places=4)

    def test_high_security_threshold_stricter_than_eer(self):
        """High-security threshold must be higher (stricter) than EER."""
        balanced = ThresholdCalibrator(ThresholdMode.BALANCED)
        high_sec = ThresholdCalibrator(ThresholdMode.HIGH_SECURITY)
        self.assertGreater(high_sec.active_threshold, balanced.active_threshold)

    def test_forensic_threshold_more_permissive_than_eer(self):
        """Forensic threshold must be lower (catches more spoofs) than EER."""
        balanced = ThresholdCalibrator(ThresholdMode.BALANCED)
        forensic = ThresholdCalibrator(ThresholdMode.FORENSIC)
        self.assertLess(forensic.active_threshold, balanced.active_threshold)

    def test_all_profiles_in_profiles_dict(self):
        """Every ThresholdMode must have an entry in PROFILES."""
        for mode in ThresholdMode:
            self.assertIn(mode, ThresholdCalibrator.PROFILES)

    def test_active_profile_matches_mode(self):
        """active_profile.mode must match the requested ThresholdMode."""
        for mode in ThresholdMode:
            cal = ThresholdCalibrator(mode)
            self.assertEqual(cal.active_profile.mode, mode)


class TestVerdictDecision(unittest.TestCase):
    """Hard decision rule: score > threshold → BONAFIDE, else SPOOF."""

    def test_above_threshold_is_bonafide_balanced(self):
        cal = ThresholdCalibrator(ThresholdMode.BALANCED)
        result = cal.calibrate(2.5)
        self.assertEqual(result.verdict, Verdict.BONAFIDE)

    def test_below_threshold_is_spoof_balanced(self):
        cal = ThresholdCalibrator(ThresholdMode.BALANCED)
        result = cal.calibrate(0.5)
        self.assertEqual(result.verdict, Verdict.SPOOF)

    def test_all_modes_decide_extreme_high_as_bonafide(self):
        """Extreme high score (10.0) must be BONAFIDE in every mode."""
        for mode in ThresholdMode:
            with self.subTest(mode=mode):
                cal = ThresholdCalibrator(mode)
                self.assertEqual(cal.calibrate(10.0).verdict, Verdict.BONAFIDE)

    def test_all_modes_decide_extreme_low_as_spoof(self):
        """Extreme low score (-10.0) must be SPOOF in every mode."""
        for mode in ThresholdMode:
            with self.subTest(mode=mode):
                cal = ThresholdCalibrator(mode)
                self.assertEqual(cal.calibrate(-10.0).verdict, Verdict.SPOOF)

    def test_score_exactly_at_threshold_is_spoof(self):
        """score == threshold → SPOOF (boundary condition, not strictly above)."""
        cal = ThresholdCalibrator(ThresholdMode.BALANCED)
        result = cal.calibrate(cal.active_threshold)
        self.assertEqual(result.verdict, Verdict.SPOOF)


class TestCalibrationMath(unittest.TestCase):
    """Temperature-scaled sigmoid calibration correctness."""

    def test_at_threshold_probability_is_exactly_half(self):
        """sigmoid((τ - τ) / T) = sigmoid(0) = 0.5 exactly."""
        cal = ThresholdCalibrator(ThresholdMode.BALANCED)
        result = cal.calibrate(cal.active_threshold)
        self.assertAlmostEqual(result.calibrated_probability, 0.5, places=5)

    def test_above_threshold_probability_exceeds_half(self):
        """Score above threshold → P(bonafide) > 0.5."""
        cal = ThresholdCalibrator(ThresholdMode.BALANCED)
        result = cal.calibrate(cal.active_threshold + 0.01)
        self.assertGreater(result.calibrated_probability, 0.5)

    def test_below_threshold_probability_less_than_half(self):
        """Score below threshold → P(bonafide) < 0.5."""
        cal = ThresholdCalibrator(ThresholdMode.BALANCED)
        result = cal.calibrate(cal.active_threshold - 0.01)
        self.assertLess(result.calibrated_probability, 0.5)

    def test_probability_bounded_zero_to_one(self):
        """P(bonafide) must always be in [0.0, 1.0]."""
        cal = ThresholdCalibrator(ThresholdMode.BALANCED)
        for score in [-100.0, -10.0, -1.0, 0.0, 1.7633, 5.0, 100.0]:
            with self.subTest(score=score):
                p = cal.calibrate(score).calibrated_probability
                self.assertGreaterEqual(p, 0.0)
                self.assertLessEqual(p, 1.0)

    def test_probability_monotonically_increasing(self):
        """Higher raw score must always produce higher P(bonafide)."""
        cal = ThresholdCalibrator(ThresholdMode.BALANCED)
        scores = [-5.0, -2.0, 0.0, 1.0, 1.7633, 2.5, 5.0]
        probs = [cal.calibrate(s).calibrated_probability for s in scores]
        for i in range(len(probs) - 1):
            with self.subTest(score_pair=(scores[i], scores[i + 1])):
                self.assertLess(probs[i], probs[i + 1])

    def test_custom_temperature_changes_steepness(self):
        """Lower temperature → steeper sigmoid → higher confidence at same offset."""
        score = 3.0
        cal_sharp = ThresholdCalibrator(ThresholdMode.BALANCED, temperature=0.5)
        cal_flat = ThresholdCalibrator(ThresholdMode.BALANCED, temperature=3.0)
        p_sharp = cal_sharp.calibrate(score).calibrated_probability
        p_flat = cal_flat.calibrate(score).calibrated_probability
        # sharper sigmoid gives higher probability further from boundary
        self.assertGreater(p_sharp, p_flat)

    def test_invalid_temperature_raises(self):
        """Temperature ≤ 0 must raise ValueError."""
        with self.assertRaises(ValueError):
            ThresholdCalibrator(temperature=0.0)
        with self.assertRaises(ValueError):
            ThresholdCalibrator(temperature=-1.5)


class TestConfidenceTiers(unittest.TestCase):
    """Confidence tier thresholds: HIGH ≥ 0.90, MEDIUM ≥ 0.70, LOW < 0.70."""

    def test_extreme_score_is_high_confidence(self):
        cal = ThresholdCalibrator(ThresholdMode.BALANCED)
        self.assertEqual(cal.calibrate(10.0).confidence_tier, ConfidenceTier.HIGH)
        self.assertEqual(cal.calibrate(-10.0).confidence_tier, ConfidenceTier.HIGH)

    def test_boundary_score_is_low_confidence(self):
        """At the decision boundary P(predicted class) = 0.5 → LOW tier."""
        cal = ThresholdCalibrator(ThresholdMode.BALANCED)
        result = cal.calibrate(cal.active_threshold)
        self.assertEqual(result.confidence_tier, ConfidenceTier.LOW)

    def test_confidence_tier_in_valid_enum(self):
        """confidence_tier must always be a ConfidenceTier member."""
        cal = ThresholdCalibrator(ThresholdMode.BALANCED)
        for score in [-10.0, 0.0, 1.7633, 5.0]:
            with self.subTest(score=score):
                tier = cal.calibrate(score).confidence_tier
                self.assertIsInstance(tier, ConfidenceTier)


class TestDeterminism(unittest.TestCase):
    """Same input must always produce identical output."""

    def test_same_score_same_result(self):
        cal = ThresholdCalibrator(ThresholdMode.BALANCED)
        r1 = cal.calibrate(1.9)
        r2 = cal.calibrate(1.9)
        self.assertEqual(r1, r2)

    def test_all_modes_deterministic(self):
        for mode in ThresholdMode:
            with self.subTest(mode=mode):
                cal = ThresholdCalibrator(mode)
                r1 = cal.calibrate(1.5)
                r2 = cal.calibrate(1.5)
                self.assertEqual(r1, r2)


class TestModeSwitch(unittest.TestCase):
    """set_mode() must correctly update the active profile."""

    def test_set_mode_changes_threshold(self):
        cal = ThresholdCalibrator(ThresholdMode.BALANCED)
        old_threshold = cal.active_threshold
        cal.set_mode(ThresholdMode.HIGH_SECURITY)
        self.assertNotAlmostEqual(cal.active_threshold, old_threshold, places=4)
        self.assertGreater(cal.active_threshold, old_threshold)

    def test_set_mode_changes_verdict_at_ambiguous_score(self):
        """A score between forensic and high_security thresholds should flip verdict."""
        # Score = 1.0: above forensic (0.8) but below high_security (2.4)
        # forensic → BONAFIDE (1.0 > 0.8)
        # high_security → SPOOF (1.0 ≤ 2.4)
        cal = ThresholdCalibrator(ThresholdMode.FORENSIC)
        self.assertEqual(cal.calibrate(1.0).verdict, Verdict.BONAFIDE)
        cal.set_mode(ThresholdMode.HIGH_SECURITY)
        self.assertEqual(cal.calibrate(1.0).verdict, Verdict.SPOOF)

    def test_mode_attribute_updated_after_set_mode(self):
        cal = ThresholdCalibrator(ThresholdMode.BALANCED)
        cal.set_mode(ThresholdMode.FORENSIC)
        self.assertEqual(cal.mode, ThresholdMode.FORENSIC)


class TestBatchCalibration(unittest.TestCase):
    """calibrate_batch must return one result per input score."""

    def test_batch_length_matches_input(self):
        cal = ThresholdCalibrator(ThresholdMode.BALANCED)
        scores = [0.5, 1.0, 1.7633, 2.0, 3.0]
        results = cal.calibrate_batch(scores)
        self.assertEqual(len(results), len(scores))

    def test_batch_matches_individual_calls(self):
        cal = ThresholdCalibrator(ThresholdMode.BALANCED)
        scores = [-3.0, 0.0, 1.7633, 3.0]
        batch = cal.calibrate_batch(scores)
        for score, batch_result in zip(scores, batch):
            individual = cal.calibrate(score)
            self.assertEqual(batch_result, individual)

    def test_empty_batch_returns_empty_list(self):
        cal = ThresholdCalibrator(ThresholdMode.BALANCED)
        self.assertEqual(cal.calibrate_batch([]), [])

    def test_batch_result_type(self):
        cal = ThresholdCalibrator(ThresholdMode.BALANCED)
        for result in cal.calibrate_batch([1.0, 2.0]):
            self.assertIsInstance(result, CalibrationResult)


class TestThresholdInfo(unittest.TestCase):
    """threshold_info() must return complete, accurate metadata."""

    def test_contains_all_required_keys(self):
        cal = ThresholdCalibrator()
        info = cal.threshold_info()
        required = (
            "mode",
            "threshold",
            "temperature",
            "description",
            "target_metric",
            "eer_reference",
        )
        for key in required:
            with self.subTest(key=key):
                self.assertIn(key, info)

    def test_eer_reference_values_match_evaluation(self):
        """EER reference must match the actual ASVspoof 2019 LA evaluation results."""
        cal = ThresholdCalibrator()
        ref = cal.threshold_info()["eer_reference"]
        self.assertAlmostEqual(ref["eer_percent"], 0.992, places=3)
        self.assertAlmostEqual(ref["eer_threshold"], 1.7633, places=4)
        self.assertEqual(ref["total_samples"], 71237)

    def test_mode_field_matches_active_mode(self):
        for mode in ThresholdMode:
            with self.subTest(mode=mode):
                cal = ThresholdCalibrator(mode)
                self.assertEqual(cal.threshold_info()["mode"], mode.value)

    def test_threshold_field_matches_active_threshold(self):
        for mode in ThresholdMode:
            with self.subTest(mode=mode):
                cal = ThresholdCalibrator(mode)
                info = cal.threshold_info()
                self.assertAlmostEqual(
                    info["threshold"], cal.active_threshold, places=6
                )


if __name__ == "__main__":
    unittest.main()
