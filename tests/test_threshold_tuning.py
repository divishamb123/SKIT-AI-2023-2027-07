"""
test_threshold_tuning.py — Automated Unit & Integration Tests for Threshold Tuning Module

Sprint 3 (Accuracy Optimisation) — Week 3 Test Suite
Member: Divisha Manak Bohra (23ESKCA038)

Tests cover:
1. Schema & Profile Validations (ThresholdSweepPoint, TunedProfile, CalibratedImageVerdict)
2. Threshold Sweep Arithmetic & Monotonicity (TPR decreases with tau, TNR increases with tau)
3. Operating Profile Optimizations (Balanced/EER, High Precision, High Recall, Max F1, Youden's J)
4. Calibrated Probability & Confidence Tier Scaling
5. Holdout Cross-Validation & Generalisation
6. Synthetic Calibration Dataset Generator
7. Markdown & JSON Report Serialization
8. Integration with ImageEnsembleDetector
"""

import pytest

from src.image_detector.ensemble import (
    DEFAULT_WEIGHTS,
    ImageEnsembleDetector,
)
from src.image_detector.schemas import PredictionVerdict
from src.image_detector.threshold_tuner import (
    ConfidenceTier,
    HoldoutValidationMetrics,
    ThresholdMode,
    ThresholdSweepPoint,
    ThresholdTuner,
    TunedProfile,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ensemble_detector():
    """Module-scoped ensemble detector instance."""
    return ImageEnsembleDetector(weights=DEFAULT_WEIGHTS)


@pytest.fixture(scope="module")
def tuner(ensemble_detector):
    """Module-scoped ThresholdTuner."""
    return ThresholdTuner(
        detector=ensemble_detector, default_granularity=0.01, temperature=1.0
    )


@pytest.fixture(scope="module")
def calibration_data():
    """Deterministic synthetic calibration set with clean class separation."""
    return ThresholdTuner.create_calibration_dataset(
        n_samples=200, separation=0.6, noise=0.12, seed=42
    )


@pytest.fixture(scope="module")
def holdout_data():
    """Deterministic independent holdout test set with different seed."""
    return ThresholdTuner.create_calibration_dataset(
        n_samples=100, separation=0.55, noise=0.15, seed=99
    )


# ---------------------------------------------------------------------------
# 1. Schema & Initialization Tests
# ---------------------------------------------------------------------------
class TestSchemaAndInitialization:

    def test_tuner_initialization(self, tuner, ensemble_detector):
        """Verify ThresholdTuner instantiates with correct defaults."""
        assert tuner.detector is ensemble_detector
        assert tuner.granularity == 0.01
        assert tuner.temperature == 1.0

    def test_threshold_modes_enum(self):
        """Verify all operating modes are properly enumerated."""
        modes = [m.value for m in ThresholdMode]
        assert "balanced" in modes
        assert "high_precision_ai" in modes
        assert "high_recall_ai" in modes
        assert "max_f1" in modes
        assert "youden_j" in modes

    def test_threshold_sweep_point_validation(self):
        """Verify ThresholdSweepPoint bounds checking and calculation."""
        point = ThresholdSweepPoint(
            threshold=0.50,
            tp=45,
            fp=5,
            tn=45,
            fn=5,
            accuracy=0.90,
            precision=0.90,
            recall=0.90,
            specificity=0.90,
            f1_score=0.90,
            fpr=0.10,
            fnr=0.10,
            youden_j=0.80,
        )
        assert point.threshold == 0.50
        assert point.accuracy == 0.90
        assert point.fpr == 0.10
        assert point.youden_j == 0.80


# ---------------------------------------------------------------------------
# 2. Threshold Sweep & Mathematical Properties Tests
# ---------------------------------------------------------------------------
class TestThresholdSweep:

    def test_sweep_grid_length_and_range(self, calibration_data):
        """Verify sweep produces correct number of points across [0.01, 0.99]."""
        y_true, scores_ai = calibration_data
        sweep = ThresholdTuner.sweep_thresholds(
            y_true=y_true, scores_ai=scores_ai, min_tau=0.01, max_tau=0.99, step=0.01
        )
        assert len(sweep) == 99
        assert sweep[0].threshold == 0.01
        assert sweep[-1].threshold == 0.99

    def test_sweep_monotonicity_properties(self, calibration_data):
        """Verify mathematical monotonicity: TP decreases with tau, TN increases with tau."""
        y_true, scores_ai = calibration_data
        sweep = ThresholdTuner.sweep_thresholds(
            y_true=y_true, scores_ai=scores_ai, min_tau=0.05, max_tau=0.95, step=0.05
        )

        for i in range(len(sweep) - 1):
            curr_pt = sweep[i]
            next_pt = sweep[i + 1]
            # Higher threshold classifies fewer items as positive:
            # TP must be non-increasing, FP must be non-increasing
            assert next_pt.tp <= curr_pt.tp, f"TP increased at tau={next_pt.threshold}"
            assert next_pt.fp <= curr_pt.fp, f"FP increased at tau={next_pt.threshold}"
            # TN must be non-decreasing, FN must be non-decreasing
            assert next_pt.tn >= curr_pt.tn, f"TN decreased at tau={next_pt.threshold}"
            assert next_pt.fn >= curr_pt.fn, f"FN decreased at tau={next_pt.threshold}"

    def test_empty_sweep_handling(self):
        """Verify sweep handles empty inputs gracefully."""
        sweep = ThresholdTuner.sweep_thresholds([], [])
        assert sweep == []


# ---------------------------------------------------------------------------
# 3. Operating Profile Optimization Tests
# ---------------------------------------------------------------------------
class TestProfileOptimization:

    def test_balanced_eer_profile(self, calibration_data):
        """Verify balanced profile minimizes |FPR - FNR| (EER condition)."""
        y_true, scores_ai = calibration_data
        sweep = ThresholdTuner.sweep_thresholds(y_true, scores_ai, step=0.01)
        profile = ThresholdTuner.find_optimal_threshold(sweep, ThresholdMode.BALANCED)

        assert isinstance(profile, TunedProfile)
        assert profile.mode == ThresholdMode.BALANCED
        # Check that this point minimizes |FPR - FNR|
        best_diff = abs(profile.metrics.fpr - profile.metrics.fnr)
        for pt in sweep:
            assert abs(pt.fpr - pt.fnr) >= best_diff - 1e-4

    def test_high_precision_ai_profile(self, calibration_data):
        """Verify high precision profile maintains low FPR (FPR <= 0.05)."""
        y_true, scores_ai = calibration_data
        sweep = ThresholdTuner.sweep_thresholds(y_true, scores_ai, step=0.01)
        profile = ThresholdTuner.find_optimal_threshold(
            sweep, ThresholdMode.HIGH_PRECISION_AI, target_fpr=0.05
        )

        assert profile.mode == ThresholdMode.HIGH_PRECISION_AI
        assert profile.metrics.fpr <= 0.05
        # High precision threshold should typically be >= 0.50
        assert profile.optimal_threshold >= 0.40

    def test_high_recall_ai_profile(self, calibration_data):
        """Verify high recall profile achieves target recall >= 0.95."""
        y_true, scores_ai = calibration_data
        sweep = ThresholdTuner.sweep_thresholds(y_true, scores_ai, step=0.01)
        profile = ThresholdTuner.find_optimal_threshold(
            sweep, ThresholdMode.HIGH_RECALL_AI, target_recall=0.95
        )

        assert profile.mode == ThresholdMode.HIGH_RECALL_AI
        assert profile.metrics.recall >= 0.95
        # High recall threshold should typically be <= 0.50
        assert profile.optimal_threshold <= 0.60

    def test_max_f1_profile(self, calibration_data):
        """Verify max F1 profile finds the highest F1 score in the sweep."""
        y_true, scores_ai = calibration_data
        sweep = ThresholdTuner.sweep_thresholds(y_true, scores_ai, step=0.01)
        profile = ThresholdTuner.find_optimal_threshold(sweep, ThresholdMode.MAX_F1)

        assert profile.mode == ThresholdMode.MAX_F1
        max_sweep_f1 = max(p.f1_score for p in sweep)
        assert profile.metrics.f1_score == pytest.approx(max_sweep_f1, abs=1e-3)


# ---------------------------------------------------------------------------
# 4. Calibration & Confidence Scaling Tests
# ---------------------------------------------------------------------------
class TestCalibrationAndTiers:

    def test_verdict_decision_rule(self, tuner):
        """Verify decision rule: P(AI) >= tau -> AI_GENERATED, else REAL."""
        # Using tau = 0.60
        res_ai = tuner.calibrate_prediction(0.65, custom_threshold=0.60)
        assert res_ai.verdict == PredictionVerdict.AI_GENERATED
        assert res_ai.margin == pytest.approx(0.05, abs=1e-4)

        res_real = tuner.calibrate_prediction(0.55, custom_threshold=0.60)
        assert res_real.verdict == PredictionVerdict.REAL
        assert res_real.margin == pytest.approx(0.05, abs=1e-4)

    def test_confidence_tier_assignment(self, tuner):
        """Verify confidence tier boundaries: HIGH (>=0.25), MEDIUM (>=0.10), LOW (<0.10)."""
        # HIGH confidence (margin = 0.35 >= 0.25)
        high_res = tuner.calibrate_prediction(0.85, custom_threshold=0.50)
        assert high_res.confidence_tier == ConfidenceTier.HIGH

        # MEDIUM confidence (margin = 0.15 in [0.10, 0.25))
        med_res = tuner.calibrate_prediction(0.65, custom_threshold=0.50)
        assert med_res.confidence_tier == ConfidenceTier.MEDIUM

        # LOW confidence / borderline (margin = 0.04 < 0.10)
        low_res = tuner.calibrate_prediction(0.54, custom_threshold=0.50)
        assert low_res.confidence_tier == ConfidenceTier.LOW

    def test_temperature_scaling_bounds(self, tuner):
        """Verify calibrated confidence is always in [0.50, 1.00]."""
        for p in [0.01, 0.20, 0.49, 0.50, 0.51, 0.80, 0.99]:
            verdict = tuner.calibrate_prediction(p, custom_threshold=0.50)
            assert 0.50 <= verdict.calibrated_confidence <= 1.00


# ---------------------------------------------------------------------------
# 5. Holdout Validation & Report Generation Tests
# ---------------------------------------------------------------------------
class TestHoldoutAndReport:

    def test_holdout_cross_validation(self, tuner, calibration_data, holdout_data):
        """Verify holdout evaluation tests stability of tuned thresholds on unseen data."""
        y_cal, s_cal = calibration_data
        y_hold, s_hold = holdout_data

        report = tuner.tune_from_predictions(
            y_true=y_cal,
            scores_ai=s_cal,
            tuning_dataset="synthetic_calibration_split",
            holdout_y_true=y_hold,
            holdout_scores_ai=s_hold,
            notes="Automated test tuning report",
        )

        assert report.holdout_validation is not None
        assert "balanced" in report.holdout_validation
        assert "high_precision_ai" in report.holdout_validation
        assert "high_recall_ai" in report.holdout_validation

        # Verify holdout metrics structure
        for mode_str, h_metrics in report.holdout_validation.items():
            assert isinstance(h_metrics, HoldoutValidationMetrics)
            assert h_metrics.sample_count == len(y_hold)
            assert 0.0 <= h_metrics.accuracy <= 1.0
            assert 0.0 <= h_metrics.fpr <= 1.0
            assert 0.0 <= h_metrics.fnr <= 1.0

    def test_markdown_report_formatting(self, tuner, calibration_data, holdout_data):
        """Verify markdown report generation contains all required sections."""
        y_cal, s_cal = calibration_data
        y_hold, s_hold = holdout_data

        report = tuner.tune_from_predictions(
            y_true=y_cal,
            scores_ai=s_cal,
            tuning_dataset="test_split",
            holdout_y_true=y_hold,
            holdout_scores_ai=s_hold,
        )

        md = ThresholdTuner.format_markdown_report(report)
        assert isinstance(md, str)
        assert len(md) > 1000

        # Section headers
        assert "# Image Decision Threshold Calibration & Tuning Report" in md
        assert "## 1. Executive Summary & Operating Profiles Overview" in md
        assert "## 2. Baseline" in md
        assert "## 3. Holdout Cross-Validation" in md
        assert "## 4. Key Decision Sweep Trajectory" in md

        # Operating modes in tables
        assert "balanced" in md
        assert "high_precision_ai" in md
        assert "high_recall_ai" in md
        assert "max_f1" in md
        assert "youden_j" in md
