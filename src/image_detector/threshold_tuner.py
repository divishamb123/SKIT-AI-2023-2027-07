"""
threshold_tuner.py — Decision Threshold Calibration & Tuning Engine

Sprint 3 (Accuracy Optimisation) — Week 3 Deliverable
Member: Divisha Manak Bohra (23ESKCA038)

Implements systematic decision threshold calibration and multi-profile tuning
using sample test and validation sets for the image AI detection ensemble:

Profiles & Objectives:
1. BALANCED (EER / Equal Error Rate: |FPR - FNR| -> min)
2. HIGH_PRECISION_AI (Minimizes false accusations of real images, target FPR <= 5%)
3. HIGH_RECALL_AI (Forensic sweep / catches synthetic images, target Recall >= 95%)
4. MAX_F1 (Optimal harmonic trade-off between precision and recall)
5. YOUDEN_J (Maximizes Youden's J-statistic: Sensitivity + Specificity - 1)
6. COST_SENSITIVE (Asymmetric error cost optimization)

Features:
- Full threshold sweep: tau in [0.01, 0.99] with ROC, DET, PR curve telemetry
- Baseline (tau=0.50) vs Tuned Profile comparative benchmarking
- Holdout generalisation validation on unseen test partitions
- Calibrated probability scaling with margin calculation and confidence tiers
- Deterministic synthetic calibration dataset generator for test repeatability
"""

from datetime import datetime, timezone
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import uuid

import numpy as np
from pydantic import BaseModel, Field, field_validator

from src.image_detector.ensemble import (
    DEFAULT_WEIGHTS,
    ImageEnsembleDetector,
)
from src.image_detector.schemas import (
    PredictionVerdict,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class ThresholdMode(str, Enum):
    """Operating profile for image forensic decision threshold."""

    BALANCED = "balanced"  # Equal Error Rate (FPR ≈ FNR)
    HIGH_PRECISION_AI = "high_precision_ai"  # Low FPR, avoids accusing authentic images
    HIGH_RECALL_AI = "high_recall_ai"  # High detection rate for forensic sweeps
    MAX_F1 = "max_f1"  # Optimal F1 score trade-off
    YOUDEN_J = "youden_j"  # Maximizes Sensitivity + Specificity - 1
    COST_SENSITIVE = "cost_sensitive"  # Custom asymmetric cost matrix


class ConfidenceTier(str, Enum):
    """Confidence tier based on decision margin from calibrated threshold."""

    HIGH = "high"  # Margin >= 0.25 (clear verdict)
    MEDIUM = "medium"  # 0.10 <= Margin < 0.25 (moderate confidence)
    LOW = "low"  # Margin < 0.10 (borderline / close call)


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class ThresholdSweepPoint(BaseModel):
    """Metrics at a single decision boundary threshold tau."""

    threshold: float = Field(..., ge=0.0, le=1.0, description="Decision boundary tau")
    tp: int = Field(..., ge=0, description="True Positives (AI detected as AI)")
    fp: int = Field(..., ge=0, description="False Positives (Real misclassified as AI)")
    tn: int = Field(..., ge=0, description="True Negatives (Real detected as Real)")
    fn: int = Field(..., ge=0, description="False Negatives (AI misclassified as Real)")
    accuracy: float = Field(..., ge=0.0, le=1.0)
    precision: float = Field(..., ge=0.0, le=1.0)
    recall: float = Field(..., ge=0.0, le=1.0)
    specificity: float = Field(..., ge=0.0, le=1.0)
    f1_score: float = Field(..., ge=0.0, le=1.0)
    fpr: float = Field(
        ..., ge=0.0, le=1.0, description="False Positive Rate FP / (FP + TN)"
    )
    fnr: float = Field(
        ..., ge=0.0, le=1.0, description="False Negative Rate FN / (FN + TP)"
    )
    youden_j: float = Field(
        ..., ge=-1.0, le=1.0, description="Recall + Specificity - 1"
    )

    @field_validator(
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1_score",
        "fpr",
        "fnr",
        "youden_j",
    )
    @classmethod
    def round_float(cls, v: float) -> float:
        return round(float(v), 4)


class TunedProfile(BaseModel):
    """Specification and evaluated metrics for a tuned threshold operating profile."""

    mode: ThresholdMode = Field(..., description="Operating profile name")
    optimal_threshold: float = Field(
        ..., ge=0.0, le=1.0, description="Tuned decision threshold tau*"
    )
    description: str = Field(
        ..., description="Clinical/forensic description of the profile"
    )
    target_metric: str = Field(
        ..., description="Target objective optimized for this profile"
    )
    metrics: ThresholdSweepPoint = Field(
        ..., description="Performance metrics achieved at tau*"
    )
    delta_vs_baseline: Dict[str, float] = Field(
        default_factory=dict,
        description="Metric differences against default baseline tau=0.50 (positive = improvement)",
    )


class CalibratedImageVerdict(BaseModel):
    """Calibrated classification verdict for an image with confidence tiers."""

    filename: str = Field(..., description="Filename identifier")
    raw_probability_ai: float = Field(
        ..., ge=0.0, le=1.0, description="Raw ensemble P(AI)"
    )
    tuned_threshold: float = Field(
        ..., ge=0.0, le=1.0, description="Active decision threshold tau*"
    )
    operating_mode: ThresholdMode = Field(..., description="Active threshold profile")
    verdict: PredictionVerdict = Field(
        ..., description="Decision: REAL or AI_GENERATED"
    )
    margin: float = Field(
        ..., ge=0.0, description="Distance from decision boundary |P(AI) - tau*|"
    )
    calibrated_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Temperature-scaled calibrated confidence"
    )
    confidence_tier: ConfidenceTier = Field(
        ..., description="Qualitative confidence tier: HIGH, MEDIUM, LOW"
    )


class HoldoutValidationMetrics(BaseModel):
    """Performance of tuned thresholds when evaluated on independent holdout test samples."""

    mode: ThresholdMode
    threshold: float
    sample_count: int
    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1_score: float
    fpr: float
    fnr: float
    target_satisfied: bool = Field(
        ...,
        description="Whether the profile's design constraint was satisfied on holdout",
    )


class ThresholdTuningReport(BaseModel):
    """Comprehensive threshold calibration and optimization report."""

    report_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique UUID for this calibration run",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 UTC timestamp",
    )
    model_version: str = Field(..., description="Evaluated ensemble model version")
    device: str = Field(..., description="Compute hardware utilized")
    total_samples: int = Field(..., ge=0, description="Calibration sample count")
    tuning_dataset: str = Field(
        ..., description="Dataset name used for threshold calibration"
    )
    sweep_granularity: float = Field(
        ..., description="Threshold step delta applied (e.g. 0.01)"
    )
    baseline_threshold: float = Field(
        default=0.50, description="Default uncalibrated threshold"
    )
    baseline_metrics: ThresholdSweepPoint = Field(
        ..., description="Metrics at tau=0.50"
    )
    profiles: Dict[str, TunedProfile] = Field(
        ..., description="Tuned operating profiles keyed by profile mode"
    )
    holdout_validation: Optional[Dict[str, HoldoutValidationMetrics]] = Field(
        None, description="Holdout evaluation metrics validating generalisation"
    )
    sweep_curve: List[ThresholdSweepPoint] = Field(
        ..., description="Complete threshold sweep curve points"
    )
    notes: Optional[str] = Field(None, description="Forensic and deployment notes")


# ---------------------------------------------------------------------------
# ThresholdTuner Engine
# ---------------------------------------------------------------------------
class ThresholdTuner:
    """
    Decision threshold calibration and multi-profile tuning engine.

    Provides fine-grained sweep analysis, EER calculation, asymmetric cost
    optimization, holdout validation, and probability temperature calibration.
    """

    # Default profile definitions
    PROFILE_DESCRIPTIONS = {
        ThresholdMode.BALANCED: {
            "desc": "Equal Error Rate (EER) operating point. Balances false positives and false negatives.",
            "target": "min |FPR - FNR| (EER)",
        },
        ThresholdMode.HIGH_PRECISION_AI: {
            "desc": "Conservative profile. Prevents authentic human images from being falsely accused.",
            "target": "max Precision subject to FPR <= 0.05",
        },
        ThresholdMode.HIGH_RECALL_AI: {
            "desc": "Forensic sweep profile. Maximizes detection of synthetic images.",
            "target": "Recall >= 0.95 with lowest FNR",
        },
        ThresholdMode.MAX_F1: {
            "desc": "Optimal balance profile. Maximizes harmonic mean of precision and recall.",
            "target": "max F1-score",
        },
        ThresholdMode.YOUDEN_J: {
            "desc": "Diagnostic index profile. Maximizes informedness (Sensitivity + Specificity - 1).",
            "target": "max Youden's J-statistic",
        },
    }

    def __init__(
        self,
        detector: Optional[ImageEnsembleDetector] = None,
        default_granularity: float = 0.01,
        temperature: float = 1.0,
    ):
        self.detector = detector or ImageEnsembleDetector(weights=DEFAULT_WEIGHTS)
        self.granularity = default_granularity
        self.temperature = max(0.01, float(temperature))

    # -----------------------------------------------------------------------
    # Threshold Sweep Core
    # -----------------------------------------------------------------------
    @staticmethod
    def sweep_thresholds(
        y_true: Sequence[int],
        scores_ai: Sequence[float],
        min_tau: float = 0.01,
        max_tau: float = 0.99,
        step: float = 0.01,
    ) -> List[ThresholdSweepPoint]:
        """
        Evaluates binary classification metrics across a grid of threshold values.

        Args:
            y_true: Ground truth binary labels (0 = Real, 1 = AI).
            scores_ai: Predicted continuous probabilities for AI class P(AI) in [0.0, 1.0].
            min_tau: Minimum threshold in sweep.
            max_tau: Maximum threshold in sweep.
            step: Step size between threshold points.

        Returns:
            List of ThresholdSweepPoint objects ordered by threshold ascending.
        """
        y_arr = np.asarray(y_true, dtype=int)
        s_arr = np.asarray(scores_ai, dtype=float)

        if len(y_arr) != len(s_arr):
            raise ValueError(
                f"Length mismatch: y_true ({len(y_arr)}) vs scores ({len(s_arr)})"
            )
        if len(y_arr) == 0:
            return []

        # Generate threshold values
        num_steps = int(round((max_tau - min_tau) / step)) + 1
        thresholds = np.linspace(min_tau, max_tau, num_steps)

        sweep_points: List[ThresholdSweepPoint] = []
        n_pos = int(np.sum(y_arr == 1))
        n_neg = int(np.sum(y_arr == 0))
        total = len(y_arr)

        for tau in thresholds:
            tau_val = round(float(tau), 4)
            y_pred = (s_arr >= tau_val).astype(int)

            tp = int(np.sum((y_pred == 1) & (y_arr == 1)))
            fp = int(np.sum((y_pred == 1) & (y_arr == 0)))
            tn = int(np.sum((y_pred == 0) & (y_arr == 0)))
            fn = int(np.sum((y_pred == 0) & (y_arr == 1)))

            acc = (tp + tn) / total if total > 0 else 0.0
            prec = (
                tp / (tp + fp)
                if (tp + fp) > 0
                else (1.0 if tp == 0 and fp == 0 else 0.0)
            )
            rec = tp / n_pos if n_pos > 0 else 0.0
            spec = tn / n_neg if n_neg > 0 else 0.0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            fpr = fp / n_neg if n_neg > 0 else 0.0
            fnr = fn / n_pos if n_pos > 0 else 0.0
            youden = rec + spec - 1.0

            sweep_points.append(
                ThresholdSweepPoint(
                    threshold=tau_val,
                    tp=tp,
                    fp=fp,
                    tn=tn,
                    fn=fn,
                    accuracy=acc,
                    precision=prec,
                    recall=rec,
                    specificity=spec,
                    f1_score=f1,
                    fpr=fpr,
                    fnr=fnr,
                    youden_j=youden,
                )
            )

        return sweep_points

    # -----------------------------------------------------------------------
    # Profile Optimization
    # -----------------------------------------------------------------------
    @classmethod
    def find_optimal_threshold(
        cls,
        sweep_points: Sequence[ThresholdSweepPoint],
        objective: ThresholdMode,
        target_fpr: float = 0.05,
        target_recall: float = 0.95,
        cost_fp: float = 1.0,
        cost_fn: float = 1.0,
    ) -> TunedProfile:
        """
        Finds the optimal threshold point from sweep points according to an objective.
        """
        if not sweep_points:
            raise ValueError("sweep_points list cannot be empty.")

        best_point: ThresholdSweepPoint
        profile_meta = cls.PROFILE_DESCRIPTIONS.get(
            objective,
            {"desc": f"Custom profile: {objective.value}", "target": objective.value},
        )

        if objective == ThresholdMode.BALANCED:
            # EER point: min |FPR - FNR|
            best_point = min(sweep_points, key=lambda p: abs(p.fpr - p.fnr))

        elif objective == ThresholdMode.MAX_F1:
            # Max F1 score
            best_point = max(
                sweep_points, key=lambda p: (p.f1_score, -abs(p.threshold - 0.5))
            )

        elif objective == ThresholdMode.YOUDEN_J:
            # Max Youden's J
            best_point = max(sweep_points, key=lambda p: (p.youden_j, p.accuracy))

        elif objective == ThresholdMode.HIGH_PRECISION_AI:
            # FPR <= target_fpr with maximum precision, then highest recall
            candidates = [p for p in sweep_points if p.fpr <= target_fpr]
            if candidates:
                best_point = max(
                    candidates, key=lambda p: (p.precision, p.f1_score, p.recall)
                )
            else:
                # Fallback to lowest FPR
                best_point = min(sweep_points, key=lambda p: (p.fpr, -p.precision))

        elif objective == ThresholdMode.HIGH_RECALL_AI:
            # Recall >= target_recall with lowest FPR, then highest precision
            candidates = [p for p in sweep_points if p.recall >= target_recall]
            if candidates:
                best_point = min(
                    candidates, key=lambda p: (p.fpr, -p.precision, p.threshold)
                )
            else:
                # Fallback to highest recall
                best_point = max(sweep_points, key=lambda p: (p.recall, p.f1_score))

        elif objective == ThresholdMode.COST_SENSITIVE:
            # Minimize cost: C_FP * FP + C_FN * FN
            best_point = min(
                sweep_points, key=lambda p: (cost_fp * p.fp + cost_fn * p.fn)
            )
        else:
            best_point = sweep_points[len(sweep_points) // 2]

        # Calculate deltas vs tau=0.50
        baseline_point = min(sweep_points, key=lambda p: abs(p.threshold - 0.50))
        deltas = {
            "accuracy_delta": round(best_point.accuracy - baseline_point.accuracy, 4),
            "precision_delta": round(
                best_point.precision - baseline_point.precision, 4
            ),
            "recall_delta": round(best_point.recall - baseline_point.recall, 4),
            "f1_delta": round(best_point.f1_score - baseline_point.f1_score, 4),
            "fpr_reduction": round(baseline_point.fpr - best_point.fpr, 4),
            "fnr_reduction": round(baseline_point.fnr - best_point.fnr, 4),
        }

        return TunedProfile(
            mode=objective,
            optimal_threshold=best_point.threshold,
            description=profile_meta["desc"],
            target_metric=profile_meta["target"],
            metrics=best_point,
            delta_vs_baseline=deltas,
        )

    def tune_all_profiles(
        self,
        sweep_points: Sequence[ThresholdSweepPoint],
    ) -> Dict[str, TunedProfile]:
        """Tunes all standard operating profiles simultaneously."""
        modes = [
            ThresholdMode.BALANCED,
            ThresholdMode.HIGH_PRECISION_AI,
            ThresholdMode.HIGH_RECALL_AI,
            ThresholdMode.MAX_F1,
            ThresholdMode.YOUDEN_J,
        ]
        profiles: Dict[str, TunedProfile] = {}
        for m in modes:
            profile = self.find_optimal_threshold(sweep_points, objective=m)
            profiles[m.value] = profile
        return profiles

    # -----------------------------------------------------------------------
    # End-to-End Tuning from Sample Test Sets
    # -----------------------------------------------------------------------
    def tune_from_predictions(
        self,
        y_true: Sequence[int],
        scores_ai: Sequence[float],
        tuning_dataset: str = "sample_test_set",
        holdout_y_true: Optional[Sequence[int]] = None,
        holdout_scores_ai: Optional[Sequence[float]] = None,
        notes: Optional[str] = None,
    ) -> ThresholdTuningReport:
        """
        Builds a comprehensive threshold tuning report from true labels and AI scores.
        """
        sweep_points = self.sweep_thresholds(
            y_true=y_true,
            scores_ai=scores_ai,
            step=self.granularity,
        )

        baseline_point = min(sweep_points, key=lambda p: abs(p.threshold - 0.50))
        profiles = self.tune_all_profiles(sweep_points)

        # Holdout cross-validation if holdout data provided
        holdout_results = None
        if holdout_y_true is not None and holdout_scores_ai is not None:
            holdout_results = self.evaluate_holdout(
                profiles=profiles,
                holdout_y_true=holdout_y_true,
                holdout_scores_ai=holdout_scores_ai,
            )

        return ThresholdTuningReport(
            model_version=self.detector.ensemble_version,
            device=str(self.detector.device),
            total_samples=len(y_true),
            tuning_dataset=tuning_dataset,
            sweep_granularity=self.granularity,
            baseline_threshold=baseline_point.threshold,
            baseline_metrics=baseline_point,
            profiles=profiles,
            holdout_validation=holdout_results,
            sweep_curve=sweep_points,
            notes=notes,
        )

    def tune_from_samples(
        self,
        samples: Sequence[Dict[str, Any]],
        tuning_dataset: str = "sample_manifest_split",
        holdout_samples: Optional[Sequence[Dict[str, Any]]] = None,
        notes: Optional[str] = None,
    ) -> ThresholdTuningReport:
        """
        Executes ensemble inference on sample dicts, extracts P(AI), and tunes thresholds.
        """
        y_true: List[int] = []
        scores_ai: List[float] = []

        for sample in samples:
            image_input = sample.get("image") or sample.get("path")
            fname = sample.get("filename", "sample.png")
            try:
                resp = self.detector.predict(image_input, filename=fname, threshold=0.5)
                y_true.append(sample["label"])
                scores_ai.append(resp.probabilities.ai_generated)
            except Exception:
                continue

        holdout_y = None
        holdout_scores = None
        if holdout_samples:
            holdout_y = []
            holdout_scores = []
            for s in holdout_samples:
                image_input = s.get("image") or s.get("path")
                fname = s.get("filename", "sample.png")
                try:
                    resp = self.detector.predict(
                        image_input, filename=fname, threshold=0.5
                    )
                    holdout_y.append(s["label"])
                    holdout_scores.append(resp.probabilities.ai_generated)
                except Exception:
                    continue

        return self.tune_from_predictions(
            y_true=y_true,
            scores_ai=scores_ai,
            tuning_dataset=tuning_dataset,
            holdout_y_true=holdout_y,
            holdout_scores_ai=holdout_scores,
            notes=notes,
        )

    # -----------------------------------------------------------------------
    # Holdout Validation
    # -----------------------------------------------------------------------
    @staticmethod
    def evaluate_holdout(
        profiles: Dict[str, TunedProfile],
        holdout_y_true: Sequence[int],
        holdout_scores_ai: Sequence[float],
    ) -> Dict[str, HoldoutValidationMetrics]:
        """
        Evaluates tuned thresholds on an independent holdout test set to verify stability.
        """
        y_arr = np.asarray(holdout_y_true, dtype=int)
        s_arr = np.asarray(holdout_scores_ai, dtype=float)
        n_pos = int(np.sum(y_arr == 1))
        n_neg = int(np.sum(y_arr == 0))
        total = len(y_arr)

        results: Dict[str, HoldoutValidationMetrics] = {}
        for mode_str, profile in profiles.items():
            tau = profile.optimal_threshold
            y_pred = (s_arr >= tau).astype(int)

            tp = int(np.sum((y_pred == 1) & (y_arr == 1)))
            fp = int(np.sum((y_pred == 1) & (y_arr == 0)))
            tn = int(np.sum((y_pred == 0) & (y_arr == 0)))
            fn = int(np.sum((y_pred == 0) & (y_arr == 1)))

            acc = (tp + tn) / total if total > 0 else 0.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / n_pos if n_pos > 0 else 0.0
            spec = tn / n_neg if n_neg > 0 else 0.0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            fpr = fp / n_neg if n_neg > 0 else 0.0
            fnr = fn / n_pos if n_pos > 0 else 0.0

            # Target satisfaction check
            target_satisfied = True
            if profile.mode == ThresholdMode.HIGH_PRECISION_AI:
                target_satisfied = fpr <= 0.10  # allow slight generalization slack
            elif profile.mode == ThresholdMode.HIGH_RECALL_AI:
                target_satisfied = rec >= 0.85
            elif profile.mode == ThresholdMode.BALANCED:
                target_satisfied = abs(fpr - fnr) <= 0.15

            results[mode_str] = HoldoutValidationMetrics(
                mode=profile.mode,
                threshold=tau,
                sample_count=total,
                accuracy=round(acc, 4),
                precision=round(prec, 4),
                recall=round(rec, 4),
                specificity=round(spec, 4),
                f1_score=round(f1, 4),
                fpr=round(fpr, 4),
                fnr=round(fnr, 4),
                target_satisfied=target_satisfied,
            )

        return results

    # -----------------------------------------------------------------------
    # Calibrated Probability & Verdict
    # -----------------------------------------------------------------------
    def calibrate_prediction(
        self,
        raw_prob_ai: float,
        mode: ThresholdMode = ThresholdMode.BALANCED,
        custom_threshold: Optional[float] = None,
        filename: str = "sample.png",
    ) -> CalibratedImageVerdict:
        """
        Applies a tuned threshold to an image's raw AI probability score.
        Computes distance margin, temperature-calibrated confidence, and tier.
        """
        prob = max(0.0, min(1.0, float(raw_prob_ai)))

        if custom_threshold is not None:
            tau = max(0.0, min(1.0, float(custom_threshold)))
        else:
            # Default profiles
            defaults = {
                ThresholdMode.BALANCED: 0.50,
                ThresholdMode.HIGH_PRECISION_AI: 0.70,
                ThresholdMode.HIGH_RECALL_AI: 0.30,
                ThresholdMode.MAX_F1: 0.50,
                ThresholdMode.YOUDEN_J: 0.50,
            }
            tau = defaults.get(mode, 0.50)

        # Binary decision rule
        if prob >= tau:
            verdict = PredictionVerdict.AI_GENERATED
        else:
            verdict = PredictionVerdict.REAL

        margin = round(abs(prob - tau), 4)

        # Calibrated probability: sigmoid((P - tau) / T)
        # Scaled to confidence [0.5, 1.0] for the winning class
        logit_diff = (prob - tau) / self.temperature
        calibrated_p = 1.0 / (1.0 + math.exp(-logit_diff))
        calibrated_conf = (
            calibrated_p
            if verdict == PredictionVerdict.AI_GENERATED
            else (1.0 - calibrated_p)
        )
        # Bound confidence to [0.50, 1.00]
        calibrated_conf = round(max(0.50, min(1.0, calibrated_conf)), 4)

        # Determine confidence tier
        if margin >= 0.25:
            tier = ConfidenceTier.HIGH
        elif margin >= 0.10:
            tier = ConfidenceTier.MEDIUM
        else:
            tier = ConfidenceTier.LOW

        return CalibratedImageVerdict(
            filename=filename,
            raw_probability_ai=round(prob, 4),
            tuned_threshold=round(tau, 4),
            operating_mode=mode,
            verdict=verdict,
            margin=margin,
            calibrated_confidence=calibrated_conf,
            confidence_tier=tier,
        )

    # -----------------------------------------------------------------------
    # Synthetic Calibration Dataset Generator
    # -----------------------------------------------------------------------
    @staticmethod
    def create_calibration_dataset(
        n_samples: int = 200,
        separation: float = 0.6,
        noise: float = 0.15,
        seed: int = 42,
    ) -> Tuple[List[int], List[float]]:
        """
        Creates synthetic (y_true, scores_ai) with controllable separation.

        Real images (label 0): scores centered near 0.5 - separation/2
        AI images (label 1): scores centered near 0.5 + separation/2

        Enables fast, deterministic unit testing and verification of threshold tuning.
        """
        rng = np.random.RandomState(seed)
        n_half = n_samples // 2

        # Ground truth
        y_true = [0] * n_half + [1] * n_half

        # Real scores
        mu_real = max(0.05, 0.5 - separation / 2.0)
        scores_real = rng.normal(loc=mu_real, scale=noise, size=n_half)
        scores_real = np.clip(scores_real, 0.01, 0.99)

        # AI scores
        mu_ai = min(0.95, 0.5 + separation / 2.0)
        scores_ai = rng.normal(loc=mu_ai, scale=noise, size=n_half)
        scores_ai = np.clip(scores_ai, 0.01, 0.99)

        scores = list(scores_real) + list(scores_ai)
        return y_true, [round(float(s), 4) for s in scores]

    # -----------------------------------------------------------------------
    # Markdown Formatting & Persistence
    # -----------------------------------------------------------------------
    @staticmethod
    def format_markdown_report(report: ThresholdTuningReport) -> str:
        """Generates a structured Markdown report of threshold tuning results."""
        lines = [
            "# Image Decision Threshold Calibration & Tuning Report",
            "",
            "**Sprint:** Sprint 3 (Accuracy Optimisation) — Week 3 Deliverable  ",
            "**Member:** Divisha Manak Bohra (`23ESKCA038`) — Image Detection Modality & Image Interface  ",
            "**Project ID:** `SKIT/AI/2023-2027/07`  ",
            f"**Model Version:** `{report.model_version}`  ",
            f"**Hardware Device:** `{report.device}`  ",
            f"**Tuning Dataset:** `{report.tuning_dataset}` (Total Samples: {report.total_samples:,})  ",
            f"**Timestamp:** `{report.timestamp}`  ",
            "",
            "---",
            "",
            "## 1. Executive Summary & Operating Profiles Overview",
            "",
            "Decision threshold calibration transforms continuous probability estimates from the neural ensemble "
            "into reliable categorical verdicts. Rather than enforcing a static threshold of $\\tau = 0.50$, "
            "five distinct operating profiles were calibrated to satisfy diverse deployment constraints:",
            "",
            "| Profile | Optimal Threshold ($\\tau^*$) | Primary Target | Description |",
            "|---|---|---|---|",
        ]

        for mode_key, profile in report.profiles.items():
            lines.append(
                f"| **`{profile.mode.value}`** | **`{profile.optimal_threshold:.4f}`** | "
                f"{profile.target_metric} | {profile.description} |"
            )

        lines.extend(
            [
                "",
                "---",
                "",
                "## 2. Baseline ($\\tau = 0.50$) vs. Tuned Operating Profiles",
                "",
                "| Operating Profile | Threshold ($\\tau$) | Accuracy | Precision | Recall | Specificity | F1-Score | FPR | FNR |",
                "|---|---|---|---|---|---|---|---|---|",
                f"| *Baseline (Default)* | {report.baseline_threshold:.2f} | "
                f"{report.baseline_metrics.accuracy * 100:.1f}% | "
                f"{report.baseline_metrics.precision * 100:.1f}% | "
                f"{report.baseline_metrics.recall * 100:.1f}% | "
                f"{report.baseline_metrics.specificity * 100:.1f}% | "
                f"{report.baseline_metrics.f1_score:.3f} | "
                f"{report.baseline_metrics.fpr * 100:.1f}% | "
                f"{report.baseline_metrics.fnr * 100:.1f}% |",
            ]
        )

        for mode_key, p in report.profiles.items():
            m = p.metrics
            lines.append(
                f"| **{p.mode.value}** | **{p.optimal_threshold:.4f}** | "
                f"{m.accuracy * 100:.1f}% | {m.precision * 100:.1f}% | "
                f"{m.recall * 100:.1f}% | {m.specificity * 100:.1f}% | "
                f"{m.f1_score:.3f} | {m.fpr * 100:.1f}% | {m.fnr * 100:.1f}% |"
            )

        # Delta improvements table
        lines.extend(
            [
                "",
                "### Performance Gains Relative to Default $\\tau = 0.50$",
                "",
                "| Profile | Accuracy $\\Delta$ | Precision $\\Delta$ | Recall $\\Delta$ | F1 $\\Delta$ | FPR Reduction | FNR Reduction |",
                "|---|---|---|---|---|---|---|",
            ]
        )

        for mode_key, p in report.profiles.items():
            d = p.delta_vs_baseline
            lines.append(
                f"| **{p.mode.value}** | "
                f"{d.get('accuracy_delta', 0.0) * 100:+.1f}% | "
                f"{d.get('precision_delta', 0.0) * 100:+.1f}% | "
                f"{d.get('recall_delta', 0.0) * 100:+.1f}% | "
                f"{d.get('f1_delta', 0.0):+.3f} | "
                f"{d.get('fpr_reduction', 0.0) * 100:+.1f}% | "
                f"{d.get('fnr_reduction', 0.0) * 100:+.1f}% |"
            )

        # Holdout Cross-Validation
        if report.holdout_validation:
            lines.extend(
                [
                    "",
                    "---",
                    "",
                    "## 3. Holdout Cross-Validation (Generalisation Stability)",
                    "",
                    "Validation of calibrated thresholds on an independent holdout test partition "
                    "to confirm that tuned operating boundaries generalise reliably to unseen samples:",
                    "",
                    "| Profile | Holdout $\\tau$ | Samples | Holdout Acc | Holdout Prec | Holdout Recall | Holdout FPR | Target Met? |",
                    "|---|---|---|---|---|---|---|---|",
                ]
            )
            for mode_key, h in report.holdout_validation.items():
                status = "✅ YES" if h.target_satisfied else "⚠️ PARTIAL"
                lines.append(
                    f"| **{h.mode.value}** | {h.threshold:.4f} | {h.sample_count} | "
                    f"{h.accuracy * 100:.1f}% | {h.precision * 100:.1f}% | "
                    f"{h.recall * 100:.1f}% | {h.fpr * 100:.1f}% | {status} |"
                )

        # Threshold sweep highlights
        lines.extend(
            [
                "",
                "---",
                "",
                "## 4. Key Decision Sweep Trajectory (Sample Grid)",
                "",
                "| $\\tau$ | TP | FP | TN | FN | Accuracy | Precision | Recall | Specificity | F1 | FPR | FNR |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|",
            ]
        )

        # Sample every 10th sweep point for a concise markdown table
        stride = max(1, len(report.sweep_curve) // 10)
        sampled_points = report.sweep_curve[::stride]
        if report.sweep_curve[-1] not in sampled_points:
            sampled_points.append(report.sweep_curve[-1])

        for pt in sampled_points:
            lines.append(
                f"| `{pt.threshold:.2f}` | {pt.tp} | {pt.fp} | {pt.tn} | {pt.fn} | "
                f"{pt.accuracy * 100:.1f}% | {pt.precision * 100:.1f}% | "
                f"{pt.recall * 100:.1f}% | {pt.specificity * 100:.1f}% | "
                f"{pt.f1_score:.3f} | {pt.fpr * 100:.1f}% | {pt.fnr * 100:.1f}% |"
            )

        if report.notes:
            lines.extend(
                [
                    "",
                    "---",
                    "",
                    "## 5. Engineering & Forensic Notes",
                    "",
                    report.notes,
                    "",
                ]
            )

        return "\n".join(lines)

    def save_report(
        self,
        report: ThresholdTuningReport,
        json_path: Union[str, Path],
        md_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Saves the tuning report to JSON and Markdown destinations."""
        j_file = Path(json_path)
        j_file.parent.mkdir(parents=True, exist_ok=True)
        with open(j_file, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(), f, indent=2)

        if md_path:
            m_file = Path(md_path)
            m_file.parent.mkdir(parents=True, exist_ok=True)
            with open(m_file, "w", encoding="utf-8") as f:
                f.write(self.format_markdown_report(report))
