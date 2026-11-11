"""
ThresholdCalibrator
===================
Audio decision threshold calibration for AASIST-L verdicts.

Converts raw AASIST logit scores (logits[:, 1] — bonafide channel) into
calibrated probabilities and categorical verdicts using three operating
profiles tuned on the ASVspoof 2019 LA evaluation set.

Decision rule:
  - score > threshold → BONAFIDE (real speech)
  - score ≤ threshold → SPOOF    (synthetic/converted speech)

Calibrated probability:
  P(bonafide) = sigmoid((score − threshold) / temperature)

  At score = threshold: P = 0.5  (maximum uncertainty, decision boundary)
  At score >> threshold: P → 1.0 (high-confidence bonafide)
  At score << threshold: P → 0.0 (high-confidence spoof)

Operating profiles (thresholds tuned from EER evaluation, Sprint 3):
  balanced      τ = 1.7633  EER operating point (EER = 0.992%)
  high_security τ = 2.4     Reduced FAR at the cost of higher FRR
  forensic      τ = 0.8     Reduced FRR at the cost of higher FAR

Sprint 3 — Accuracy Optimisation
Dev Khandelwal (23ESKCA035)
Baseline EER source: results/asvspoof_baseline_metrics.json (19-Aug-2026)
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ThresholdMode(str, Enum):
    """Operating profile for audio verdict threshold."""

    BALANCED = "balanced"
    HIGH_SECURITY = "high_security"
    FORENSIC = "forensic"


class Verdict(str, Enum):
    """Binary classification verdict for an audio sample."""

    BONAFIDE = "bonafide"
    SPOOF = "spoof"


class ConfidenceTier(str, Enum):
    """Qualitative confidence tier derived from calibrated probability."""

    HIGH = "high"  # P(predicted class) >= 0.90
    MEDIUM = "medium"  # 0.70 <= P(predicted class) < 0.90
    LOW = "low"  # P(predicted class) < 0.70  (close call)


# ---------------------------------------------------------------------------
# Immutable data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThresholdProfile:
    """Immutable specification of a threshold operating mode."""

    mode: ThresholdMode
    threshold: float
    description: str
    target_metric: str


@dataclass(frozen=True)
class CalibrationResult:
    """Output of a single-sample calibration call."""

    verdict: Verdict
    raw_score: float
    calibrated_probability: float  # P(bonafide) ∈ [0.0, 1.0]
    confidence_tier: ConfidenceTier
    active_mode: ThresholdMode
    active_threshold: float


# ---------------------------------------------------------------------------
# ThresholdCalibrator
# ---------------------------------------------------------------------------


class ThresholdCalibrator:
    """
    Calibrate AASIST-L raw logit scores into verdicts and probabilities.

    Usage::

        from app.services.threshold_calibrator import (
            ThresholdCalibrator, ThresholdMode
        )

        cal = ThresholdCalibrator(ThresholdMode.BALANCED)
        result = cal.calibrate(raw_logit_score)
        print(result.verdict, result.calibrated_probability)
    """

    # EER operating threshold — from actual ASVspoof 2019 LA EVAL run.
    # Source: results/asvspoof_baseline_metrics.json
    # Run date: 19-Aug-2026 on 71,237 utterances, AASIST-L pretrained weights.
    EER_THRESHOLD: float = 1.7633
    EER_PERCENT: float = 0.992

    # Temperature for probability calibration (logit-scale units).
    # Chosen so that a ±1.5 logit shift from the boundary maps to roughly
    # P ≈ 0.73 (MEDIUM confidence), which matches empirical score spreads
    # observed on ASVspoof 2019 LA.
    DEFAULT_TEMPERATURE: float = 1.5

    # Available operating profiles.
    PROFILES: dict = {
        ThresholdMode.BALANCED: ThresholdProfile(
            mode=ThresholdMode.BALANCED,
            threshold=1.7633,
            description=(
                "EER operating point — balanced FAR and FRR. "
                "Use for general-purpose detection."
            ),
            target_metric="EER = 0.992% on ASVspoof 2019 LA eval (71,237 samples)",
        ),
        ThresholdMode.HIGH_SECURITY: ThresholdProfile(
            mode=ThresholdMode.HIGH_SECURITY,
            threshold=2.4,
            description=(
                "Low FAR — reduced false acceptance of spoofs. "
                "Use when spoofs must not slip through (e.g. access control)."
            ),
            target_metric="Estimated FAR < 0.5% on ASVspoof 2019 LA (to be measured)",
        ),
        ThresholdMode.FORENSIC: ThresholdProfile(
            mode=ThresholdMode.FORENSIC,
            threshold=0.8,
            description=(
                "Low FRR — reduced missed detections of spoofs. "
                "Use when every potential spoof must be flagged (e.g. forensics)."
            ),
            target_metric="Estimated FRR < 0.5% on ASVspoof 2019 LA (to be measured)",
        ),
    }

    def __init__(
        self,
        mode: ThresholdMode = ThresholdMode.BALANCED,
        temperature: Optional[float] = None,
    ) -> None:
        """
        Initialise the calibrator.

        Args:
            mode:        Operating profile (default: BALANCED).
            temperature: Sigmoid temperature in logit units.
                         Defaults to DEFAULT_TEMPERATURE (1.5).
        """
        self.mode: ThresholdMode = ThresholdMode(mode)
        self.temperature: float = (
            temperature if temperature is not None else self.DEFAULT_TEMPERATURE
        )
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
        self._profile: ThresholdProfile = self.PROFILES[self.mode]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def active_threshold(self) -> float:
        """Decision threshold for the current operating mode."""
        return self._profile.threshold

    @property
    def active_profile(self) -> ThresholdProfile:
        """Full specification of the current operating profile."""
        return self._profile

    # ------------------------------------------------------------------
    # Mode management
    # ------------------------------------------------------------------

    def set_mode(self, mode: ThresholdMode) -> None:
        """
        Switch to a different operating profile at runtime.

        Args:
            mode: New ThresholdMode to activate.
        """
        self.mode = ThresholdMode(mode)
        self._profile = self.PROFILES[self.mode]

    # ------------------------------------------------------------------
    # Core calibration
    # ------------------------------------------------------------------

    def calibrate(self, raw_score: float) -> CalibrationResult:
        """
        Convert a single AASIST raw logit to a calibrated result.

        Args:
            raw_score: logits[:, 1] from AASIST forward pass.
                       Positive = bonafide; negative = spoof.

        Returns:
            CalibrationResult with verdict, probability, and confidence tier.
        """
        threshold = self._profile.threshold

        # Hard decision: BONAFIDE if score exceeds threshold.
        verdict = Verdict.BONAFIDE if raw_score > threshold else Verdict.SPOOF

        # Temperature-scaled sigmoid centred on the active threshold.
        # P(bonafide) = 1 / (1 + exp(-(score - threshold) / T))
        z = (raw_score - threshold) / self.temperature
        # Numerically stable sigmoid: clamp exponent to avoid overflow.
        if z >= 0:
            prob = 1.0 / (1.0 + math.exp(-z))
        else:
            e = math.exp(z)
            prob = e / (1.0 + e)

        # Confidence tier from the probability of the predicted class.
        p_predicted = prob if verdict == Verdict.BONAFIDE else (1.0 - prob)
        if p_predicted >= 0.90:
            tier = ConfidenceTier.HIGH
        elif p_predicted >= 0.70:
            tier = ConfidenceTier.MEDIUM
        else:
            tier = ConfidenceTier.LOW

        return CalibrationResult(
            verdict=verdict,
            raw_score=raw_score,
            calibrated_probability=round(prob, 6),
            confidence_tier=tier,
            active_mode=self.mode,
            active_threshold=threshold,
        )

    def calibrate_batch(self, raw_scores: List[float]) -> List[CalibrationResult]:
        """
        Calibrate a batch of raw scores.

        Args:
            raw_scores: List of logit scores from AASIST forward passes.

        Returns:
            List of CalibrationResult, one per input score, in input order.
        """
        return [self.calibrate(s) for s in raw_scores]

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def threshold_info(self) -> dict:
        """
        Return metadata about the current threshold configuration.

        Returns:
            Dict with keys: mode, threshold, temperature, description,
            target_metric, eer_reference.
        """
        return {
            "mode": self.mode.value,
            "threshold": self._profile.threshold,
            "temperature": self.temperature,
            "description": self._profile.description,
            "target_metric": self._profile.target_metric,
            "eer_reference": {
                "eer_percent": self.EER_PERCENT,
                "eer_threshold": self.EER_THRESHOLD,
                "total_samples": 71237,
                "dataset": "ASVspoof 2019 LA eval",
            },
        }
