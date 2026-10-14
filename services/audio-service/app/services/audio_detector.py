"""
AASISTDetector
==============
Wraps the AASIST-L pretrained model for synchronous inference.

Model weight and config paths are resolved (in order of preference):
  1. Environment variables  AASIST_WEIGHTS_PATH  /  AASIST_CONFIG_PATH
  2. Compile-time defaults  /models/aasist_l_v1.0/AASIST-L.pth  /  .conf

Graceful-degradation mode
--------------------------
If the weight file is absent at startup the detector enters *degraded* mode:
  • detector.is_ready() returns False
  • detector.predict_sync() raises RuntimeError with a clear message
  • /internal/health reports {"status": "degraded", "model_loaded": false}

This prevents the container from crash-looping when model weights have not
yet been volume-mounted, which is the expected state in CI / local dev.
"""

import json
import logging
import os
import time

import torch

from app.models.aasist import Model as AASIST

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Path resolution — environment variables take precedence over defaults
# ------------------------------------------------------------------
_DEFAULT_WEIGHTS = "/models/aasist_l_v1.0/AASIST-L.pth"
_DEFAULT_CONFIG = "/models/aasist_l_v1.0/AASIST-L.conf"

WEIGHTS_PATH: str = os.environ.get("AASIST_WEIGHTS_PATH", _DEFAULT_WEIGHTS)
CONFIG_PATH: str = os.environ.get("AASIST_CONFIG_PATH", _DEFAULT_CONFIG)


# ------------------------------------------------------------------
# Score → verdict mapping
# Thresholds match the project Engineering Blueprint (Section 4.3).
# ------------------------------------------------------------------
def _score_to_verdict(s: float) -> str:
    if s >= 0.80:
        return "AI_GENERATED"
    if s >= 0.60:
        return "LIKELY_AI"
    if s >= 0.40:
        return "INCONCLUSIVE"
    if s >= 0.20:
        return "LIKELY_HUMAN"
    return "HUMAN_GENERATED"


class AASISTDetector:
    """
    Lifecycle:
        detector = AASISTDetector()
        detector.load()            # called once at FastAPI startup
        result  = detector.predict_sync(waveform)
    """

    def __init__(self) -> None:
        self._model: AASIST | None = None
        self._device: torch.device | None = None
        self._model_loaded: bool = False
        self._load_error: str | None = None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        """Return True only when model weights are fully loaded."""
        return self._model_loaded

    def health(self) -> dict:
        """Return a structured health dict suitable for /internal/health."""
        if self._model_loaded:
            return {
                "status": "healthy",
                "model_loaded": True,
                "weights_path": WEIGHTS_PATH,
                "device": str(self._device),
            }
        return {
            "status": "degraded",
            "model_loaded": False,
            "weights_path": WEIGHTS_PATH,
            "error": self._load_error or "Model weights not loaded.",
        }

    # ------------------------------------------------------------------
    # Startup / load
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Attempt to load the AASIST-L weights.

        On success  → self._model_loaded = True
        On failure  → self._load_error is set; service stays up in degraded mode.
        """
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Check weight file exists before attempting to load ---
        if not os.path.isfile(WEIGHTS_PATH):
            self._load_error = (
                f"Weight file not found: {WEIGHTS_PATH}. "
                "Mount the pretrained weights via a Docker volume "
                "(e.g. - ./models:/models) or set the "
                "AASIST_WEIGHTS_PATH environment variable."
            )
            logger.warning(
                "AASIST-L weights missing — running in degraded mode. %s",
                self._load_error,
            )
            return

        if not os.path.isfile(CONFIG_PATH):
            self._load_error = (
                f"Config file not found: {CONFIG_PATH}. "
                "Ensure the model directory contains both AASIST-L.pth and AASIST-L.conf."
            )
            logger.warning(
                "AASIST-L config missing — running in degraded mode. %s",
                self._load_error,
            )
            return

        try:
            with open(CONFIG_PATH) as fh:
                config = json.load(fh)

            self._model = AASIST(config["model_config"])
            state = torch.load(
                WEIGHTS_PATH, map_location=self._device, weights_only=True
            )
            self._model.load_state_dict(state)
            self._model.eval().to(self._device)
            self._model_loaded = True
            logger.info(
                "AASIST-L loaded successfully. device=%s  weights=%s",
                self._device,
                WEIGHTS_PATH,
            )
        except Exception as exc:  # noqa: BLE001
            self._load_error = f"Failed to load AASIST-L weights: {exc}"
            logger.error("AASIST-L load failed — degraded mode. %s", exc)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_sync(self, waveform: torch.Tensor) -> dict:
        """
        Run synchronous inference.

        Args:
            waveform: Tensor of shape [1, 64600] (16 kHz, mono, 4.04 s).

        Returns:
            dict with keys: ai_probability, confidence_score, verdict,
                            processing_time_ms, device_used.

        Raises:
            RuntimeError: if the model is not loaded (degraded mode).
        """
        if not self._model_loaded or self._model is None:
            raise RuntimeError(
                "AASIST-L model is not loaded. "
                f"Reason: {self._load_error or 'unknown'}. "
                "Mount pretrained weights or set AASIST_WEIGHTS_PATH."
            )

        t0 = time.perf_counter()
        waveform = waveform.to(self._device)

        with torch.no_grad():
            _, logit = self._model(waveform)
            # logit shape: [batch=1, 2]
            # AASIST convention: index 0 = spoof (AI-generated), index 1 = bonafide (human).
            probs = torch.softmax(logit[0], dim=-1)
            spoof_prob = probs[0].item()

        return {
            "ai_probability": round(spoof_prob, 4),
            "confidence_score": round(float(probs.max()), 4),
            "verdict": _score_to_verdict(spoof_prob),
            "processing_time_ms": int((time.perf_counter() - t0) * 1000),
            "device_used": str(self._device),
        }
