"""
Unit tests for AASISTDetector (Week 2 — Docker / degraded-mode behaviour).

These tests verify the detector's graceful-degradation logic without requiring
the actual AASIST-L pretrained weights.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

import torch

# Patch the AASIST Model import so tests do not need the full model architecture
with patch.dict("sys.modules", {"app.models.aasist": MagicMock()}):
    from app.services.audio_detector import AASISTDetector


class TestAASISTDetectorDegradedMode(unittest.TestCase):
    """Tests for graceful degradation when weights are absent."""

    def setUp(self):
        self.detector = AASISTDetector()

    # ------------------------------------------------------------------
    # is_ready() before load()
    # ------------------------------------------------------------------

    def test_not_ready_before_load(self):
        """Detector must not report ready before load() is called."""
        self.assertFalse(self.detector.is_ready())

    # ------------------------------------------------------------------
    # health() before load()
    # ------------------------------------------------------------------

    def test_health_degraded_before_load(self):
        """health() must return status=degraded before load() is called."""
        info = self.detector.health()
        self.assertEqual(info["status"], "degraded")
        self.assertFalse(info["model_loaded"])

    # ------------------------------------------------------------------
    # load() with missing weight file → still degraded
    # ------------------------------------------------------------------

    def test_load_missing_weights_stays_degraded(self):
        """load() with a non-existent weight path → degraded mode (no crash)."""
        with patch.dict(
            os.environ,
            {
                "AASIST_WEIGHTS_PATH": "/nonexistent/AASIST-L.pth",
                "AASIST_CONFIG_PATH": "/nonexistent/AASIST-L.conf",
            },
        ):
            d = AASISTDetector()
            d.load()  # must NOT raise
            self.assertFalse(d.is_ready())
            info = d.health()
            self.assertEqual(info["status"], "degraded")
            self.assertIn("error", info)

    # ------------------------------------------------------------------
    # predict_sync() in degraded mode → RuntimeError
    # ------------------------------------------------------------------

    def test_predict_sync_raises_when_degraded(self):
        """predict_sync() must raise RuntimeError in degraded mode."""
        dummy_waveform = torch.zeros(1, 64600)
        with self.assertRaises(RuntimeError):
            self.detector.predict_sync(dummy_waveform)

    # ------------------------------------------------------------------
    # health() structure validation
    # ------------------------------------------------------------------

    def test_health_contains_weights_path(self):
        """health() dict must always include weights_path for diagnostics."""
        info = self.detector.health()
        self.assertIn("weights_path", info)


class TestAASISTDetectorHealthyMode(unittest.TestCase):
    """Tests for the happy path when the model loads successfully."""

    def _make_loaded_detector(self) -> AASISTDetector:
        """Return a detector in 'loaded' state using a mock model."""
        detector = AASISTDetector()
        mock_model = MagicMock()
        # AASIST returns (_, logit); logit[0] has shape [2]
        fake_logit = torch.tensor([[2.0, 1.0]])  # spoof=index-0 > bonafide=index-1
        mock_model.return_value = (None, fake_logit)
        detector._model = mock_model
        detector._model_loaded = True
        detector._device = torch.device("cpu")
        return detector

    def test_is_ready_when_loaded(self):
        detector = self._make_loaded_detector()
        self.assertTrue(detector.is_ready())

    def test_health_healthy_when_loaded(self):
        detector = self._make_loaded_detector()
        info = detector.health()
        self.assertEqual(info["status"], "healthy")
        self.assertTrue(info["model_loaded"])

    def test_predict_sync_returns_expected_keys(self):
        detector = self._make_loaded_detector()
        waveform = torch.zeros(1, 64600)
        result = detector.predict_sync(waveform)
        for key in (
            "ai_probability",
            "confidence_score",
            "verdict",
            "processing_time_ms",
            "device_used",
        ):
            self.assertIn(key, result)

    def test_predict_sync_ai_probability_range(self):
        detector = self._make_loaded_detector()
        waveform = torch.zeros(1, 64600)
        result = detector.predict_sync(waveform)
        self.assertGreaterEqual(result["ai_probability"], 0.0)
        self.assertLessEqual(result["ai_probability"], 1.0)

    def test_predict_sync_verdict_is_valid(self):
        detector = self._make_loaded_detector()
        waveform = torch.zeros(1, 64600)
        result = detector.predict_sync(waveform)
        valid_verdicts = {
            "AI_GENERATED",
            "LIKELY_AI",
            "INCONCLUSIVE",
            "LIKELY_HUMAN",
            "HUMAN_GENERATED",
        }
        self.assertIn(result["verdict"], valid_verdicts)


if __name__ == "__main__":
    unittest.main()
