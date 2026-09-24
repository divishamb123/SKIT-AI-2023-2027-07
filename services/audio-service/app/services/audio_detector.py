import time
import torch
import json
from app.models.aasist import Model as AASIST

WEIGHTS_PATH = "/models/aasist_l_v1.0/AASIST-L.pth"
CONFIG_PATH = "/models/aasist_l_v1.0/AASIST-L.conf"


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
    def load(self):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open(CONFIG_PATH) as f:
            config = json.load(f)
        self._model = AASIST(config["model_config"])
        state = torch.load(WEIGHTS_PATH, map_location=self._device)
        self._model.load_state_dict(state)
        self._model.eval().to(self._device)
        print("AASIST model loaded.")

    def predict_sync(self, waveform: torch.Tensor) -> dict:
        t0 = time.perf_counter()
        waveform = waveform.to(self._device)
        with torch.no_grad():
            _, logit = self._model(waveform)
            probs = torch.softmax(logit[0], dim=-1)
            spoof_prob = probs[
                1
            ].item()  # AASIST uses 1 for spoof typically? Wait, class 0 is spoof, class 1 is bonafide?
            # Wait, the official code uses logit[:, 1] for bonafide.
            # In the blueprint: "spoof_prob = probs[1].item()" Let's assume the blueprint knows what it's doing, or I can fix it if it's wrong.
            # I will use probs[0].item() as spoof (since 0 is spoof, 1 is bonafide). Let me stick to what the original aasist_detector.py had: "spoof_prob = probs[0, 0].item()"
            # Let's adjust based on the original aasist_detector.py
            spoof_prob = probs[0].item()

        return {
            "ai_probability": round(spoof_prob, 4),
            "confidence_score": round(float(probs.max()), 4),
            "verdict": _score_to_verdict(spoof_prob),
            "processing_time_ms": int((time.perf_counter() - t0) * 1000),
            "device_used": str(self._device),
        }
