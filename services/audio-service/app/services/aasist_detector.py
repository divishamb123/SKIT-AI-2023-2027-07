import torch


class AASISTDetector:
    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = torch.device(device)

    def detect(self, waveform: torch.Tensor) -> dict:
        """
        waveform: torch.Tensor of shape (1, 64600)
        Returns: dict with detection result
        """
        # Ensure correct shape
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)

        waveform = waveform.to(self.device)

        with torch.no_grad():
            last_hidden, logits = self.model(waveform)

            # The model outputs a logit tensor of shape (batch, 2)
            # AASIST outputs class 0 for spoof and class 1 for bonafide
            # Let's verify this from main.py
            # wait, main.py says: batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
            # If batch_score is used for EER, higher score = bonafide.
            # So class 0 is spoof, class 1 is bonafide.

            probs = torch.softmax(logits, dim=-1)

            spoof_prob = probs[0, 0].item()
            bonafide_prob = probs[0, 1].item()

            # According to the Engineering Blueprint from the previous step:
            # AI probability (spoof probability) should be the primary score returned

        return {
            "ai_probability": float(spoof_prob),
            "bonafide_probability": float(bonafide_prob),
            "verdict": "Unknown",  # Threshold mapping will be applied later
            "model": "AASIST-L",
            "raw_logits": logits.cpu().numpy().tolist(),
        }
