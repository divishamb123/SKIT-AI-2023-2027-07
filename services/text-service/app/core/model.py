import os
from pathlib import Path
import structlog
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from app.schemas import TextDetectionResponse

logger = structlog.get_logger()

DEFAULT_MODEL_NAME = "microsoft/deberta-v3-base"
MODEL_NAME = DEFAULT_MODEL_NAME


class TextDetector:
    def __init__(self, model_path: str | None = None):
        target_path = model_path or os.getenv("MODEL_PATH")

        if target_path and target_path.strip():
            self.model_name = target_path.strip()
            self.model_version = Path(self.model_name).name or self.model_name
            logger.info("loading_custom_model", model_path=self.model_name)
        else:
            self.model_name = DEFAULT_MODEL_NAME
            self.model_version = f"{DEFAULT_MODEL_NAME}-pretrained"
            logger.info("loading_default_model", model_name=DEFAULT_MODEL_NAME)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2
        )
        self.model.eval()
        logger.info("model_loaded", model_version=self.model_version)

    def predict(self, text: str) -> TextDetectionResponse:
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1).squeeze()

            if probabilities.dim() == 0:
                probabilities = probabilities.unsqueeze(0)

            prob_list = probabilities.tolist()

        predicted_class_id = int(torch.argmax(logits, dim=1).item())
        confidence = prob_list[predicted_class_id]

        # Use model's configured id2label if available, otherwise fallback to index 1 -> ai
        if hasattr(self.model.config, "id2label") and self.model.config.id2label:
            raw_label = str(
                self.model.config.id2label.get(predicted_class_id, "")
            ).lower()
            label = "ai" if "ai" in raw_label or raw_label == "1" else "human"
        else:
            label = "ai" if predicted_class_id == 1 else "human"

        return TextDetectionResponse(
            label=label, confidence=confidence, model_version=self.model_version
        )


# Singleton instance to be loaded once at startup
detector = TextDetector()
