import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from app.schemas import TextDetectionResponse
import structlog

logger = structlog.get_logger()

MODEL_NAME = "microsoft/deberta-v3-base"

class TextDetector:
    def __init__(self):
        logger.info("loading_model", model_name=MODEL_NAME)
        # Using pretrained DeBERTa-v3-base for Sprint 2 baseline (no fine-tuning yet)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
        self.model.eval()
        logger.info("model_loaded")
        
    def predict(self, text: str) -> TextDetectionResponse:
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1).squeeze()
            
            if probabilities.dim() == 0:
                probabilities = probabilities.unsqueeze(0)
            
            prob_list = probabilities.tolist()
            
        # By default, a totally untrained DeBERTa model won't output meaningful binary classifications
        # for our specific AI/Human classes. We'll map argmax(0) -> human, argmax(1) -> ai for the baseline.
        predicted_class_id = int(torch.argmax(logits, dim=1).item())
        confidence = prob_list[predicted_class_id]
        
        label = "ai" if predicted_class_id == 1 else "human"
        
        return TextDetectionResponse(
            label=label,
            confidence=confidence,
            model_version=f"{MODEL_NAME}-pretrained"
        )

# Singleton instance to be loaded once at startup
detector = TextDetector()
