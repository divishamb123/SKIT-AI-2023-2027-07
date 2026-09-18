"""
service.py — Core Inference Service Orchestrator for Image Detection

Sprint 2: Baseline Detection Service (Member 1: Divisha Manak Bohra - 23ESKCA038)
Task 1: Developing an image inference service with a defined input/output interface
"""

import time
from pathlib import Path
from typing import Optional, Union

from PIL import Image
import torch
import torch.nn.functional as F

from src.image_detector.model import (
    MODEL_VERSION,
    get_optimal_device,
    load_baseline_model,
)
from src.image_detector.preprocessing import (
    DEFAULT_IMAGE_SIZE,
    SUPPORTED_FORMATS,
    decode_image,
    preprocess_image_tensor,
)
from src.image_detector.schemas import (
    ClassProbabilities,
    ImageInferenceResponse,
    PredictionVerdict,
    ServiceInfo,
)


class ImageDetectionService:
    """
    Production-ready image inference service.

    Orchestrates decoding, preprocessing, tensor conversion, neural forward pass,
    probability calculation, and structured response construction.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        target_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    ):
        if isinstance(device, str):
            device = get_optimal_device(device)
        self.model, self.device = load_baseline_model(checkpoint_path, device)
        self.target_size = target_size
        self.model_version = MODEL_VERSION

    def predict(
        self,
        image_input: Union[bytes, str, Path, Image.Image],
        filename: str = "sample.png",
    ) -> ImageInferenceResponse:
        """
        Executes end-to-end inference on a single image input.

        Args:
            image_input: Raw image bytes, base64 data URI/string, file path, or PIL Image.
            filename: Name identifier of the input image.

        Returns:
            ImageInferenceResponse adhering strictly to the Form-2 defined interface.
        """
        start_time = time.perf_counter()

        # 1. Decode & validate image payload
        pil_image, metadata = decode_image(image_input)

        # 2. Preprocess into normalized 4D tensor (1, 3, H, W)
        tensor = preprocess_image_tensor(
            pil_image, target_size=self.target_size, unsqueeze=True
        ).to(self.device)

        # 3. Model inference pass
        with torch.inference_mode():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).cpu()[0]

        prob_real = float(probs[0].item())
        prob_ai = float(probs[1].item())

        # 4. Determine categorical verdict and decision confidence
        if prob_ai >= 0.5:
            verdict = PredictionVerdict.AI_GENERATED
            is_ai = True
            confidence = prob_ai
        else:
            verdict = PredictionVerdict.REAL
            is_ai = False
            confidence = prob_real

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # 5. Construct standardized response payload
        return ImageInferenceResponse(
            filename=filename,
            verdict=verdict,
            is_ai=is_ai,
            confidence=round(confidence, 4),
            probabilities=ClassProbabilities(
                real=prob_real,
                ai_generated=prob_ai,
            ),
            image_metadata=metadata,
            latency_ms=round(elapsed_ms, 2),
            model_version=self.model_version,
            device=str(self.device),
        )

    def get_service_info(self) -> ServiceInfo:
        """Returns runtime telemetry, model details, and hardware status."""
        return ServiceInfo(
            service_name="Baseline Image Detection Inference Service",
            model_name="BaselineImageClassifier (ResNet-18)",
            model_version=self.model_version,
            device=str(self.device),
            supported_formats=SUPPORTED_FORMATS,
            status="healthy",
        )
