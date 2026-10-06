"""
service.py — Core Inference Service Orchestrator for Image Detection

Sprint 2: Baseline Detection Service (Member 1: Divisha Manak Bohra - 23ESKCA038)
Task 1: Developing an image inference service with a defined input/output interface
"""

import time
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

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
    CorruptedImageError,
    EmptyImageError,
    UnsupportedFormatError,
    decode_image,
    preprocess_image_tensor,
)
from src.image_detector.schemas import (
    BatchImageInferenceResponse,
    BatchItemError,
    BatchSummary,
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

    def predict_batch(
        self,
        images: Sequence[
            Union[
                bytes,
                str,
                Path,
                Image.Image,
                Tuple[str, Union[bytes, str, Path, Image.Image]],
            ]
        ],
        filenames: Optional[List[str]] = None,
        chunk_size: int = 16,
        return_errors: bool = True,
    ) -> BatchImageInferenceResponse:
        """
        Executes high-throughput batch inference across multiple image inputs (Form-2 Sprint 2 Task 2).

        Features:
        - Batched tensor collation into 4D tensor (B, 3, 224, 224).
        - Dynamic chunking (B <= 32) to prevent VRAM / memory saturation.
        - Vectorized neural forward pass and probability distribution calculation.
        - Fault-isolated partial failure handling (malformed images do not abort the entire batch).
        - Granular batch summary metrics and latency telemetry.

        Args:
            images: Sequence of image payloads (bytes, base64 strings, file paths, PIL Images,
                    or (filename, payload) tuples).
            filenames: Optional list of filename identifiers matching the image sequence.
            chunk_size: Maximum batch collation chunk size (capped between 1 and 32).
            return_errors: If True, bad items are captured into errors list; if False, raises on first failure.

        Returns:
            BatchImageInferenceResponse containing per-image results, errors, and batch summary.
        """
        start_time = time.perf_counter()

        # Clamp chunk_size between 1 and 32
        chunk_size = min(max(1, chunk_size), 32)

        total_submitted = len(images)
        if total_submitted == 0:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return BatchImageInferenceResponse(
                results=[],
                errors=[],
                summary=BatchSummary(
                    total_submitted=0,
                    successful_count=0,
                    failed_count=0,
                    real_count=0,
                    ai_generated_count=0,
                    mean_confidence=0.0,
                ),
                batch_latency_ms=round(elapsed_ms, 2),
                avg_latency_per_image_ms=0.0,
                chunk_size_used=chunk_size,
                model_version=self.model_version,
                device=str(self.device),
            )

        # 1. Decode, validate, and collate valid tensors
        valid_items: List[Tuple[int, str, Any, torch.Tensor]] = []
        errors: List[BatchItemError] = []

        for idx, item in enumerate(images):
            # Extract filename and raw payload
            if isinstance(item, tuple) and len(item) == 2:
                name, payload = item
            else:
                payload = item
                if filenames and idx < len(filenames) and filenames[idx]:
                    name = filenames[idx]
                else:
                    name = f"image_{idx + 1:03d}.png"

            try:
                pil_img, metadata = decode_image(payload)
                # unsqueeze=False produces (3, H, W) tensor suitable for torch.stack
                tensor = preprocess_image_tensor(
                    pil_img, target_size=self.target_size, unsqueeze=False
                )
                valid_items.append((idx, name, metadata, tensor))
            except Exception as e:
                if not return_errors:
                    raise
                error_code = "PROCESSING_ERROR"
                if isinstance(e, EmptyImageError):
                    error_code = "EMPTY_PAYLOAD"
                elif isinstance(e, UnsupportedFormatError):
                    error_code = "UNSUPPORTED_FORMAT"
                elif isinstance(e, CorruptedImageError):
                    error_code = "CORRUPTED_IMAGE"
                errors.append(
                    BatchItemError(
                        index=idx,
                        filename=name,
                        error_code=error_code,
                        message=str(e),
                    )
                )

        # 2. Chunked Batched Neural Inference
        results: List[ImageInferenceResponse] = []

        if valid_items:
            for chunk_start in range(0, len(valid_items), chunk_size):
                chunk = valid_items[chunk_start : chunk_start + chunk_size]
                chunk_tensors = torch.stack([item[3] for item in chunk]).to(self.device)

                with torch.inference_mode():
                    logits = self.model(chunk_tensors)
                    probs = F.softmax(logits, dim=1).cpu()

                for i, (orig_idx, name, metadata, _) in enumerate(chunk):
                    p_real = float(probs[i, 0].item())
                    p_ai = float(probs[i, 1].item())

                    if p_ai >= 0.5:
                        verdict = PredictionVerdict.AI_GENERATED
                        is_ai = True
                        confidence = p_ai
                    else:
                        verdict = PredictionVerdict.REAL
                        is_ai = False
                        confidence = p_real

                    results.append(
                        ImageInferenceResponse(
                            filename=name,
                            verdict=verdict,
                            is_ai=is_ai,
                            confidence=round(confidence, 4),
                            probabilities=ClassProbabilities(
                                real=p_real,
                                ai_generated=p_ai,
                            ),
                            image_metadata=metadata,
                            latency_ms=0.0,
                            model_version=self.model_version,
                            device=str(self.device),
                        )
                    )

        # 3. Calculate Batch Aggregates and Telemetry
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        successful_count = len(results)
        failed_count = len(errors)

        avg_latency = (
            round(elapsed_ms / successful_count, 2) if successful_count > 0 else 0.0
        )

        real_count = sum(1 for r in results if r.verdict == PredictionVerdict.REAL)
        ai_count = sum(
            1 for r in results if r.verdict == PredictionVerdict.AI_GENERATED
        )
        mean_conf = (
            round(sum(r.confidence for r in results) / successful_count, 4)
            if successful_count > 0
            else 0.0
        )

        for r in results:
            r.latency_ms = avg_latency

        summary = BatchSummary(
            total_submitted=total_submitted,
            successful_count=successful_count,
            failed_count=failed_count,
            real_count=real_count,
            ai_generated_count=ai_count,
            mean_confidence=mean_conf,
        )

        return BatchImageInferenceResponse(
            results=results,
            errors=errors,
            summary=summary,
            batch_latency_ms=round(elapsed_ms, 2),
            avg_latency_per_image_ms=avg_latency,
            chunk_size_used=chunk_size,
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
