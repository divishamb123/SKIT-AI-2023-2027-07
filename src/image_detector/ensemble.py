"""
ensemble.py — Multi-Model Neural Ensemble Architecture for Image AI Detection

Sprint 3 (Accuracy Optimisation) — Week 1 Deliverable
Member: Divisha Manak Bohra (23ESKCA038)

Combines complementary computer vision deep learning backbones:
1. ResNet-18 (Deep Spatial Residual CNN)
2. MobileNetV3-Small (Inverted Residual with Squeeze-and-Excitation Channel Attention)

Implements:
- Weighted soft-voting probability fusion
- Sub-model consensus agreement and disagreement variance metrics
- Fault-isolated vectorized batch inference (B <= 32)
- Dynamic device acceleration (Apple Silicon Metal 'mps', NVIDIA CUDA, or CPU)
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from src.image_detector.model import (
    NUM_CLASSES,
    BaselineImageClassifier,
    get_optimal_device,
)
from src.image_detector.preprocessing import (
    DEFAULT_IMAGE_SIZE,
    decode_image,
    preprocess_image_tensor,
)
from src.image_detector.schemas import (
    BatchImageInferenceResponse,
    BatchItemError,
    BatchSummary,
    ClassProbabilities,
    EnsembleMetadata,
    ImageInferenceResponse,
    ImageMetadata,
    PredictionVerdict,
    SubModelPrediction,
)

ENSEMBLE_VERSION = "ensemble-resnet18-mobilenetv3-v1.0"
DEFAULT_WEIGHTS = {"resnet18": 0.55, "mobilenet_v3": 0.45}


class MobileNetV3Classifier(nn.Module):
    """
    MobileNetV3-Small based forensic classifier.
    Uses inverted residual bottlenecks with Hard-Swish and Squeeze-and-Excitation attention.
    Complements ResNet-18 by focusing on lightweight, multi-scale channel representations.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout_rate: float = 0.2):
        super().__init__()
        self.backbone = models.mobilenet_v3_small(weights=None)
        in_features = self.backbone.classifier[0].in_features

        # Replace ImageNet classifier head with 2-class forensic head
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.Hardswish(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(1024, num_classes),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Kaiming normal initialization for deterministic baseline behavior."""
        for m in self.backbone.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits (Batch, 2)."""
        return self.backbone(x)


class ImageEnsembleDetector:
    """
    Multi-model ensemble detector for synthetic and AI-generated image forensics.
    Fuses predictions from ResNet-18 and MobileNetV3 via weighted soft-voting.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        device: Optional[Union[str, torch.device]] = None,
        target_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    ):
        if isinstance(device, str):
            device = get_optimal_device(device)
        self.device = device or get_optimal_device()
        self.target_size = target_size
        self.ensemble_version = ENSEMBLE_VERSION

        # Instantiate complementary vision backbones
        self.resnet = BaselineImageClassifier(num_classes=NUM_CLASSES).to(self.device)
        self.mobilenet = MobileNetV3Classifier(num_classes=NUM_CLASSES).to(self.device)

        self.resnet.eval()
        self.mobilenet.eval()

        # Set and normalize ensemble weights
        self.set_weights(weights or DEFAULT_WEIGHTS)

    def set_weights(self, weights: Dict[str, float]) -> None:
        """Sets and normalizes ensemble fusion weights so sum(weights) == 1.0."""
        w_resnet = max(0.0, float(weights.get("resnet18", 0.5)))
        w_mob = max(0.0, float(weights.get("mobilenet_v3", 0.5)))
        total = w_resnet + w_mob
        if total <= 0:
            w_resnet, w_mob = 0.5, 0.5
            total = 1.0
        self.weights = {
            "resnet18": round(w_resnet / total, 4),
            "mobilenet_v3": round(w_mob / total, 4),
        }

    def load_checkpoints(
        self,
        resnet_checkpoint: Optional[Union[str, Path]] = None,
        mobilenet_checkpoint: Optional[Union[str, Path]] = None,
    ) -> None:
        """Loads trained weights for individual ensemble sub-models."""
        if resnet_checkpoint and Path(resnet_checkpoint).is_file():
            ckpt = torch.load(resnet_checkpoint, map_location=self.device)
            state_dict = ckpt.get("state_dict", ckpt)
            self.resnet.load_state_dict(state_dict)
            self.resnet.eval()

        if mobilenet_checkpoint and Path(mobilenet_checkpoint).is_file():
            ckpt = torch.load(mobilenet_checkpoint, map_location=self.device)
            state_dict = ckpt.get("state_dict", ckpt)
            self.mobilenet.load_state_dict(state_dict)
            self.mobilenet.eval()

    def predict(
        self,
        image_input: Union[bytes, str, Path, Image.Image],
        filename: str = "sample.png",
        threshold: float = 0.5,
    ) -> ImageInferenceResponse:
        """
        Executes multi-model ensemble inference on a single image.

        Args:
            image_input: Raw image bytes, Base64 URI, file path, or PIL Image.
            filename: Identifier string for tracking.
            threshold: Decision boundary threshold tau in [0.10, 0.90].

        Returns:
            ImageInferenceResponse with enriched EnsembleMetadata telemetry.
        """
        start_time = time.perf_counter()

        # 1. Decode & standardize image
        pil_image, raw_metadata = decode_image(image_input)

        # 2. Preprocess to normalized 4D tensor (1, 3, 224, 224)
        tensor = preprocess_image_tensor(
            pil_image, target_size=self.target_size, unsqueeze=True
        ).to(self.device)

        # 3. Parallel model forward passes
        with torch.inference_mode():
            t_res0 = time.perf_counter()
            logits_res = self.resnet(tensor)
            probs_res = F.softmax(logits_res, dim=1).cpu()[0]
            lat_res = (time.perf_counter() - t_res0) * 1000.0

            t_mob0 = time.perf_counter()
            logits_mob = self.mobilenet(tensor)
            probs_mob = F.softmax(logits_mob, dim=1).cpu()[0]
            lat_mob = (time.perf_counter() - t_mob0) * 1000.0

        p_res_real = float(probs_res[0].item())
        p_res_ai = float(probs_res[1].item())

        p_mob_real = float(probs_mob[0].item())
        p_mob_ai = float(probs_mob[1].item())

        # 4. Weighted soft-voting probability fusion
        w_res = self.weights["resnet18"]
        w_mob = self.weights["mobilenet_v3"]

        p_fused_ai = (w_res * p_res_ai) + (w_mob * p_mob_ai)
        p_fused_real = 1.0 - p_fused_ai

        # 5. Calculate consensus and disagreement metrics
        consensus_score = max(0.0, min(1.0, 1.0 - abs(p_res_ai - p_mob_ai)))
        disagreement_var = float(((p_res_ai - p_fused_ai) ** 2 + (p_mob_ai - p_fused_ai) ** 2) / 2.0)

        # 6. Categorical verdict based on threshold tau
        if p_fused_ai >= threshold:
            verdict = PredictionVerdict.AI_GENERATED
            is_ai = True
            confidence = p_fused_ai
        else:
            verdict = PredictionVerdict.REAL
            is_ai = False
            confidence = p_fused_real

        total_latency_ms = (time.perf_counter() - start_time) * 1000.0

        submodel_preds = [
            SubModelPrediction(
                model_name="resnet18",
                weight=w_res,
                verdict=PredictionVerdict.AI_GENERATED if p_res_ai >= threshold else PredictionVerdict.REAL,
                confidence=round(p_res_ai if p_res_ai >= threshold else p_res_real, 4),
                probabilities=ClassProbabilities(real=p_res_real, ai_generated=p_res_ai),
                latency_ms=round(lat_res, 2),
            ),
            SubModelPrediction(
                model_name="mobilenet_v3",
                weight=w_mob,
                verdict=PredictionVerdict.AI_GENERATED if p_mob_ai >= threshold else PredictionVerdict.REAL,
                confidence=round(p_mob_ai if p_mob_ai >= threshold else p_mob_real, 4),
                probabilities=ClassProbabilities(real=p_mob_real, ai_generated=p_mob_ai),
                latency_ms=round(lat_mob, 2),
            ),
        ]

        ensemble_meta = EnsembleMetadata(
            ensemble_strategy="weighted_soft_voting",
            submodels=submodel_preds,
            consensus_score=round(consensus_score, 4),
            disagreement_variance=round(disagreement_var, 6),
        )

        return ImageInferenceResponse(
            filename=filename,
            verdict=verdict,
            is_ai=is_ai,
            confidence=round(confidence, 4),
            probabilities=ClassProbabilities(real=p_fused_real, ai_generated=p_fused_ai),
            image_metadata=raw_metadata,
            latency_ms=round(total_latency_ms, 2),
            model_version=self.ensemble_version,
            device=str(self.device),
            ensemble_metadata=ensemble_meta,
        )

    def predict_batch(
        self,
        image_inputs: Sequence[Union[bytes, str, Path, Image.Image]],
        filenames: Optional[Sequence[str]] = None,
        chunk_size: int = 32,
        threshold: float = 0.5,
    ) -> BatchImageInferenceResponse:
        """
        Executes high-throughput batch inference across the multi-model ensemble.
        Utilizes vectorized tensor collation and dynamic chunking (chunk_size <= 32).
        """
        batch_start_time = time.perf_counter()
        total_items = len(image_inputs)

        resolved_filenames = [
            filenames[i] if filenames and i < len(filenames) else f"item_{i}.png"
            for i in range(total_items)
        ]

        if total_items == 0:
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
                batch_latency_ms=0.0,
                avg_latency_per_image_ms=0.0,
                chunk_size_used=chunk_size,
                model_version=self.ensemble_version,
                device=str(self.device),
            )

        # 1. Decode & preprocess all valid images (unsqueeze=False for stacking)
        valid_items: List[Tuple[int, str, torch.Tensor, ImageMetadata]] = []
        errors: List[BatchItemError] = []

        for idx, (inp, fn) in enumerate(zip(image_inputs, resolved_filenames)):
            try:
                pil_img, meta = decode_image(inp)
                tensor = preprocess_image_tensor(
                    pil_img, target_size=self.target_size, unsqueeze=False
                )
                valid_items.append((idx, fn, tensor, meta))
            except Exception as exc:
                errors.append(
                    BatchItemError(
                        index=idx,
                        filename=fn,
                        error_code=type(exc).__name__,
                        message=str(exc),
                    )
                )

        results: List[ImageInferenceResponse] = []
        w_res = self.weights["resnet18"]
        w_mob = self.weights["mobilenet_v3"]

        # 2. Dynamic chunking over valid tensors
        for chunk_start in range(0, len(valid_items), chunk_size):
            chunk = valid_items[chunk_start : chunk_start + chunk_size]
            batch_tensor = torch.stack([item[2] for item in chunk], dim=0).to(self.device)

            chunk_t0 = time.perf_counter()
            with torch.inference_mode():
                # Parallel forward passes for entire batch tensor
                logits_res = self.resnet(batch_tensor)
                probs_res = F.softmax(logits_res, dim=1).cpu()

                logits_mob = self.mobilenet(batch_tensor)
                probs_mob = F.softmax(logits_mob, dim=1).cpu()

            chunk_lat_ms = (time.perf_counter() - chunk_t0) * 1000.0
            avg_item_lat = chunk_lat_ms / len(chunk)

            # Collate individual predictions
            for sub_idx, (orig_idx, fn, _, meta) in enumerate(chunk):
                p_res_real = float(probs_res[sub_idx, 0].item())
                p_res_ai = float(probs_res[sub_idx, 1].item())

                p_mob_real = float(probs_mob[sub_idx, 0].item())
                p_mob_ai = float(probs_mob[sub_idx, 1].item())

                p_fused_ai = (w_res * p_res_ai) + (w_mob * p_mob_ai)
                p_fused_real = 1.0 - p_fused_ai

                consensus_score = max(0.0, min(1.0, 1.0 - abs(p_res_ai - p_mob_ai)))
                disagreement_var = float(
                    ((p_res_ai - p_fused_ai) ** 2 + (p_mob_ai - p_fused_ai) ** 2) / 2.0
                )

                if p_fused_ai >= threshold:
                    verdict = PredictionVerdict.AI_GENERATED
                    is_ai = True
                    confidence = p_fused_ai
                else:
                    verdict = PredictionVerdict.REAL
                    is_ai = False
                    confidence = p_fused_real

                submodel_preds = [
                    SubModelPrediction(
                        model_name="resnet18",
                        weight=w_res,
                        verdict=PredictionVerdict.AI_GENERATED if p_res_ai >= threshold else PredictionVerdict.REAL,
                        confidence=round(p_res_ai if p_res_ai >= threshold else p_res_real, 4),
                        probabilities=ClassProbabilities(real=p_res_real, ai_generated=p_res_ai),
                    ),
                    SubModelPrediction(
                        model_name="mobilenet_v3",
                        weight=w_mob,
                        verdict=PredictionVerdict.AI_GENERATED if p_mob_ai >= threshold else PredictionVerdict.REAL,
                        confidence=round(p_mob_ai if p_mob_ai >= threshold else p_mob_real, 4),
                        probabilities=ClassProbabilities(real=p_mob_real, ai_generated=p_mob_ai),
                    ),
                ]

                results.append(
                    ImageInferenceResponse(
                        filename=fn,
                        verdict=verdict,
                        is_ai=is_ai,
                        confidence=round(confidence, 4),
                        probabilities=ClassProbabilities(real=p_fused_real, ai_generated=p_fused_ai),
                        image_metadata=meta,
                        latency_ms=round(avg_item_lat, 2),
                        model_version=self.ensemble_version,
                        device=str(self.device),
                        ensemble_metadata=EnsembleMetadata(
                            ensemble_strategy="weighted_soft_voting",
                            submodels=submodel_preds,
                            consensus_score=round(consensus_score, 4),
                            disagreement_variance=round(disagreement_var, 6),
                        ),
                    )
                )

        total_latency_ms = (time.perf_counter() - batch_start_time) * 1000.0
        successful_count = len(results)
        failed_count = len(errors)
        avg_latency = round(total_latency_ms / successful_count, 2) if successful_count > 0 else 0.0

        real_count = sum(1 for r in results if r.verdict == PredictionVerdict.REAL)
        ai_count = sum(1 for r in results if r.verdict == PredictionVerdict.AI_GENERATED)
        mean_conf = (
            round(sum(r.confidence for r in results) / successful_count, 4)
            if successful_count > 0
            else 0.0
        )

        summary = BatchSummary(
            total_submitted=total_items,
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
            batch_latency_ms=round(total_latency_ms, 2),
            avg_latency_per_image_ms=avg_latency,
            chunk_size_used=chunk_size,
            model_version=self.ensemble_version,
            device=str(self.device),
        )
