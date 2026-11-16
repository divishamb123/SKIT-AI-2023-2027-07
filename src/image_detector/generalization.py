"""
generalization.py — Cross-Generator Generalization Testing Engine

Sprint 3 (Accuracy Optimisation) — Week 2 Deliverable
Member: Divisha Manak Bohra (23ESKCA038)

Tests ensemble detection performance on unseen image generators to measure
out-of-domain robustness and identify generalization gaps.

In-Domain Generators (training distribution): cifar10, stable-diffusion-1.4
Out-of-Domain / Unseen Generators: real, stable-diffusion, midjourney, biggan

Implements:
- Per-generator binary classification metrics (accuracy, precision, recall, F1)
- Domain-level aggregated evaluation (in-domain vs out-of-domain)
- Generalization gap computation (Δ = in-domain accuracy − out-of-domain accuracy)
- Per-generator robustness scoring
- Structured Pydantic report with JSON/Markdown export
"""

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import uuid

from PIL import Image
from pydantic import BaseModel, Field

from src.image_detector.ensemble import (
    DEFAULT_WEIGHTS,
    ImageEnsembleDetector,
)
from src.image_detector.schemas import (
    ConfusionMatrixMetrics,
    PredictionVerdict,
)

# --------------------------------------------------------------------------
# Constants — Domain Classification
# --------------------------------------------------------------------------
IN_DOMAIN_GENERATORS: Set[str] = {"cifar10", "stable-diffusion-1.4"}
OUT_OF_DOMAIN_GENERATORS: Set[str] = {
    "real",
    "stable-diffusion",
    "midjourney",
    "biggan",
}
ALL_KNOWN_GENERATORS: Set[str] = IN_DOMAIN_GENERATORS | OUT_OF_DOMAIN_GENERATORS


# --------------------------------------------------------------------------
# Pydantic Schemas — Generalization Report
# --------------------------------------------------------------------------
class GeneratorMetrics(BaseModel):
    """Evaluation metrics for a single image generator source."""

    generator_name: str = Field(
        ..., description="Identifier of the image generator (e.g. midjourney, biggan)"
    )
    domain: str = Field(
        ..., description="Domain classification: 'in_domain' or 'out_of_domain'"
    )
    sample_count: int = Field(
        ..., ge=0, description="Total samples evaluated for this generator"
    )
    accuracy: float = Field(
        ..., ge=0.0, le=1.0, description="Binary classification accuracy"
    )
    precision: float = Field(
        ..., ge=0.0, le=1.0, description="Precision for AI-generated class"
    )
    recall: float = Field(
        ..., ge=0.0, le=1.0, description="Recall / sensitivity for AI-generated class"
    )
    specificity: float = Field(
        ..., ge=0.0, le=1.0, description="Specificity / true negative rate"
    )
    f1_score: float = Field(
        ..., ge=0.0, le=1.0, description="Harmonic mean of precision and recall"
    )
    mean_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Mean prediction confidence across samples"
    )
    robustness_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Robustness score: 1.0 − |0.5 − accuracy|, higher indicates stronger generalisation",
    )
    confusion_matrix: ConfusionMatrixMetrics = Field(
        ..., description="Raw count confusion matrix (TP, FP, TN, FN)"
    )
    latency_ms: float = Field(
        ..., ge=0.0, description="Total inference latency for this generator's samples"
    )
    throughput_fps: float = Field(
        ..., ge=0.0, description="Throughput in frames per second"
    )


class DomainMetrics(BaseModel):
    """Aggregated metrics across all generators within a domain."""

    domain: str = Field(..., description="Domain label: 'in_domain' or 'out_of_domain'")
    generators: List[str] = Field(
        ..., description="List of generator names in this domain"
    )
    total_samples: int = Field(
        ..., ge=0, description="Total samples across all generators in this domain"
    )
    accuracy: float = Field(
        ..., ge=0.0, le=1.0, description="Aggregate accuracy across the domain"
    )
    precision: float = Field(..., ge=0.0, le=1.0, description="Aggregate precision")
    recall: float = Field(..., ge=0.0, le=1.0, description="Aggregate recall")
    specificity: float = Field(..., ge=0.0, le=1.0, description="Aggregate specificity")
    f1_score: float = Field(..., ge=0.0, le=1.0, description="Aggregate F1 score")
    mean_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Mean confidence across the domain"
    )
    mean_robustness: float = Field(
        ..., ge=0.0, le=1.0, description="Mean robustness score across generators"
    )
    confusion_matrix: ConfusionMatrixMetrics = Field(
        ..., description="Aggregate confusion matrix across the domain"
    )


class GeneralizationGap(BaseModel):
    """Quantified gap between in-domain and out-of-domain performance."""

    in_domain_accuracy: float = Field(
        ..., ge=0.0, le=1.0, description="Accuracy on in-domain test generators"
    )
    out_of_domain_accuracy: float = Field(
        ..., ge=0.0, le=1.0, description="Accuracy on unseen out-of-domain generators"
    )
    accuracy_gap: float = Field(
        ...,
        description="Δ = in_domain_accuracy − out_of_domain_accuracy (positive = drop on unseen)",
    )
    f1_gap: float = Field(..., description="Δ = in_domain_F1 − out_of_domain_F1")
    precision_gap: float = Field(
        ..., description="Δ = in_domain_precision − out_of_domain_precision"
    )
    recall_gap: float = Field(
        ..., description="Δ = in_domain_recall − out_of_domain_recall"
    )
    severity: str = Field(
        ...,
        description="Qualitative severity: 'minimal' (|Δ|<0.05), 'moderate' (0.05≤|Δ|<0.15), 'significant' (|Δ|≥0.15)",
    )


class GeneralizationReport(BaseModel):
    """Comprehensive cross-generator generalization evaluation report."""

    report_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique UUID for this evaluation report",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 UTC timestamp of evaluation",
    )
    ensemble_version: str = Field(..., description="Ensemble model version identifier")
    device: str = Field(..., description="Hardware device used for inference")
    threshold: float = Field(
        ..., ge=0.0, le=1.0, description="Decision boundary threshold τ used"
    )
    total_samples_evaluated: int = Field(
        ..., ge=0, description="Total number of samples evaluated across all generators"
    )
    overall_accuracy: float = Field(
        ..., ge=0.0, le=1.0, description="Aggregate accuracy across all generators"
    )
    overall_f1: float = Field(
        ..., ge=0.0, le=1.0, description="Aggregate F1 across all generators"
    )
    per_generator: List[GeneratorMetrics] = Field(
        ..., description="Per-generator evaluation breakdowns"
    )
    in_domain: DomainMetrics = Field(
        ..., description="Aggregated in-domain performance metrics"
    )
    out_of_domain: DomainMetrics = Field(
        ..., description="Aggregated out-of-domain performance metrics"
    )
    generalization_gap: GeneralizationGap = Field(
        ...,
        description="Quantified gap between in-domain and out-of-domain performance",
    )
    total_latency_ms: float = Field(
        ..., ge=0.0, description="Total evaluation latency in milliseconds"
    )
    notes: Optional[str] = Field(None, description="Qualitative analysis notes")


# --------------------------------------------------------------------------
# Core Evaluator
# --------------------------------------------------------------------------
class GeneralizationEvaluator:
    """
    Cross-generator generalization testing engine for the image ensemble detector.

    Evaluates how well the ensemble generalises from in-domain generators (seen
    during training) to out-of-domain / unseen generators, computing per-generator
    metrics, domain-level aggregates, and the generalization gap.
    """

    def __init__(
        self,
        detector: Optional[ImageEnsembleDetector] = None,
        in_domain_generators: Optional[Set[str]] = None,
        out_of_domain_generators: Optional[Set[str]] = None,
        threshold: float = 0.5,
    ):
        self.detector = detector or ImageEnsembleDetector(weights=DEFAULT_WEIGHTS)
        self.in_domain_generators = in_domain_generators or IN_DOMAIN_GENERATORS.copy()
        self.out_of_domain_generators = (
            out_of_domain_generators or OUT_OF_DOMAIN_GENERATORS.copy()
        )
        self.threshold = threshold

    def classify_domain(self, generator: str) -> str:
        """Classifies a generator as 'in_domain' or 'out_of_domain'."""
        if generator in self.in_domain_generators:
            return "in_domain"
        return "out_of_domain"

    # ------------------------------------------------------------------
    # Synthetic Sample Generation (for unit testing without real files)
    # ------------------------------------------------------------------
    @staticmethod
    def create_synthetic_samples(
        generators: Optional[Sequence[str]] = None,
        samples_per_class_per_generator: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Creates synthetic test samples with balanced classes per generator.

        Each sample is a dict with keys: image, label, generator, dataset, split, filename.
        The 'image' key holds a PIL Image object (no file on disk needed).
        Label 0 = REAL, Label 1 = AI_GENERATED.

        Args:
            generators: List of generator names. Defaults to all known generators.
            samples_per_class_per_generator: Number of samples per class per generator.

        Returns:
            List of sample dictionaries ready for evaluation.
        """
        if generators is None:
            generators = sorted(ALL_KNOWN_GENERATORS)

        color_map: Dict[str, Tuple[int, int, int]] = {
            "cifar10": (200, 50, 50),
            "stable-diffusion-1.4": (50, 200, 50),
            "real": (50, 50, 200),
            "stable-diffusion": (200, 200, 50),
            "midjourney": (200, 50, 200),
            "biggan": (50, 200, 200),
        }

        samples: List[Dict[str, Any]] = []
        for gen in generators:
            base_color = color_map.get(gen, (128, 128, 128))
            domain = "cifake" if gen in IN_DOMAIN_GENERATORS else "genimage"
            split = "test" if gen in IN_DOMAIN_GENERATORS else "holdout"

            for label in (0, 1):
                for i in range(samples_per_class_per_generator):
                    # Vary color slightly per sample for diversity
                    r = min(255, base_color[0] + i * 5 + label * 30)
                    g = min(255, base_color[1] + i * 3 + label * 20)
                    b = min(255, base_color[2] + i * 7 + label * 10)
                    img = Image.new("RGB", (224, 224), color=(r, g, b))

                    fname = f"synth_{gen}_label{label}_{i:03d}.png"
                    samples.append(
                        {
                            "image": img,
                            "label": label,
                            "generator": gen,
                            "dataset": domain,
                            "split": split,
                            "filename": fname,
                        }
                    )
        return samples

    # ------------------------------------------------------------------
    # Manifest Loading (for real dataset evaluation)
    # ------------------------------------------------------------------
    @staticmethod
    def load_manifest_samples(
        manifest_path: Union[str, Path],
        split: Optional[str] = None,
        generators: Optional[Sequence[str]] = None,
        limit_per_generator: Optional[int] = None,
        project_root: Optional[Union[str, Path]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Loads evaluation samples from manifest.csv with optional filtering.

        Args:
            manifest_path: Path to manifest.csv.
            split: Filter by split (e.g. 'holdout', 'test').
            generators: Filter to specific generators only.
            limit_per_generator: Maximum samples per generator.
            project_root: Root directory for resolving relative paths.

        Returns:
            List of sample dictionaries.
        """
        manifest_file = Path(manifest_path)
        if not manifest_file.is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest_file}")

        root_dir = (
            Path(project_root) if project_root else manifest_file.parent.parent.parent
        )

        rows: List[Dict[str, Any]] = []
        gen_filter = set(generators) if generators else None

        with open(manifest_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if split and row.get("split") != split:
                    continue
                gen = row.get("generator", "unknown")
                if gen_filter and gen not in gen_filter:
                    continue

                raw_path = row["path"].strip()
                abs_path = (
                    (root_dir / raw_path).resolve()
                    if not Path(raw_path).is_absolute()
                    else Path(raw_path)
                )

                rows.append(
                    {
                        "dataset": row.get("dataset", "unknown"),
                        "split": row.get("split", "unknown"),
                        "label": int(row["label"]),
                        "generator": gen,
                        "path": str(abs_path),
                        "filename": abs_path.name,
                    }
                )

        if limit_per_generator is not None:
            stratified: List[Dict[str, Any]] = []
            grouped: Dict[str, List[Dict[str, Any]]] = {}
            for r in rows:
                grouped.setdefault(r["generator"], []).append(r)
            for gen_samples in grouped.values():
                stratified.extend(gen_samples[:limit_per_generator])
            return stratified

        return rows

    # ------------------------------------------------------------------
    # Per-Generator Evaluation
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_binary_metrics(
        y_true: Sequence[int], y_pred: Sequence[int], confidences: Sequence[float]
    ) -> Dict[str, Any]:
        """Computes binary classification metrics from ground truth and predictions."""
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
        tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

        total = len(y_true)
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        mean_conf = sum(confidences) / len(confidences) if len(confidences) > 0 else 0.0

        return {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "specificity": round(specificity, 4),
            "f1_score": round(f1, 4),
            "mean_confidence": round(mean_conf, 4),
            "confusion_matrix": ConfusionMatrixMetrics(tp=tp, fp=fp, tn=tn, fn=fn),
            "total": total,
        }

    def evaluate_generator(
        self,
        samples: Sequence[Dict[str, Any]],
        generator_name: str,
    ) -> GeneratorMetrics:
        """
        Evaluates ensemble detection on samples from a single generator.

        Each sample dict must have 'label' (0=real, 1=AI) and either 'image' (PIL)
        or 'path' (filesystem path to image file).

        Returns:
            GeneratorMetrics with full binary classification telemetry.
        """
        if not samples:
            return GeneratorMetrics(
                generator_name=generator_name,
                domain=self.classify_domain(generator_name),
                sample_count=0,
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                specificity=0.0,
                f1_score=0.0,
                mean_confidence=0.0,
                robustness_score=0.5,
                confusion_matrix=ConfusionMatrixMetrics(tp=0, fp=0, tn=0, fn=0),
                latency_ms=0.0,
                throughput_fps=0.0,
            )

        y_true: List[int] = []
        y_pred: List[int] = []
        confidences: List[float] = []

        start_time = time.perf_counter()

        for sample in samples:
            # Get image input: either a PIL Image object or a file path
            image_input = sample.get("image") or sample.get("path")
            filename = sample.get("filename", "unknown.png")

            try:
                response = self.detector.predict(
                    image_input, filename=filename, threshold=self.threshold
                )
                y_true.append(sample["label"])
                pred_label = (
                    1 if response.verdict == PredictionVerdict.AI_GENERATED else 0
                )
                y_pred.append(pred_label)
                confidences.append(response.confidence)
            except Exception:
                # Skip samples that fail to process (e.g. missing files)
                continue

        total_latency_ms = (time.perf_counter() - start_time) * 1000.0
        evaluated_count = len(y_true)

        if evaluated_count == 0:
            return GeneratorMetrics(
                generator_name=generator_name,
                domain=self.classify_domain(generator_name),
                sample_count=len(samples),
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                specificity=0.0,
                f1_score=0.0,
                mean_confidence=0.0,
                robustness_score=0.5,
                confusion_matrix=ConfusionMatrixMetrics(tp=0, fp=0, tn=0, fn=0),
                latency_ms=round(total_latency_ms, 2),
                throughput_fps=0.0,
            )

        metrics = self._compute_binary_metrics(y_true, y_pred, confidences)
        throughput = (
            (evaluated_count / (total_latency_ms / 1000.0))
            if total_latency_ms > 0
            else 0.0
        )

        # Robustness score: 1.0 - |0.5 - accuracy| — higher is better
        # At accuracy=1.0 or accuracy=0.0, robustness=0.5; at accuracy=0.5 (random), robustness=1.0
        # Inverted so accuracy closer to 1.0 is better:
        robustness = round(1.0 - abs(0.5 - metrics["accuracy"]), 4)

        return GeneratorMetrics(
            generator_name=generator_name,
            domain=self.classify_domain(generator_name),
            sample_count=evaluated_count,
            accuracy=metrics["accuracy"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            specificity=metrics["specificity"],
            f1_score=metrics["f1_score"],
            mean_confidence=metrics["mean_confidence"],
            robustness_score=robustness,
            confusion_matrix=metrics["confusion_matrix"],
            latency_ms=round(total_latency_ms, 2),
            throughput_fps=round(throughput, 2),
        )

    # ------------------------------------------------------------------
    # Domain-Level Aggregation
    # ------------------------------------------------------------------
    def _aggregate_domain_metrics(
        self,
        generator_metrics: Sequence[GeneratorMetrics],
        domain_label: str,
    ) -> DomainMetrics:
        """Aggregates per-generator metrics into domain-level metrics."""
        if not generator_metrics:
            return DomainMetrics(
                domain=domain_label,
                generators=[],
                total_samples=0,
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                specificity=0.0,
                f1_score=0.0,
                mean_confidence=0.0,
                mean_robustness=0.0,
                confusion_matrix=ConfusionMatrixMetrics(tp=0, fp=0, tn=0, fn=0),
            )

        generators = [m.generator_name for m in generator_metrics]
        total_samples = sum(m.sample_count for m in generator_metrics)

        # Aggregate confusion matrix by summing raw counts
        total_tp = sum(m.confusion_matrix.tp for m in generator_metrics)
        total_fp = sum(m.confusion_matrix.fp for m in generator_metrics)
        total_tn = sum(m.confusion_matrix.tn for m in generator_metrics)
        total_fn = sum(m.confusion_matrix.fn for m in generator_metrics)

        total_evaluated = total_tp + total_fp + total_tn + total_fn
        accuracy = (
            (total_tp + total_tn) / total_evaluated if total_evaluated > 0 else 0.0
        )
        precision = (
            total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        )
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        specificity = (
            total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0.0
        )
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        mean_conf = (
            sum(m.mean_confidence * m.sample_count for m in generator_metrics)
            / total_samples
            if total_samples > 0
            else 0.0
        )
        mean_robustness = sum(m.robustness_score for m in generator_metrics) / len(
            generator_metrics
        )

        return DomainMetrics(
            domain=domain_label,
            generators=generators,
            total_samples=total_samples,
            accuracy=round(accuracy, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            specificity=round(specificity, 4),
            f1_score=round(f1, 4),
            mean_confidence=round(mean_conf, 4),
            mean_robustness=round(mean_robustness, 4),
            confusion_matrix=ConfusionMatrixMetrics(
                tp=total_tp, fp=total_fp, tn=total_tn, fn=total_fn
            ),
        )

    # ------------------------------------------------------------------
    # Generalization Gap
    # ------------------------------------------------------------------
    @staticmethod
    def compute_generalization_gap(
        in_domain: DomainMetrics, out_of_domain: DomainMetrics
    ) -> GeneralizationGap:
        """Computes the generalization gap between in-domain and out-of-domain performance."""
        accuracy_gap = round(in_domain.accuracy - out_of_domain.accuracy, 4)
        f1_gap = round(in_domain.f1_score - out_of_domain.f1_score, 4)
        precision_gap = round(in_domain.precision - out_of_domain.precision, 4)
        recall_gap = round(in_domain.recall - out_of_domain.recall, 4)

        abs_gap = abs(accuracy_gap)
        if abs_gap < 0.05:
            severity = "minimal"
        elif abs_gap < 0.15:
            severity = "moderate"
        else:
            severity = "significant"

        return GeneralizationGap(
            in_domain_accuracy=in_domain.accuracy,
            out_of_domain_accuracy=out_of_domain.accuracy,
            accuracy_gap=accuracy_gap,
            f1_gap=f1_gap,
            precision_gap=precision_gap,
            recall_gap=recall_gap,
            severity=severity,
        )

    # ------------------------------------------------------------------
    # Full Generalization Evaluation
    # ------------------------------------------------------------------
    def evaluate_generalization(
        self,
        samples: Sequence[Dict[str, Any]],
        notes: Optional[str] = None,
    ) -> GeneralizationReport:
        """
        Executes comprehensive cross-generator generalization evaluation.

        Groups samples by generator, evaluates each independently, aggregates
        into domain-level metrics, and computes the generalization gap.

        Args:
            samples: List of sample dicts with keys: label, generator, and image/path.
            notes: Optional qualitative analysis notes.

        Returns:
            GeneralizationReport with full evaluation telemetry.
        """
        eval_start = time.perf_counter()

        # Group samples by generator
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for sample in samples:
            gen = sample.get("generator", "unknown")
            grouped.setdefault(gen, []).append(sample)

        # Evaluate each generator
        per_generator: List[GeneratorMetrics] = []
        for gen_name in sorted(grouped.keys()):
            gen_samples = grouped[gen_name]
            gen_metrics = self.evaluate_generator(gen_samples, gen_name)
            per_generator.append(gen_metrics)

        # Split into domain groups
        in_domain_metrics = [m for m in per_generator if m.domain == "in_domain"]
        out_of_domain_metrics = [
            m for m in per_generator if m.domain == "out_of_domain"
        ]

        # Aggregate domain-level metrics
        in_domain = self._aggregate_domain_metrics(in_domain_metrics, "in_domain")
        out_of_domain = self._aggregate_domain_metrics(
            out_of_domain_metrics, "out_of_domain"
        )

        # Compute generalization gap
        gap = self.compute_generalization_gap(in_domain, out_of_domain)

        # Overall aggregate
        all_metrics = in_domain_metrics + out_of_domain_metrics
        total_samples = sum(m.sample_count for m in all_metrics)
        total_tp = sum(m.confusion_matrix.tp for m in all_metrics)
        total_fp = sum(m.confusion_matrix.fp for m in all_metrics)
        total_tn = sum(m.confusion_matrix.tn for m in all_metrics)
        total_fn = sum(m.confusion_matrix.fn for m in all_metrics)
        total_evaluated = total_tp + total_fp + total_tn + total_fn

        overall_accuracy = (
            (total_tp + total_tn) / total_evaluated if total_evaluated > 0 else 0.0
        )
        overall_precision = (
            total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        )
        overall_recall = (
            total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        )
        overall_f1 = (
            2
            * (overall_precision * overall_recall)
            / (overall_precision + overall_recall)
            if (overall_precision + overall_recall) > 0
            else 0.0
        )

        total_latency_ms = (time.perf_counter() - eval_start) * 1000.0

        return GeneralizationReport(
            ensemble_version=self.detector.ensemble_version,
            device=str(self.detector.device),
            threshold=self.threshold,
            total_samples_evaluated=total_samples,
            overall_accuracy=round(overall_accuracy, 4),
            overall_f1=round(overall_f1, 4),
            per_generator=per_generator,
            in_domain=in_domain,
            out_of_domain=out_of_domain,
            generalization_gap=gap,
            total_latency_ms=round(total_latency_ms, 2),
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Report Formatting & Export
    # ------------------------------------------------------------------
    @staticmethod
    def format_markdown_report(report: GeneralizationReport) -> str:
        """Generates a structured Markdown evaluation report for cross-generator generalization."""
        lines = [
            "# Cross-Generator Generalization Evaluation Report",
            "",
            "**Sprint:** Sprint 3 (Accuracy Optimisation) — Week 2 Deliverable  ",
            "**Member:** Divisha Manak Bohra (`23ESKCA038`) — Image Detection Modality & Image Interface  ",
            "**Project ID:** `SKIT/AI/2023-2027/07`  ",
            f"**Ensemble Version:** `{report.ensemble_version}`  ",
            f"**Device:** `{report.device}`  ",
            f"**Decision Threshold (τ):** `{report.threshold}`  ",
            f"**Timestamp:** `{report.timestamp}`  ",
            "",
            "---",
            "",
            "## 1. Executive Summary",
            "",
            "This report evaluates the multi-model ensemble detector's ability to generalise "
            "from in-domain generators (seen during training) to out-of-domain / unseen generators. "
            "A smaller generalization gap indicates stronger cross-domain robustness.",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Total Samples Evaluated | **{report.total_samples_evaluated:,}** |",
            f"| Overall Accuracy | **{report.overall_accuracy * 100:.2f}%** |",
            f"| Overall F1 Score | **{report.overall_f1:.4f}** |",
            f"| In-Domain Accuracy | **{report.in_domain.accuracy * 100:.2f}%** |",
            f"| Out-of-Domain Accuracy | **{report.out_of_domain.accuracy * 100:.2f}%** |",
            f"| **Generalization Gap (Δ)** | **{report.generalization_gap.accuracy_gap * 100:+.2f}%** |",
            f"| Gap Severity | **{report.generalization_gap.severity.upper()}** |",
            f"| Total Latency | **{report.total_latency_ms:.1f} ms** |",
            "",
            "---",
            "",
            "## 2. Per-Generator Breakdown",
            "",
            "| Generator | Domain | Samples | Accuracy | Precision | Recall | F1 | Robustness | Confidence | Throughput |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]

        for gm in report.per_generator:
            lines.append(
                f"| **{gm.generator_name}** | {gm.domain} | {gm.sample_count} | "
                f"{gm.accuracy * 100:.1f}% | {gm.precision * 100:.1f}% | "
                f"{gm.recall * 100:.1f}% | {gm.f1_score:.3f} | "
                f"{gm.robustness_score:.3f} | {gm.mean_confidence:.3f} | "
                f"{gm.throughput_fps:.1f} FPS |"
            )

        lines.extend(
            [
                "",
                "---",
                "",
                "## 3. Domain-Level Analysis",
                "",
                "### 3.1 In-Domain Performance",
                "",
                f"**Generators:** {', '.join(report.in_domain.generators)}  ",
                f"**Total Samples:** {report.in_domain.total_samples}  ",
                "",
                "| Metric | Value |",
                "|---|---|",
                f"| Accuracy | {report.in_domain.accuracy * 100:.2f}% |",
                f"| Precision | {report.in_domain.precision * 100:.2f}% |",
                f"| Recall | {report.in_domain.recall * 100:.2f}% |",
                f"| Specificity | {report.in_domain.specificity * 100:.2f}% |",
                f"| F1 Score | {report.in_domain.f1_score:.4f} |",
                f"| Mean Confidence | {report.in_domain.mean_confidence:.4f} |",
                f"| Mean Robustness | {report.in_domain.mean_robustness:.4f} |",
                "",
                "### 3.2 Out-of-Domain / Unseen Generator Performance",
                "",
                f"**Generators:** {', '.join(report.out_of_domain.generators)}  ",
                f"**Total Samples:** {report.out_of_domain.total_samples}  ",
                "",
                "| Metric | Value |",
                "|---|---|",
                f"| Accuracy | {report.out_of_domain.accuracy * 100:.2f}% |",
                f"| Precision | {report.out_of_domain.precision * 100:.2f}% |",
                f"| Recall | {report.out_of_domain.recall * 100:.2f}% |",
                f"| Specificity | {report.out_of_domain.specificity * 100:.2f}% |",
                f"| F1 Score | {report.out_of_domain.f1_score:.4f} |",
                f"| Mean Confidence | {report.out_of_domain.mean_confidence:.4f} |",
                f"| Mean Robustness | {report.out_of_domain.mean_robustness:.4f} |",
                "",
                "---",
                "",
                "## 4. Generalization Gap Analysis",
                "",
                "| Metric | In-Domain | Out-of-Domain | Gap (Δ) |",
                "|---|---|---|---|",
                f"| Accuracy | {report.generalization_gap.in_domain_accuracy * 100:.2f}% | "
                f"{report.generalization_gap.out_of_domain_accuracy * 100:.2f}% | "
                f"{report.generalization_gap.accuracy_gap * 100:+.2f}% |",
                f"| F1 Score | {report.in_domain.f1_score:.4f} | "
                f"{report.out_of_domain.f1_score:.4f} | "
                f"{report.generalization_gap.f1_gap:+.4f} |",
                f"| Precision | {report.in_domain.precision * 100:.2f}% | "
                f"{report.out_of_domain.precision * 100:.2f}% | "
                f"{report.generalization_gap.precision_gap * 100:+.2f}% |",
                f"| Recall | {report.in_domain.recall * 100:.2f}% | "
                f"{report.out_of_domain.recall * 100:.2f}% | "
                f"{report.generalization_gap.recall_gap * 100:+.2f}% |",
                "",
                f"**Severity Assessment:** {report.generalization_gap.severity.upper()}  ",
                "",
            ]
        )

        # Per-generator confusion matrices
        lines.extend(
            [
                "---",
                "",
                "## 5. Per-Generator Confusion Matrices",
                "",
            ]
        )
        for gm in report.per_generator:
            cm = gm.confusion_matrix
            lines.extend(
                [
                    f"### {gm.generator_name} ({gm.domain})",
                    "```text",
                    "                 Predicted Real    Predicted AI",
                    f"Actual Real          {cm.tn:<10}       {cm.fp:<10}",
                    f"Actual AI            {cm.fn:<10}       {cm.tp:<10}",
                    "```",
                    "",
                ]
            )

        if report.notes:
            lines.extend(
                [
                    "---",
                    "",
                    "## 6. Engineering Notes & Analysis",
                    "",
                    report.notes,
                    "",
                ]
            )

        return "\n".join(lines)

    def save_report(
        self,
        report: GeneralizationReport,
        json_path: Union[str, Path],
        md_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Saves the generalization report to JSON and optionally Markdown."""
        j_file = Path(json_path)
        j_file.parent.mkdir(parents=True, exist_ok=True)
        with open(j_file, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(), f, indent=2)

        if md_path:
            m_file = Path(md_path)
            m_file.parent.mkdir(parents=True, exist_ok=True)
            with open(m_file, "w", encoding="utf-8") as f:
                f.write(self.format_markdown_report(report))
