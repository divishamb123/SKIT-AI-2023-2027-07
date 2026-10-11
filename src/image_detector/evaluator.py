"""
evaluator.py — Grouped Evaluation and Testing Engine for Image AI Detection

Sprint 2 Week 4: Testing image inference using grouped input samples.
Student: Divisha Manak Bohra (23ESKCA038)
"""

import csv
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Union

from src.image_detector.schemas import (
    ConfusionMatrixMetrics,
    GroupEvaluationMetrics,
    GroupedEvaluationReport,
    PredictionVerdict,
)
from src.image_detector.service import ImageDetectionService


class GroupedImageEvaluator:
    """
    Evaluator engine for testing image inference using grouped input samples.

    Supports grouping by:
    - Generator (e.g., cifar10, stable-diffusion-1.4, midjourney, biggan)
    - Ground truth class (Real vs AI)
    - Dataset partition & split (in-domain CIFAKE test vs out-of-domain GenImage holdout)
    - Batch/group size (1, 4, 8, 16, 32)
    - File format (PNG, JPEG, WEBP)
    """

    def __init__(
        self,
        service: Optional[ImageDetectionService] = None,
        default_chunk_size: int = 16,
    ):
        self.service = service or ImageDetectionService()
        self.default_chunk_size = default_chunk_size

    @staticmethod
    def load_manifest_samples(
        manifest_path: Union[str, Path],
        split: Optional[str] = None,
        limit_per_class: Optional[int] = None,
        limit_per_generator: Optional[int] = None,
        balance_classes: bool = True,
        project_root: Optional[Union[str, Path]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Loads and filters samples from manifest.csv.

        Resolves relative image paths against project_root if provided.
        """
        manifest_file = Path(manifest_path)
        if not manifest_file.is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest_file}")

        root_dir = Path(project_root) if project_root else manifest_file.parent.parent.parent

        rows: List[Dict[str, Any]] = []
        with open(manifest_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if split and row.get("split") != split:
                    continue

                raw_path = row["path"].strip()
                abs_path = (root_dir / raw_path).resolve() if not Path(raw_path).is_absolute() else Path(raw_path)

                rows.append({
                    "dataset": row.get("dataset", "unknown"),
                    "split": row.get("split", "unknown"),
                    "label": int(row["label"]),
                    "generator": row.get("generator", "unknown"),
                    "path": str(abs_path),
                    "filename": abs_path.name,
                })

        if limit_per_generator is not None:
            stratified: List[Dict[str, Any]] = []
            grouped = {}
            for r in rows:
                gen = r["generator"]
                grouped.setdefault(gen, []).append(r)
            for gen_samples in grouped.values():
                stratified.extend(gen_samples[:limit_per_generator])
            return stratified

        if limit_per_class is not None:
            if balance_classes:
                real_samples = [r for r in rows if r["label"] == 0][:limit_per_class]
                ai_samples = [r for r in rows if r["label"] == 1][:limit_per_class]
                return real_samples + ai_samples
            return rows[: limit_per_class * 2]

        return rows


    @staticmethod
    def group_samples_by(
        samples: Sequence[Dict[str, Any]], key: str = "generator"
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Groups sample dictionaries by a specific dictionary key."""
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for sample in samples:
            val = str(sample.get(key, "unspecified"))
            if val not in grouped:
                grouped[val] = []
            grouped[val].append(sample)
        return grouped

    @staticmethod
    def compute_metrics(
        y_true: Sequence[int],
        y_pred: Sequence[int],
        confidences: Sequence[float],
        group_name: str,
        group_type: str,
        total_latency_ms: float,
        failed_count: int = 0,
    ) -> GroupEvaluationMetrics:
        """
        Computes binary classification metrics from ground truth and predictions.
        Label 0 = REAL, Label 1 = AI_GENERATED.
        """
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
        tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

        total_evaluated = len(y_true)
        total_submitted = total_evaluated + failed_count

        accuracy = (tp + tn) / total_evaluated if total_evaluated > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        mean_conf = (
            sum(confidences) / len(confidences) if len(confidences) > 0 else 0.0
        )

        avg_latency = (
            total_latency_ms / total_evaluated if total_evaluated > 0 else 0.0
        )
        throughput_fps = (
            (total_evaluated / (total_latency_ms / 1000.0))
            if total_latency_ms > 0
            else 0.0
        )

        return GroupEvaluationMetrics(
            group_name=group_name,
            group_type=group_type,
            sample_count=total_submitted,
            successful_count=total_evaluated,
            failed_count=failed_count,
            accuracy=round(accuracy, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            specificity=round(specificity, 4),
            f1_score=round(f1, 4),
            mean_confidence=round(mean_conf, 4),
            confusion_matrix=ConfusionMatrixMetrics(tp=tp, fp=fp, tn=tn, fn=fn),
            total_latency_ms=round(total_latency_ms, 2),
            avg_latency_ms=round(avg_latency, 2),
            throughput_fps=round(throughput_fps, 2),
        )

    def evaluate_group(
        self,
        samples: Sequence[Dict[str, Any]],
        group_name: str,
        group_type: str,
        chunk_size: Optional[int] = None,
    ) -> GroupEvaluationMetrics:
        """
        Executes batched inference and calculates metrics for a specific sample group.
        """
        if not samples:
            return self.compute_metrics(
                [], [], [], group_name, group_type, total_latency_ms=0.0
            )

        active_chunk = chunk_size or self.default_chunk_size
        image_inputs = [(s.get("filename", Path(s["path"]).name), s["path"]) for s in samples]

        start_time = time.perf_counter()
        batch_resp = self.service.predict_batch(
            image_inputs, chunk_size=active_chunk, return_errors=True
        )
        total_time_ms = (time.perf_counter() - start_time) * 1000.0

        # Map predictions by filename for robust matching
        res_map = {r.filename: r for r in batch_resp.results}

        y_true: List[int] = []
        y_pred: List[int] = []
        confidences: List[float] = []

        for sample in samples:
            fname = sample.get("filename", Path(sample["path"]).name)
            if fname in res_map:
                res = res_map[fname]
                y_true.append(sample["label"])
                pred_label = 1 if res.verdict == PredictionVerdict.AI_GENERATED else 0
                y_pred.append(pred_label)
                confidences.append(res.confidence)

        return self.compute_metrics(
            y_true=y_true,
            y_pred=y_pred,
            confidences=confidences,
            group_name=group_name,
            group_type=group_type,
            total_latency_ms=total_time_ms,
            failed_count=len(batch_resp.errors),
        )

    def evaluate_grouped_dataset(
        self,
        samples: Sequence[Dict[str, Any]],
        grouping_keys: Sequence[str] = ("generator", "split", "dataset"),
        chunk_size: Optional[int] = None,
        notes: Optional[str] = None,
    ) -> GroupedEvaluationReport:
        """
        Evaluates a complete dataset with overall metrics and multiple grouped breakdowns.
        """
        # Overall evaluation
        overall_metric = self.evaluate_group(
            samples,
            group_name="All Samples Combined",
            group_type="overall",
            chunk_size=chunk_size,
        )

        breakdowns: Dict[str, List[GroupEvaluationMetrics]] = {}

        # Breakdown by label/class
        class_samples = self.group_samples_by(samples, "label")
        class_metrics: List[GroupEvaluationMetrics] = []
        for label_key, group_items in sorted(class_samples.items()):
            name = "Real (Class 0)" if str(label_key) == "0" else "AI-Generated (Class 1)"
            m = self.evaluate_group(group_items, group_name=name, group_type="class", chunk_size=chunk_size)
            class_metrics.append(m)
        breakdowns["by_class"] = class_metrics

        # Breakdown by each requested grouping key
        for g_key in grouping_keys:
            grouped = self.group_samples_by(samples, g_key)
            metrics_list: List[GroupEvaluationMetrics] = []
            for group_val, group_items in sorted(grouped.items()):
                m = self.evaluate_group(
                    group_items,
                    group_name=f"{g_key}:{group_val}",
                    group_type=g_key,
                    chunk_size=chunk_size,
                )
                metrics_list.append(m)
            breakdowns[f"by_{g_key}"] = metrics_list

        return GroupedEvaluationReport(
            model_version=self.service.model_version,
            device=str(self.service.device),
            overall_metrics=overall_metric,
            group_breakdowns=breakdowns,
            notes=notes,
        )

    def benchmark_group_sizes(
        self,
        samples: Sequence[Dict[str, Any]],
        group_sizes: Sequence[int] = (1, 4, 8, 16, 32),
    ) -> List[GroupEvaluationMetrics]:
        """
        Profiles throughput and latency scaling across varying group/chunk sizes.
        """
        benchmark_results: List[GroupEvaluationMetrics] = []
        for g_size in group_sizes:
            metrics = self.evaluate_group(
                samples,
                group_name=f"Group Size {g_size}",
                group_type="batch_size_scaling",
                chunk_size=g_size,
            )
            benchmark_results.append(metrics)
        return benchmark_results

    @staticmethod
    def format_markdown_report(report: GroupedEvaluationReport) -> str:
        """Generates a structured Markdown evaluation report."""
        lines = [
            "# Image Detection Baseline Evaluation Report — Grouped Input Samples",
            "",
            "**Sprint:** Sprint 2 (Baseline Detection Service) — Week 4 Deliverable  ",
            "**Member:** Divisha Manak Bohra (`23ESKCA038`) — Image Detection Modality & Image Interface  ",
            "**Project ID:** `SKIT/AI/2023-2027/07`  ",
            f"**Model Version:** `{report.model_version}`  ",
            f"**Device Utilized:** `{report.device}`  ",
            f"**Timestamp:** `{report.timestamp}`  ",
            "",
            "---",
            "",
            "## 1. Executive Summary & Overall Baseline Performance",
            "",
            "This evaluation assesses the baseline **ResNet-18** image classifier across representative grouped sample partitions from `manifest.csv`. "
            "Because the baseline model uses untrained random classifier head weights prior to Sprint 3 fine-tuning, its raw binary classification accuracy is expected to hover around random chance (~50%), while serving as the verified baseline reference point.",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Total Samples Evaluated | **{report.overall_metrics.successful_count:,}** |",
            f"| Failed / Inaccessible | **{report.overall_metrics.failed_count}** |",
            f"| Overall Accuracy | **{report.overall_metrics.accuracy * 100:.2f}%** |",
            f"| Precision (AI) | **{report.overall_metrics.precision * 100:.2f}%** |",
            f"| Recall (AI) | **{report.overall_metrics.recall * 100:.2f}%** |",
            f"| Specificity (Real) | **{report.overall_metrics.specificity * 100:.2f}%** |",
            f"| F1-Score | **{report.overall_metrics.f1_score:.4f}** |",
            f"| Mean Confidence | **{report.overall_metrics.mean_confidence:.4f}** |",
            f"| Total Latency | **{report.overall_metrics.total_latency_ms:.1f} ms** |",
            f"| Avg Latency / Image | **{report.overall_metrics.avg_latency_ms:.2f} ms** |",
            f"| Throughput | **{report.overall_metrics.throughput_fps:.1f} FPS** |",
            "",
            "### Confusion Matrix (Overall)",
            "```text",
            "                 Predicted Real    Predicted AI",
            f"Actual Real          {report.overall_metrics.confusion_matrix.tn:<10}       {report.overall_metrics.confusion_matrix.fp:<10}",
            f"Actual AI            {report.overall_metrics.confusion_matrix.fn:<10}       {report.overall_metrics.confusion_matrix.tp:<10}",
            "```",
            "",
            "---",
            "",
            "## 2. Grouped Breakdown Analysis",
            "",
        ]

        for b_name, b_metrics in report.group_breakdowns.items():
            clean_title = b_name.replace("by_", "Grouped by ").replace("_", " ").title()
            lines.append(f"### {clean_title}")
            lines.append("")
            lines.append(
                "| Group Name | Samples | Accuracy | Precision | Recall | F1 | Mean Conf | Latency (ms) | Throughput (FPS) |"
            )
            lines.append(
                "|---|---|---|---|---|---|---|---|---|"
            )
            for m in b_metrics:
                lines.append(
                    f"| **{m.group_name}** | {m.successful_count} | {m.accuracy * 100:.1f}% | {m.precision * 100:.1f}% | {m.recall * 100:.1f}% | {m.f1_score:.3f} | {m.mean_confidence:.3f} | {m.total_latency_ms:.1f} | {m.throughput_fps:.1f} |"
                )
            lines.append("")

        if report.notes:
            lines.extend([
                "---",
                "",
                "## 3. Engineering Notes & Analysis",
                "",
                report.notes,
                "",
            ])

        return "\n".join(lines)

    def save_report(
        self,
        report: GroupedEvaluationReport,
        json_path: Union[str, Path],
        md_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Saves evaluation report to JSON and Markdown destinations."""
        j_file = Path(json_path)
        j_file.parent.mkdir(parents=True, exist_ok=True)
        with open(j_file, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(), f, indent=2)

        if md_path:
            m_file = Path(md_path)
            m_file.parent.mkdir(parents=True, exist_ok=True)
            with open(m_file, "w", encoding="utf-8") as f:
                f.write(self.format_markdown_report(report))
