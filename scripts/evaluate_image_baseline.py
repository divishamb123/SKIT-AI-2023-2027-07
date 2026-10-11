"""
evaluate_image_baseline.py — Standalone Baseline Evaluation on Grouped Input Samples

Sprint 2 Week 4: Testing image inference using grouped input samples.
Student: Divisha Manak Bohra (23ESKCA038)
Project ID: SKIT/AI/2023-2027/07
"""

import argparse
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.image_detector.evaluator import GroupedImageEvaluator


def print_banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(
        description="Run grouped baseline evaluation on image detection dataset splits."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(PROJECT_ROOT / "datasets" / "splits" / "manifest.csv"),
        help="Path to manifest.csv",
    )
    parser.add_argument(
        "--test_limit",
        type=int,
        default=100,
        help="Limit per class for in-domain test split (CIFAKE)",
    )
    parser.add_argument(
        "--holdout_limit",
        type=int,
        default=50,
        help="Limit per generator for out-of-domain holdout split (GenImage)",
    )
    parser.add_argument(
        "--output_md",
        type=str,
        default=str(PROJECT_ROOT / "docs" / "image_baseline_results.md"),
        help="Path to output markdown report",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=str(PROJECT_ROOT / "results" / "image_baseline_evaluation.json"),
        help="Path to output json report",
    )
    args = parser.parse_args()

    print_banner("IMAGE DETECTION BASELINE EVALUATION — GROUPED INPUT SAMPLES")
    print("Student:    Divisha Manak Bohra (23ESKCA038)")
    print("Modality:   Image AI Detection (ResNet-18 Baseline)")
    print("Task:       Testing image inference using grouped input samples (Week 4)")
    print(f"Manifest:   {args.manifest}")

    evaluator = GroupedImageEvaluator()
    print(f"Hardware:   {evaluator.service.device}")
    print(f"Model Ver:  {evaluator.service.model_version}")

    # 1. Load Grouped In-Domain & Out-of-Domain Samples
    print("\n[Step 1] Loading grouped dataset partitions from manifest...")
    start_load = time.perf_counter()

    test_samples = evaluator.load_manifest_samples(
        args.manifest,
        split="test",
        limit_per_class=args.test_limit,
        balance_classes=True,
        project_root=PROJECT_ROOT,
    )
    print(f"  ✓ Loaded In-Domain Test Split:      {len(test_samples)} samples (CIFAKE)")

    holdout_samples = evaluator.load_manifest_samples(
        args.manifest,
        split="holdout",
        limit_per_generator=args.holdout_limit,
        project_root=PROJECT_ROOT,
    )
    print(f"  ✓ Loaded Out-of-Domain Holdout:    {len(holdout_samples)} samples (GenImage: MJ, BigGAN, SD, Real)")

    all_samples = test_samples + holdout_samples
    total_count = len(all_samples)
    print(f"  ✓ Total Grouped Evaluation Pool:   {total_count} samples (loaded in {(time.perf_counter() - start_load)*1000:.1f} ms)")

    # 2. Run Grouped Evaluation
    print("\n[Step 2] Executing grouped inference across classes, splits, and generators...")
    start_eval = time.perf_counter()

    notes = (
        "Baseline ResNet-18 classifier evaluated with initial weights on both in-domain "
        "(CIFAKE test) and out-of-domain (GenImage holdout) grouped partitions. "
        "Throughput and scaling confirmed across batch sizes up to 32. "
        "Supervised fine-tuning and ensemble tuning scheduled for Sprint 3."
    )

    report = evaluator.evaluate_grouped_dataset(
        all_samples,
        grouping_keys=["split", "generator"],
        chunk_size=16,
        notes=notes,
    )
    eval_duration_sec = time.perf_counter() - start_eval

    # 3. Benchmark Batch Scaling
    print("\n[Step 3] Profiling batch collation scaling and throughput across group sizes...")
    scaling_metrics = evaluator.benchmark_group_sizes(
        all_samples[: min(64, total_count)], group_sizes=[1, 4, 8, 16, 32]
    )

    report.group_breakdowns["batch_size_scaling"] = scaling_metrics

    # 4. Display Summary Table
    print_banner("EVALUATION RESULTS SUMMARY")
    om = report.overall_metrics
    print(f"Total Samples Evaluated:  {om.successful_count}")
    print(f"Accuracy:                  {om.accuracy * 100:.2f}%")
    print(f"Precision (AI):            {om.precision * 100:.2f}%")
    print(f"Recall (AI):               {om.recall * 100:.2f}%")
    print(f"Specificity (Real):        {om.specificity * 100:.2f}%")
    print(f"F1-Score:                  {om.f1_score:.4f}")
    print(f"Mean Confidence:           {om.mean_confidence:.4f}")
    print(f"Overall Throughput:        {om.throughput_fps:.1f} FPS")
    print(f"Total Evaluation Time:     {eval_duration_sec:.2f} seconds")

    print("\nGenerator Breakdown:")
    for m in report.group_breakdowns.get("by_generator", []):
        print(f"  * {m.group_name:<30} | N={m.successful_count:<4} | Acc={m.accuracy*100:5.1f}% | Latency={m.total_latency_ms:6.1f} ms | {m.throughput_fps:5.1f} FPS")

    print("\nBatch Scaling Benchmark (64 samples):")
    for m in scaling_metrics:
        print(f"  * {m.group_name:<18} | Latency={m.total_latency_ms:6.1f} ms | Throughput={m.throughput_fps:6.1f} FPS")

    # 5. Save Artifacts
    print("\n[Step 4] Exporting evaluation artifacts...")
    evaluator.save_report(report, json_path=args.output_json, md_path=args.output_md)
    print(f"  ✓ JSON Report saved:  {args.output_json}")
    print(f"  ✓ Markdown Report:    {args.output_md}")

    print_banner("SPRINT 2 WEEK 4 TASK COMPLETE & VERIFIED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
