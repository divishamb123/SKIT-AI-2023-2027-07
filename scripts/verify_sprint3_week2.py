"""
verify_sprint3_week2.py — Cross-Generator Generalization Verification Suite

Sprint 3 (Accuracy Optimisation) — Week 2 Verification
Member: Divisha Manak Bohra (23ESKCA038)

Validates:
1. GeneralizationEvaluator Initialization & Domain Configuration
2. Synthetic Sample Generation & Class Balance
3. Per-Generator Ensemble Evaluation (cifar10, stable-diffusion-1.4, stable-diffusion, midjourney, biggan)
4. In-Domain vs Out-of-Domain Accuracy Metrics
5. Generalization Gap Computation & Analysis
6. Per-Generator Robustness Scoring
7. Markdown & JSON Report Generation
8. Pytest Test Suite Execution (tests/test_generalization_unseen.py)
"""

import json
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from src.image_detector.ensemble import (
    DEFAULT_WEIGHTS,
    ImageEnsembleDetector,
)
from src.image_detector.generalization import (
    ALL_KNOWN_GENERATORS,
    IN_DOMAIN_GENERATORS,
    OUT_OF_DOMAIN_GENERATORS,
    DomainMetrics,
    GeneralizationEvaluator,
    GeneralizationGap,
    GeneratorMetrics,
)


def run_verification():
    print("=" * 75)
    print("SPRINT 3 — WEEK 2: CROSS-GENERATOR GENERALIZATION VERIFICATION SUITE")
    print("Candidate: Divisha Manak Bohra (23ESKCA038)")
    print("User Story: Refining the image detection module")
    print("Task: Testing generalisation on unseen image generators")
    print("=" * 75)

    # --------------------------------------------------------------------------
    # Stage 1: GeneralizationEvaluator Initialization & Domain Configuration
    # --------------------------------------------------------------------------
    print("\n--- [Stage 1] Verifying GeneralizationEvaluator Initialization ---")
    detector = ImageEnsembleDetector(weights=DEFAULT_WEIGHTS)
    evaluator = GeneralizationEvaluator(
        detector=detector,
        threshold=0.5,
    )

    assert evaluator.detector is detector, "Detector not assigned correctly!"
    assert evaluator.threshold == 0.5, "Threshold not set correctly!"
    assert evaluator.in_domain_generators == IN_DOMAIN_GENERATORS
    assert evaluator.out_of_domain_generators == OUT_OF_DOMAIN_GENERATORS
    print(f"  [PASS] Ensemble detector: {detector.ensemble_version}")
    print(f"  [PASS] Hardware device: {detector.device}")
    print(f"  [PASS] Decision threshold (τ): {evaluator.threshold}")

    # Domain classification
    assert evaluator.classify_domain("cifar10") == "in_domain"
    assert evaluator.classify_domain("stable-diffusion-1.4") == "in_domain"
    assert evaluator.classify_domain("midjourney") == "out_of_domain"
    assert evaluator.classify_domain("biggan") == "out_of_domain"
    assert evaluator.classify_domain("real") == "out_of_domain"
    assert evaluator.classify_domain("stable-diffusion") == "out_of_domain"
    print(f"  [PASS] In-domain generators: {sorted(IN_DOMAIN_GENERATORS)}")
    print(f"  [PASS] Out-of-domain generators: {sorted(OUT_OF_DOMAIN_GENERATORS)}")

    # --------------------------------------------------------------------------
    # Stage 2: Synthetic Sample Generation & Class Balance
    # --------------------------------------------------------------------------
    print("\n--- [Stage 2] Verifying Synthetic Sample Generation & Balance ---")
    samples = GeneralizationEvaluator.create_synthetic_samples(
        samples_per_class_per_generator=4,
    )

    expected_count = len(ALL_KNOWN_GENERATORS) * 2 * 4  # generators × classes × samples
    assert (
        len(samples) == expected_count
    ), f"Expected {expected_count} samples, got {len(samples)}"
    print(
        f"  [PASS] Generated {len(samples)} synthetic samples across {len(ALL_KNOWN_GENERATORS)} generators"
    )

    # Verify class balance per generator
    for gen in sorted(ALL_KNOWN_GENERATORS):
        gen_samples = [s for s in samples if s["generator"] == gen]
        label_counts = Counter(s["label"] for s in gen_samples)
        assert (
            label_counts[0] == 4
        ), f"Class imbalance for {gen}: real={label_counts[0]}"
        assert label_counts[1] == 4, f"Class imbalance for {gen}: AI={label_counts[1]}"

    print(
        "  [PASS] All generators have perfectly balanced classes (4 Real / 4 AI each)"
    )

    # Verify required keys
    required_keys = {"image", "label", "generator", "dataset", "split", "filename"}
    for s in samples:
        assert required_keys.issubset(s.keys())
        assert isinstance(s["image"], Image.Image)
    print(
        "  [PASS] All samples contain required keys: image, label, generator, dataset, split, filename"
    )

    # --------------------------------------------------------------------------
    # Stage 3: Per-Generator Ensemble Evaluation
    # --------------------------------------------------------------------------
    print("\n--- [Stage 3] Evaluating Per-Generator Ensemble Performance ---")
    grouped = defaultdict(list)
    for s in samples:
        grouped[s["generator"]].append(s)

    all_gen_metrics = []
    for gen_name in sorted(grouped.keys()):
        gen_samples = grouped[gen_name]
        t0 = time.perf_counter()
        gen_metrics = evaluator.evaluate_generator(gen_samples, gen_name)
        lat = (time.perf_counter() - t0) * 1000.0

        assert isinstance(gen_metrics, GeneratorMetrics)
        assert gen_metrics.sample_count == len(gen_samples)
        assert 0.0 <= gen_metrics.accuracy <= 1.0
        assert 0.0 <= gen_metrics.precision <= 1.0
        assert 0.0 <= gen_metrics.recall <= 1.0
        assert 0.0 <= gen_metrics.f1_score <= 1.0

        domain_tag = gen_metrics.domain.upper().replace("_", "-")
        print(
            f"  [PASS] {gen_name:<24s} | {domain_tag:<13s} | "
            f"Acc: {gen_metrics.accuracy * 100:5.1f}% | "
            f"F1: {gen_metrics.f1_score:.3f} | "
            f"Robust: {gen_metrics.robustness_score:.3f} | "
            f"Lat: {lat:.1f}ms"
        )
        all_gen_metrics.append(gen_metrics)

    assert len(all_gen_metrics) == len(ALL_KNOWN_GENERATORS)
    print(
        f"\n  [PASS] All {len(all_gen_metrics)} generators evaluated with valid metrics"
    )

    # --------------------------------------------------------------------------
    # Stage 4: In-Domain vs Out-of-Domain Accuracy Metrics
    # --------------------------------------------------------------------------
    print("\n--- [Stage 4] Computing Domain-Level Aggregated Metrics ---")
    report = evaluator.evaluate_generalization(
        samples,
        notes="Sprint 3 Week 2 verification using synthetic test images.",
    )

    in_dom = report.in_domain
    out_dom = report.out_of_domain

    assert isinstance(in_dom, DomainMetrics)
    assert isinstance(out_dom, DomainMetrics)
    assert in_dom.domain == "in_domain"
    assert out_dom.domain == "out_of_domain"
    assert set(in_dom.generators) == IN_DOMAIN_GENERATORS
    assert set(out_dom.generators) == OUT_OF_DOMAIN_GENERATORS
    assert (
        in_dom.total_samples + out_dom.total_samples == report.total_samples_evaluated
    )

    print(
        f"  [PASS] In-Domain:      Generators={sorted(in_dom.generators)}, "
        f"Samples={in_dom.total_samples}, Acc={in_dom.accuracy * 100:.1f}%, F1={in_dom.f1_score:.3f}"
    )
    print(
        f"  [PASS] Out-of-Domain:  Generators={sorted(out_dom.generators)}, "
        f"Samples={out_dom.total_samples}, Acc={out_dom.accuracy * 100:.1f}%, F1={out_dom.f1_score:.3f}"
    )

    # --------------------------------------------------------------------------
    # Stage 5: Generalization Gap Computation & Analysis
    # --------------------------------------------------------------------------
    print("\n--- [Stage 5] Verifying Generalization Gap Computation ---")
    gap = report.generalization_gap

    assert isinstance(gap, GeneralizationGap)
    expected_acc_gap = round(gap.in_domain_accuracy - gap.out_of_domain_accuracy, 4)
    assert (
        gap.accuracy_gap == expected_acc_gap
    ), f"Accuracy gap mismatch: {gap.accuracy_gap} != {expected_acc_gap}"
    assert gap.in_domain_accuracy == in_dom.accuracy
    assert gap.out_of_domain_accuracy == out_dom.accuracy
    assert gap.severity in ("minimal", "moderate", "significant")

    print(f"  [PASS] In-Domain Accuracy    : {gap.in_domain_accuracy * 100:.2f}%")
    print(f"  [PASS] Out-of-Domain Accuracy: {gap.out_of_domain_accuracy * 100:.2f}%")
    print(f"  [PASS] Accuracy Gap (Δ)      : {gap.accuracy_gap * 100:+.2f}%")
    print(f"  [PASS] F1 Gap (Δ)            : {gap.f1_gap:+.4f}")
    print(f"  [PASS] Precision Gap (Δ)     : {gap.precision_gap * 100:+.2f}%")
    print(f"  [PASS] Recall Gap (Δ)        : {gap.recall_gap * 100:+.2f}%")
    print(f"  [PASS] Gap Severity          : {gap.severity.upper()}")

    # --------------------------------------------------------------------------
    # Stage 6: Per-Generator Robustness Scoring
    # --------------------------------------------------------------------------
    print("\n--- [Stage 6] Verifying Per-Generator Robustness Scores ---")
    for gm in report.per_generator:
        # Robustness = 1.0 - |0.5 - accuracy|
        expected_robustness = round(1.0 - abs(0.5 - gm.accuracy), 4)
        assert gm.robustness_score == expected_robustness, (
            f"Robustness mismatch for {gm.generator_name}: "
            f"{gm.robustness_score} != {expected_robustness}"
        )
        assert 0.0 <= gm.robustness_score <= 1.0

        print(
            f"  [PASS] {gm.generator_name:<24s} | "
            f"Robustness: {gm.robustness_score:.4f} | "
            f"Accuracy: {gm.accuracy * 100:.1f}%"
        )

    # --------------------------------------------------------------------------
    # Stage 7: Markdown & JSON Report Generation
    # --------------------------------------------------------------------------
    print("\n--- [Stage 7] Verifying Report Generation (Markdown & JSON) ---")

    # Generate Markdown
    md_report = GeneralizationEvaluator.format_markdown_report(report)
    assert isinstance(md_report, str)
    assert len(md_report) > 500

    # Verify key sections exist
    assert "# Cross-Generator Generalization Evaluation Report" in md_report
    assert "## 1. Executive Summary" in md_report
    assert "## 2. Per-Generator Breakdown" in md_report
    assert "## 3. Domain-Level Analysis" in md_report
    assert "## 4. Generalization Gap Analysis" in md_report
    assert "## 5. Per-Generator Confusion Matrices" in md_report
    print(
        f"  [PASS] Markdown report generated: {len(md_report):,} characters, all 5 sections present"
    )

    # Save to temporary files
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "generalization_report.json"
        md_path = Path(tmpdir) / "generalization_report.md"

        evaluator.save_report(report, json_path=json_path, md_path=md_path)

        assert json_path.is_file(), "JSON report not saved!"
        assert md_path.is_file(), "Markdown report not saved!"

        # Verify JSON is parseable and contains expected fields
        with open(json_path, "r") as f:
            parsed = json.load(f)
        assert "report_id" in parsed
        assert "per_generator" in parsed
        assert "generalization_gap" in parsed
        assert "in_domain" in parsed
        assert "out_of_domain" in parsed

        print(
            f"  [PASS] JSON report saved ({json_path.stat().st_size:,} bytes) and parseable"
        )
        print(f"  [PASS] Markdown report saved ({md_path.stat().st_size:,} bytes)")

    # --------------------------------------------------------------------------
    # Stage 8: Pytest Test Suite Execution
    # --------------------------------------------------------------------------
    print(
        "\n--- [Stage 8] Running Pytest Suite (tests/test_generalization_unseen.py) ---"
    )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_generalization_unseen.py", "-v"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode == 0:
        # Count passed tests from pytest output
        passed_count = result.stdout.count(" PASSED")
        print(f"  [PASS] All {passed_count} automated pytest unit tests passed (100%).")
    else:
        print(f"  [FAIL] Pytest encountered errors:\n{result.stdout}\n{result.stderr}")
        sys.exit(1)

    print("\n" + "=" * 75)
    print("ALL SPRINT 3 — WEEK 2 DELIVERABLES FULLY VERIFIED (100% COMPLETE)")
    print("=" * 75)


if __name__ == "__main__":
    run_verification()
