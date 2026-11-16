"""
test_generalization_unseen.py — Automated Tests for Cross-Generator Generalization

Sprint 3 (Accuracy Optimisation) — Week 2 Test Suite
Member: Divisha Manak Bohra (23ESKCA038)

Validates the GeneralizationEvaluator's ability to:
1. Classify generators as in-domain vs out-of-domain
2. Generate balanced synthetic test samples
3. Evaluate per-generator binary classification metrics
4. Compute domain-level aggregated metrics and generalization gap
5. Produce structured Markdown and JSON reports
"""

from PIL import Image
import pytest

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
    GeneralizationReport,
    GeneratorMetrics,
)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ensemble_detector():
    """Module-scoped ensemble detector (expensive to create)."""
    return ImageEnsembleDetector(weights=DEFAULT_WEIGHTS)


@pytest.fixture(scope="module")
def evaluator(ensemble_detector):
    """Module-scoped GeneralizationEvaluator backed by ensemble detector."""
    return GeneralizationEvaluator(detector=ensemble_detector, threshold=0.5)


@pytest.fixture
def synthetic_samples(evaluator):
    """Creates synthetic test samples with 3 samples per class per generator."""
    return GeneralizationEvaluator.create_synthetic_samples(
        samples_per_class_per_generator=3
    )


# --------------------------------------------------------------------------
# 1. Schema & Initialization Tests
# --------------------------------------------------------------------------
class TestSchemaAndInitialization:

    def test_generalization_evaluator_initialization(
        self, evaluator, ensemble_detector
    ):
        """Verify GeneralizationEvaluator instantiates with ensemble detector and correct domain sets."""
        assert evaluator.detector is ensemble_detector
        assert evaluator.threshold == 0.5
        assert evaluator.in_domain_generators == IN_DOMAIN_GENERATORS
        assert evaluator.out_of_domain_generators == OUT_OF_DOMAIN_GENERATORS

    def test_domain_classification(self, evaluator):
        """Verify generators are correctly classified as in_domain vs out_of_domain."""
        # In-domain generators
        assert evaluator.classify_domain("cifar10") == "in_domain"
        assert evaluator.classify_domain("stable-diffusion-1.4") == "in_domain"

        # Out-of-domain / unseen generators
        assert evaluator.classify_domain("real") == "out_of_domain"
        assert evaluator.classify_domain("stable-diffusion") == "out_of_domain"
        assert evaluator.classify_domain("midjourney") == "out_of_domain"
        assert evaluator.classify_domain("biggan") == "out_of_domain"

        # Unknown generator defaults to out_of_domain
        assert evaluator.classify_domain("dall-e-3") == "out_of_domain"

    def test_generator_metrics_schema(self):
        """Verify GeneratorMetrics Pydantic model validates correctly."""
        from src.image_detector.schemas import ConfusionMatrixMetrics

        metrics = GeneratorMetrics(
            generator_name="midjourney",
            domain="out_of_domain",
            sample_count=100,
            accuracy=0.75,
            precision=0.80,
            recall=0.70,
            specificity=0.80,
            f1_score=0.7467,
            mean_confidence=0.82,
            robustness_score=0.75,
            confusion_matrix=ConfusionMatrixMetrics(tp=35, fp=10, tn=40, fn=15),
            latency_ms=150.5,
            throughput_fps=664.5,
        )
        assert metrics.generator_name == "midjourney"
        assert metrics.domain == "out_of_domain"
        assert metrics.sample_count == 100
        assert 0.0 <= metrics.accuracy <= 1.0
        assert 0.0 <= metrics.robustness_score <= 1.0
        assert metrics.confusion_matrix.tp == 35


# --------------------------------------------------------------------------
# 2. Synthetic Sample Generation Tests
# --------------------------------------------------------------------------
class TestSyntheticSamples:

    def test_create_synthetic_samples(self):
        """Verify synthetic test samples have correct structure."""
        samples = GeneralizationEvaluator.create_synthetic_samples(
            generators=["cifar10", "midjourney"],
            samples_per_class_per_generator=2,
        )

        # 2 generators × 2 classes × 2 samples = 8
        assert len(samples) == 8

        # Verify required keys
        required_keys = {"image", "label", "generator", "dataset", "split", "filename"}
        for sample in samples:
            assert required_keys.issubset(
                sample.keys()
            ), f"Missing keys: {required_keys - sample.keys()}"
            assert isinstance(sample["image"], Image.Image)
            assert sample["label"] in (0, 1)
            assert sample["generator"] in ("cifar10", "midjourney")

    def test_synthetic_samples_class_balance(self):
        """Verify synthetic samples have balanced classes per generator."""
        samples = GeneralizationEvaluator.create_synthetic_samples(
            samples_per_class_per_generator=5
        )

        # Group by generator and check balance
        from collections import Counter

        for gen in ALL_KNOWN_GENERATORS:
            gen_samples = [s for s in samples if s["generator"] == gen]
            label_counts = Counter(s["label"] for s in gen_samples)
            assert (
                label_counts[0] == 5
            ), f"Expected 5 real samples for {gen}, got {label_counts[0]}"
            assert (
                label_counts[1] == 5
            ), f"Expected 5 AI samples for {gen}, got {label_counts[1]}"


# --------------------------------------------------------------------------
# 3. Per-Generator Evaluation Tests
# --------------------------------------------------------------------------
class TestPerGeneratorEvaluation:

    def test_evaluate_single_generator(self, evaluator):
        """Evaluate ensemble on synthetic samples for one generator, verify metrics schema."""
        samples = GeneralizationEvaluator.create_synthetic_samples(
            generators=["midjourney"],
            samples_per_class_per_generator=3,
        )

        metrics = evaluator.evaluate_generator(samples, "midjourney")

        assert isinstance(metrics, GeneratorMetrics)
        assert metrics.generator_name == "midjourney"
        assert metrics.domain == "out_of_domain"
        assert metrics.sample_count == 6  # 3 per class × 2 classes
        assert 0.0 <= metrics.accuracy <= 1.0
        assert 0.0 <= metrics.precision <= 1.0
        assert 0.0 <= metrics.recall <= 1.0
        assert 0.0 <= metrics.f1_score <= 1.0
        assert metrics.latency_ms > 0.0

    def test_evaluate_all_generators(self, evaluator):
        """Evaluate across all generators, verify each returns valid metrics."""
        samples = GeneralizationEvaluator.create_synthetic_samples(
            samples_per_class_per_generator=2,
        )

        # Group and evaluate each generator
        from collections import defaultdict

        grouped = defaultdict(list)
        for s in samples:
            grouped[s["generator"]].append(s)

        all_metrics = []
        for gen_name, gen_samples in grouped.items():
            m = evaluator.evaluate_generator(gen_samples, gen_name)
            all_metrics.append(m)
            assert isinstance(m, GeneratorMetrics)
            assert m.generator_name == gen_name
            assert m.sample_count > 0

        assert len(all_metrics) == len(ALL_KNOWN_GENERATORS)

    def test_per_generator_metrics_values(self, evaluator):
        """Verify accuracy, precision, recall, F1 are all in [0,1]."""
        samples = GeneralizationEvaluator.create_synthetic_samples(
            generators=["cifar10", "biggan"],
            samples_per_class_per_generator=4,
        )

        for gen in ["cifar10", "biggan"]:
            gen_samples = [s for s in samples if s["generator"] == gen]
            metrics = evaluator.evaluate_generator(gen_samples, gen)

            assert 0.0 <= metrics.accuracy <= 1.0
            assert 0.0 <= metrics.precision <= 1.0
            assert 0.0 <= metrics.recall <= 1.0
            assert 0.0 <= metrics.specificity <= 1.0
            assert 0.0 <= metrics.f1_score <= 1.0
            assert 0.0 <= metrics.mean_confidence <= 1.0
            assert 0.0 <= metrics.robustness_score <= 1.0

            # Confusion matrix components should be non-negative
            cm = metrics.confusion_matrix
            assert cm.tp >= 0 and cm.fp >= 0 and cm.tn >= 0 and cm.fn >= 0
            # Sum of confusion matrix should equal sample count
            assert cm.tp + cm.fp + cm.tn + cm.fn == metrics.sample_count


# --------------------------------------------------------------------------
# 4. Cross-Domain Generalization Tests
# --------------------------------------------------------------------------
class TestCrossDomainGeneralization:

    def test_generalization_gap_computation(self, evaluator):
        """Verify generalization gap = in_domain_accuracy - out_of_domain_accuracy."""
        samples = GeneralizationEvaluator.create_synthetic_samples(
            samples_per_class_per_generator=3,
        )

        report = evaluator.evaluate_generalization(samples)
        gap = report.generalization_gap

        # Verify gap computation
        expected_gap = round(gap.in_domain_accuracy - gap.out_of_domain_accuracy, 4)
        assert gap.accuracy_gap == expected_gap

        # Verify in-domain accuracy matches the in_domain DomainMetrics
        assert gap.in_domain_accuracy == report.in_domain.accuracy
        assert gap.out_of_domain_accuracy == report.out_of_domain.accuracy

        # Severity should be one of the valid levels
        assert gap.severity in ("minimal", "moderate", "significant")

    def test_domain_level_metrics(self, evaluator):
        """Verify in_domain and out_of_domain DomainMetrics are computed correctly."""
        samples = GeneralizationEvaluator.create_synthetic_samples(
            samples_per_class_per_generator=3,
        )

        report = evaluator.evaluate_generalization(samples)

        # In-domain should contain cifar10 and stable-diffusion-1.4
        assert isinstance(report.in_domain, DomainMetrics)
        assert report.in_domain.domain == "in_domain"
        assert set(report.in_domain.generators) == IN_DOMAIN_GENERATORS

        # Out-of-domain should contain real, stable-diffusion, midjourney, biggan
        assert isinstance(report.out_of_domain, DomainMetrics)
        assert report.out_of_domain.domain == "out_of_domain"
        assert set(report.out_of_domain.generators) == OUT_OF_DOMAIN_GENERATORS

        # Total samples should add up
        assert (
            report.in_domain.total_samples + report.out_of_domain.total_samples
            == report.total_samples_evaluated
        )

        # Domain-level metrics in valid ranges
        for domain in (report.in_domain, report.out_of_domain):
            assert 0.0 <= domain.accuracy <= 1.0
            assert 0.0 <= domain.precision <= 1.0
            assert 0.0 <= domain.recall <= 1.0
            assert 0.0 <= domain.f1_score <= 1.0
            assert 0.0 <= domain.mean_robustness <= 1.0


# --------------------------------------------------------------------------
# 5. Report Generation Tests
# --------------------------------------------------------------------------
class TestReportGeneration:

    def test_full_generalization_report(self, evaluator):
        """Generate complete GeneralizationReport, verify all fields populated."""
        samples = GeneralizationEvaluator.create_synthetic_samples(
            samples_per_class_per_generator=2,
        )

        report = evaluator.evaluate_generalization(
            samples,
            notes="Test evaluation using synthetic samples for verification.",
        )

        assert isinstance(report, GeneralizationReport)
        assert report.report_id is not None and len(report.report_id) > 0
        assert report.timestamp is not None
        assert report.ensemble_version is not None
        assert report.device is not None
        assert report.threshold == 0.5
        assert report.total_samples_evaluated > 0
        assert 0.0 <= report.overall_accuracy <= 1.0
        assert 0.0 <= report.overall_f1 <= 1.0
        assert len(report.per_generator) == len(ALL_KNOWN_GENERATORS)
        assert report.in_domain is not None
        assert report.out_of_domain is not None
        assert report.generalization_gap is not None
        assert report.total_latency_ms > 0.0
        assert (
            report.notes == "Test evaluation using synthetic samples for verification."
        )

    def test_markdown_report_generation(self, evaluator):
        """Verify markdown report contains expected sections and table headers."""
        samples = GeneralizationEvaluator.create_synthetic_samples(
            samples_per_class_per_generator=2,
        )

        report = evaluator.evaluate_generalization(samples)
        md = GeneralizationEvaluator.format_markdown_report(report)

        assert isinstance(md, str)
        assert len(md) > 500  # Should be a substantial report

        # Verify key section headers
        assert "# Cross-Generator Generalization Evaluation Report" in md
        assert "## 1. Executive Summary" in md
        assert "## 2. Per-Generator Breakdown" in md
        assert "## 3. Domain-Level Analysis" in md
        assert "## 4. Generalization Gap Analysis" in md
        assert "## 5. Per-Generator Confusion Matrices" in md

        # Verify table headers present
        assert "| Generator | Domain | Samples |" in md
        assert "| Metric | In-Domain | Out-of-Domain | Gap" in md

        # Verify generator names appear
        assert "midjourney" in md
        assert "biggan" in md
        assert "cifar10" in md

        # Verify domain labels appear
        assert "in_domain" in md
        assert "out_of_domain" in md

        # Verify severity label is present
        assert report.generalization_gap.severity.upper() in md
