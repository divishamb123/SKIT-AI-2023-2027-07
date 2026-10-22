"""
test_text_dataset_audit.py — Unit tests for text dataset audit & leakage inspection
"""

import numpy as np
import pandas as pd

from scripts.audit_text_dataset import (
    compute_exact_duplicates,
    compute_length_stats,
    compute_near_duplicates_jaccard,
    compute_normalized_duplicates,
    compute_split_label_counts,
    generate_markdown_report,
    get_3gram_shingles,
    normalize_text,
)


def test_normalize_text():
    raw = "  Hello, WORLD!  This is a TEST... with \n whitespace and --punctuation. "
    normalized = normalize_text(raw)
    assert normalized == "hello world this is a test with whitespace and punctuation"


def test_get_3gram_shingles():
    short_text = "hello world"
    short_shingles = get_3gram_shingles(short_text)
    assert short_shingles == {"hello", "world"}

    long_text = "the quick brown fox jumps over the lazy dog"
    long_shingles = get_3gram_shingles(long_text)
    assert "the quick brown" in long_shingles
    assert "quick brown fox" in long_shingles
    assert "over the lazy" in long_shingles
    assert len(long_shingles) == 7


def test_compute_length_stats():
    series = pd.Series([10, 20, 30, 40, 50])
    stats = compute_length_stats(series)
    assert stats["count"] == 5
    assert stats["mean"] == 30.0
    assert stats["median"] == 30.0
    assert stats["min"] == 10
    assert stats["max"] == 50
    assert stats["p5"] == round(float(np.percentile([10, 20, 30, 40, 50], 5)), 2)
    assert stats["p95"] == round(float(np.percentile([10, 20, 30, 40, 50], 95)), 2)

    empty_stats = compute_length_stats(pd.Series([], dtype=float))
    assert empty_stats["count"] == 0
    assert empty_stats["mean"] == 0.0


def test_compute_split_label_counts():
    df = pd.DataFrame(
        [
            {"split": "train", "label": 0},
            {"split": "train", "label": 1},
            {"split": "train", "label": 1},
            {"split": "val", "label": 0},
            {"split": "val", "label": 1},
            {"split": "test", "label": 0},
            {"split": "test", "label": 0},
        ]
    )
    res = compute_split_label_counts(df)
    assert res["train"]["human"] == 1
    assert res["train"]["ai"] == 2
    assert res["train"]["total"] == 3
    assert res["val"]["human"] == 1
    assert res["val"]["ai"] == 1
    assert res["test"]["human"] == 2
    assert res["test"]["ai"] == 0
    assert res["total"]["total"] == 7


def test_compute_exact_duplicates_and_cross_split_leakage():
    df = pd.DataFrame(
        [
            {"split": "train", "label": 0, "text": "Sample text alpha"},
            {
                "split": "train",
                "label": 0,
                "text": "Sample text alpha",
            },  # duplicate in train
            {"split": "train", "label": 1, "text": "Unique AI train sample"},
            {"split": "val", "label": 0, "text": "Sample text alpha"},  # leak into val
            {"split": "val", "label": 1, "text": "Unique AI val sample"},
            {
                "split": "test",
                "label": 0,
                "text": "Sample text alpha",
            },  # leak into test
            {"split": "test", "label": 1, "text": "Unique AI test sample"},
        ]
    )
    res = compute_exact_duplicates(df)
    assert res["total_duplicate_rows"] == 3
    assert res["unique_texts"] == 4
    assert res["within_split"]["train"] == 1
    assert res["train_val_leakage"] == 1
    assert res["train_test_leakage"] == 1
    assert res["by_label"]["human"]["train_val"] == 1
    assert res["by_label"]["ai"]["train_val"] == 0


def test_compute_normalized_duplicates():
    df = pd.DataFrame(
        [
            {"split": "train", "label": 0, "text": "Artificial Intelligence is Great!"},
            {"split": "val", "label": 0, "text": "artificial intelligence is great."},
            {"split": "test", "label": 1, "text": "Completely distinct content."},
        ]
    )
    res = compute_normalized_duplicates(df)
    assert res["train_val_leakage"] == 1
    assert res["train_test_leakage"] == 0
    assert res["by_label"]["human"]["train_val"] == 1


def test_compute_near_duplicates_jaccard():
    # Construct base text with 20 words
    base_words = [f"token{i}" for i in range(20)]
    train_text = " ".join(base_words)
    # Val text differs by only 1 word at end -> very high 3-gram Jaccard similarity (> 0.85)
    val_words = base_words[:19] + ["modifiedtoken"]
    val_text = " ".join(val_words)
    # Test text completely different
    test_text = "completely different sentence with totally distinct shingles"

    df = pd.DataFrame(
        [
            {"split": "train", "text": train_text},
            {"split": "val", "text": val_text},
            {"split": "test", "text": test_text},
        ]
    )
    res = compute_near_duplicates_jaccard(df, threshold=0.85)
    assert res["train_val_near_duplicates"] == 1
    assert res["train_test_near_duplicates"] == 0


def test_generate_markdown_report():
    audit_mock = {
        "manifest_path": "datasets/splits/text_manifest.csv",
        "total_rows": 100,
        "columns": ["split", "label", "generator", "domain", "text"],
        "has_generator": True,
        "has_domain": True,
        "has_attack": False,
        "has_dataset": False,
        "split_counts": {
            "train": {"human": 40, "ai": 40, "total": 80},
            "val": {"human": 5, "ai": 5, "total": 10},
            "test": {"human": 5, "ai": 5, "total": 10},
            "total": {"human": 50, "ai": 50, "total": 100},
        },
        "length_stats": {
            k: {
                "count": 10,
                "mean": 150.0,
                "median": 140.0,
                "p5": 50.0,
                "p95": 250.0,
                "min": 10,
                "max": 300,
                "std": 30.0,
            }
            for k in ["overall", "train", "val", "test", "human", "ai"]
        },
        "exact_duplicates": {
            "total_duplicate_rows": 2,
            "unique_texts": 98,
            "within_split": {"train": 1, "val": 0, "test": 0},
            "train_val_leakage": 1,
            "train_test_leakage": 0,
            "val_test_overlap": 0,
            "train_val_or_test_leakage": 1,
            "by_label": {
                "human": {"train_val": 1, "train_test": 0, "val_test": 0},
                "ai": {"train_val": 0, "train_test": 0, "val_test": 0},
            },
        },
        "normalized_duplicates": {
            "total_normalized_duplicate_rows": 2,
            "unique_normalized_texts": 98,
            "train_val_leakage": 1,
            "train_test_leakage": 0,
            "val_test_overlap": 0,
            "by_label": {
                "human": {"train_val": 1, "train_test": 0, "val_test": 0},
                "ai": {"train_val": 0, "train_test": 0, "val_test": 0},
            },
        },
        "near_duplicates": {
            "train_val_near_duplicates": 1,
            "train_test_near_duplicates": 0,
        },
        "jaccard_threshold": 0.85,
        "generator_counts": {"human": 50, "chatgpt": 50},
        "generator_by_split": {
            "train": {"human": 40, "chatgpt": 40},
            "val": {"human": 5, "chatgpt": 5},
            "test": {"human": 5, "chatgpt": 5},
        },
        "domain_counts": {"abstracts": 100},
        "domain_by_split": {
            "train": {"abstracts": 80},
            "val": {"abstracts": 10},
            "test": {"abstracts": 10},
        },
    }

    md = generate_markdown_report(audit_mock)
    assert "# Text Dataset Audit & Leakage Analysis" in md
    assert "Aryansh Agarwal" in md
    assert "chatgpt" in md
    assert "abstracts" in md
    assert "Train ↔ Validation Exact Leakage" in md
