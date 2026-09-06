"""
test_dataset_splits.py — Automated Unit & Invariant Tests for Sprint 1 Data Preparation
Author: Divisha Manak Bohra (23ESKCA038)
"""

import csv
from collections import Counter
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "datasets" / "splits" / "manifest.csv"


@pytest.fixture(scope="module")
def manifest_rows():
    assert MANIFEST_PATH.is_file(), f"Manifest file missing at {MANIFEST_PATH}"
    rows = []
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def test_manifest_schema_and_size(manifest_rows):
    assert len(manifest_rows) == 132000, f"Expected 132,000 records, found {len(manifest_rows)}"
    required_cols = {"dataset", "split", "label", "generator", "path"}
    for r in manifest_rows[:50]:
        assert required_cols.issubset(set(r.keys())), "Missing required column headers in row"


def test_cifake_train_val_test_counts(manifest_rows):
    cifake_train = [r for r in manifest_rows if r["dataset"] == "cifake" and r["split"] == "train"]
    cifake_val = [r for r in manifest_rows if r["dataset"] == "cifake" and r["split"] == "val"]
    cifake_test = [r for r in manifest_rows if r["dataset"] == "cifake" and r["split"] == "test"]

    assert len(cifake_train) == 90000, f"Train split count should be 90k, got {len(cifake_train)}"
    assert len(cifake_val) == 10000, f"Validation split count should be 10k, got {len(cifake_val)}"
    assert len(cifake_test) == 20000, f"Test split count should be 20k, got {len(cifake_test)}"


def test_strict_50_50_class_balancing(manifest_rows):
    for split_name in ["train", "val", "test"]:
        sub = [r for r in manifest_rows if r["split"] == split_name]
        reals = sum(1 for r in sub if r["label"] == "0")
        fakes = sum(1 for r in sub if r["label"] == "1")
        assert reals == fakes, f"Split '{split_name}' is unbalanced! Real: {reals}, Fake: {fakes}"


def test_zero_data_leakage(manifest_rows):
    splits = {"train", "val", "test", "holdout"}
    paths_by_split = {s: set(r["path"] for r in manifest_rows if r["split"] == s) for s in splits}

    for s1 in splits:
        for s2 in splits:
            if s1 != s2:
                overlap = paths_by_split[s1].intersection(paths_by_split[s2])
                assert len(overlap) == 0, f"Data leakage between {s1} and {s2}: {len(overlap)} duplicate paths"


def test_genimage_holdout_isolation(manifest_rows):
    genimage_rows = [r for r in manifest_rows if r["dataset"] == "genimage"]
    assert len(genimage_rows) == 12000, f"Expected 12k GenImage records, got {len(genimage_rows)}"
    for r in genimage_rows:
        assert r["split"] == "holdout", f"GenImage image marked as '{r['split']}' instead of 'holdout'"


def test_path_resolution(manifest_rows):
    # Sample test paths
    for r in manifest_rows[::4000]:  # Every 4000th row
        p = PROJECT_ROOT / r["path"]
        assert p.is_file(), f"Path in manifest not found on disk: {p}"
