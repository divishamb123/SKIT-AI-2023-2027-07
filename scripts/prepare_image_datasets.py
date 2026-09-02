"""
prepare_image_datasets.py — Sprint 1 (Data Preparation & Interface)

Purpose:
Builds deterministic, balanced, and reproducible train / validation / test splits 
for real and AI-generated image datasets (CIFAKE & GenImage holdout pools) and 
exports a unified manifest CSV.

Design Invariants:
1. CIFAKE (120,000 images):
   - train/ (100,000 images: 50,000 Real, 50,000 Fake) -> 90/10 stratified split
     -> Train split: 90,000 images (45,000 Real / 45,000 Fake)
     -> Validation split: 10,000 images (5,000 Real / 5,000 Fake)
   - test/ (20,000 images: 10,000 Real, 10,000 Fake) -> Kept intact as evaluation test split
2. GenImage Holdouts (12,000 images across multi-generator pools):
   - Never trained or validated on. Marked exclusively as 'holdout' to prevent data leakage.
   - Evaluates out-of-distribution generalization (Midjourney, BigGAN, Stable Diffusion, Real).
3. Determinism:
   - Fixed random seed (SEED = 42) ensures 100% byte-reproducible splits across environments.

Usage:
    python scripts/prepare_image_datasets.py
"""

import csv
import os
import random
from collections import Counter
from pathlib import Path

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
SEED = 42
VAL_FRACTION = 0.10  # 10% of CIFAKE train becomes validation

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CIFAKE_ROOT = PROJECT_ROOT / "datasets" / "cifake"
GENIMAGE_ROOT = PROJECT_ROOT / "datasets" / "genimage"
OUT_DIR = PROJECT_ROOT / "datasets" / "splits"
MANIFEST = OUT_DIR / "manifest.csv"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

GENIMAGE_POOLS = {
    "real_pool": (0, "real"),
    "sd_pool": (1, "stable-diffusion"),
    "mj_pool": (1, "midjourney"),
    "gan_pool": (1, "biggan"),
}

CIFAKE_LABELS = {"REAL": 0, "FAKE": 1}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def list_images(folder: Path) -> list[Path]:
    """Return all valid image files inside `folder`, sorted for determinism."""
    if not folder.is_dir():
        return []
    files = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]
    return sorted(files)


def rel(path: Path) -> str:
    """Return path relative to project root for platform portability."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------
# CIFAKE Processing
# --------------------------------------------------------------------------
def collect_cifake(rng: random.Random) -> list[dict]:
    rows: list[dict] = []

    # Process train split -> Stratified 90/10 into train / val
    for label_name, label in CIFAKE_LABELS.items():
        folder = CIFAKE_ROOT / "train" / label_name
        files = list_images(folder)
        if not files:
            print(f"  [WARN] No images found in: {folder}")
            continue

        shuffled = files[:]  # Copy to preserve sorted original
        rng.shuffle(shuffled)
        n_val = int(round(len(shuffled) * VAL_FRACTION))
        val_files = shuffled[:n_val]
        train_files = shuffled[n_val:]

        for f in train_files:
            rows.append({
                "dataset": "cifake",
                "split": "train",
                "label": label,
                "generator": "stable-diffusion-1.4" if label else "cifar10",
                "path": rel(f),
            })
        for f in val_files:
            rows.append({
                "dataset": "cifake",
                "split": "val",
                "label": label,
                "generator": "stable-diffusion-1.4" if label else "cifar10",
                "path": rel(f),
            })

    # Process test split -> Kept intact
    for label_name, label in CIFAKE_LABELS.items():
        folder = CIFAKE_ROOT / "test" / label_name
        files = list_images(folder)
        if not files:
            print(f"  [WARN] No images found in: {folder}")
            continue

        for f in files:
            rows.append({
                "dataset": "cifake",
                "split": "test",
                "label": label,
                "generator": "stable-diffusion-1.4" if label else "cifar10",
                "path": rel(f),
            })

    return rows


# --------------------------------------------------------------------------
# GenImage Holdout Processing
# --------------------------------------------------------------------------
def collect_genimage() -> list[dict]:
    rows: list[dict] = []
    for pool_folder, (label, generator) in GENIMAGE_POOLS.items():
        folder = GENIMAGE_ROOT / pool_folder
        files = list_images(folder)
        if not files:
            print(f"  [WARN] No images found in: {folder}")
            continue

        for f in files:
            rows.append({
                "dataset": "genimage",
                "split": "holdout",
                "label": label,
                "generator": generator,
                "path": rel(f),
            })
    return rows


# --------------------------------------------------------------------------
# Main Execution
# --------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Sprint 1 Data Preparation — Image Dataset Partitioning & Manifest")
    print(f"Author: Divisha Manak Bohra (23ESKCA038) | Random Seed: {SEED}")
    print("=" * 70)

    rng = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] Collecting and partitioning CIFAKE dataset...")
    cifake_rows = collect_cifake(rng)
    print(f"      Total CIFAKE records partitioned: {len(cifake_rows):,}")

    print("\n[2/3] Collecting GenImage holdout pools...")
    genimage_rows = collect_genimage()
    print(f"      Total GenImage holdout records: {len(genimage_rows):,}")

    all_rows = cifake_rows + genimage_rows
    if not all_rows:
        print("[ERROR] No image records were collected. Please check dataset directories.")
        return

    print(f"\n[3/3] Writing manifest CSV to: {MANIFEST}")
    fieldnames = ["dataset", "split", "label", "generator", "path"]
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"      Successfully saved {len(all_rows):,} rows.")

    # Print summary breakdown
    print("\n" + "=" * 70)
    print("DATASET PARTITION SUMMARY BREAKDOWN")
    print("=" * 70)

    split_counts = Counter(r["split"] for r in all_rows)
    for split, count in split_counts.most_common():
        real_cnt = sum(1 for r in all_rows if r["split"] == split and r["label"] == 0)
        fake_cnt = sum(1 for r in all_rows if r["split"] == split and r["label"] == 1)
        print(f"  * Split: {split:<10} Total: {count:>8,} | Real (0): {real_cnt:>7,} | Fake (1): {fake_cnt:>7,}")

    print("\nGenerator Distribution:")
    gen_counts = Counter(r["generator"] for r in all_rows)
    for gen, count in gen_counts.most_common():
        print(f"  * {gen:<25}: {count:>8,}")

    print("=" * 70)
    print("Sprint 1 Data Preparation completed successfully.\n")


if __name__ == "__main__":
    main()
