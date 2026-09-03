"""
prepare_text_dataset.py — Sprint 1 (Data Preparation & Interface)

Purpose:
Downloads a deterministic, balanced, and reproducible train/validation/test split
from the Robust AI Detection (RAID) benchmark on Hugging Face. Streams the data
to avoid downloading the entire 6 million record corpus.

Design Invariants:
1. Dataset: `liamdugan/raid`
2. Subset Size: SAMPLES_PER_CLASS (default 10,000 Human / 10,000 AI)
3. Labels: Human = 0, AI = 1
4. Splits: 80% Train, 10% Val, 10% Test (stratified by label)
5. Generator Diversity: AI samples are explicitly sampled evenly from available generator subtypes.
6. Determinism: Fixed random seed (SEED = 42), explicit sorting of IDs before sampling
   ensures 100% reproducible splits across environments.
7. Manifest Output: Saves raw text directly into a unified CSV manifest to prevent
   filesystem bloat (avoiding hundreds of thousands of small .txt files).

Usage:
    python scripts/prepare_text_dataset.py
"""

import csv
import sys
import random
from collections import defaultdict, Counter
from pathlib import Path

# Try importing datasets, fail gracefully
try:
    from datasets import load_dataset
except ImportError:
    print("[ERROR] Hugging Face 'datasets' is not installed. Please pip install datasets.")
    sys.exit(1)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
SEED = 42
SAMPLES_PER_CLASS = 10000  # Total = 20,000 rows
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1

HF_DATASET_ID = "liamdugan/raid"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "datasets" / "splits"
MANIFEST_PATH = OUT_DIR / "text_manifest.csv"

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def get_split_indices(total: int) -> tuple[int, int]:
    """Returns the (val_start, test_start) indices for an 80/10/10 split."""
    train_end = int(round(total * TRAIN_FRAC))
    val_end = train_end + int(round(total * VAL_FRAC))
    return train_end, val_end

# --------------------------------------------------------------------------
# Dataset Loading & Buffering (Streaming)
# --------------------------------------------------------------------------
def stream_and_buffer_raid(target_per_class: int):
    """
    Streams the dataset and buffers a sufficient amount of Human and AI records
    across multiple generators to allow for diverse, balanced sampling.
    """
    print(f"      Connecting to HF Hub ({HF_DATASET_ID}) in streaming mode...")
    try:
        # Load the train split in streaming mode
        dataset = load_dataset(HF_DATASET_ID, split="train", streaming=True)
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        sys.exit(1)

    human_buffer = []
    ai_buffer_by_gen = defaultdict(list)
    
    # We want a buffer somewhat larger than target to ensure we can sort & sample
    # deterministically. We cap per generator to avoid OOM and ensure diversity.
    buffer_multiplier = 2
    max_human = target_per_class * buffer_multiplier
    max_per_ai_gen = max(1000, (target_per_class * buffer_multiplier) // 4) 

    print("      Buffering records from stream (this may take a minute)...")
    
    count = 0
    for row in dataset:
        count += 1
        
        # Required fields validation
        if not row.get("generation") or not str(row["generation"]).strip():
            continue
        if not row.get("model"):
            continue
            
        model = str(row["model"]).strip().lower()
        
        # Prepare minimal dict for memory efficiency
        record = {
            "id": str(row.get("id", f"idx-{count}")),
            "dataset": "raid",
            "model": model,
            "domain": str(row.get("domain", "unknown")),
            "attack": str(row.get("attack", "none")),
            "text": str(row["generation"]).strip()
        }
        
        if model == "human":
            if len(human_buffer) < max_human:
                human_buffer.append(record)
        else:
            if len(ai_buffer_by_gen[model]) < max_per_ai_gen:
                ai_buffer_by_gen[model].append(record)
                
        # Check if we have buffered enough
        ai_total_buffered = sum(len(v) for v in ai_buffer_by_gen.values())
        if len(human_buffer) >= max_human and ai_total_buffered >= (target_per_class * buffer_multiplier):
            break
            
        if count % 50000 == 0:
            print(f"        Scanned {count:,} records... (Human: {len(human_buffer):,}, AI: {ai_total_buffered:,})")
            
    return human_buffer, ai_buffer_by_gen

# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------
def deterministic_sample(human_buffer: list, ai_buffer_by_gen: dict, rng: random.Random, target_per_class: int):
    """
    Sorts the buffers deterministically by ID, then uses the RNG to sample
    the target counts. Balances AI sampling evenly across available generators.
    """
    # 1. Sample Human records
    human_buffer.sort(key=lambda x: x["id"])
    rng.shuffle(human_buffer)
    
    if len(human_buffer) < target_per_class:
        print(f"[ERROR] Insufficient human records. Found {len(human_buffer)}, needed {target_per_class}.")
        sys.exit(1)
        
    human_sample = human_buffer[:target_per_class]
    
    # 2. Sample AI records (stratified by generator roughly)
    ai_generators = sorted(list(ai_buffer_by_gen.keys()))
    if not ai_generators:
        print("[ERROR] No AI generators found in dataset.")
        sys.exit(1)
        
    ai_sample = []
    # Distribute quota roughly evenly across available generators
    target_per_gen = target_per_class // len(ai_generators)
    remainder = target_per_class % len(ai_generators)
    
    for i, gen in enumerate(ai_generators):
        gen_list = ai_buffer_by_gen[gen]
        gen_list.sort(key=lambda x: x["id"])
        rng.shuffle(gen_list)
        
        quota = target_per_gen + (1 if i < remainder else 0)
        # If a generator doesn't have enough, we just take what it has (might slightly under-sample AI class, 
        # but with raid buffer sizes it's extremely unlikely)
        ai_sample.extend(gen_list[:quota])
        
    # If we fell short because some generators had too few (unlikely), fill from others
    if len(ai_sample) < target_per_class:
        deficit = target_per_class - len(ai_sample)
        pool = []
        for gen in ai_generators:
            used = target_per_gen + (1 if ai_generators.index(gen) < remainder else 0)
            pool.extend(ai_buffer_by_gen[gen][used:])
        pool.sort(key=lambda x: x["id"])
        rng.shuffle(pool)
        ai_sample.extend(pool[:deficit])
        
    if len(ai_sample) < target_per_class:
        print(f"[ERROR] Insufficient AI records. Found {len(ai_sample)}, needed {target_per_class}.")
        sys.exit(1)
        
    return human_sample, ai_sample

# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------
def construct_splits(human_sample: list, ai_sample: list):
    """
    Assigns splits (train, val, test) and labels (0=human, 1=ai).
    """
    all_rows = []
    
    # Process Human (Label 0)
    train_end, val_end = get_split_indices(len(human_sample))
    for i, record in enumerate(human_sample):
        if i < train_end:
            split = "train"
        elif i < val_end:
            split = "val"
        else:
            split = "test"
            
        all_rows.append({
            "dataset": record["dataset"],
            "split": split,
            "label": 0,
            "generator": record["model"],
            "domain": record["domain"],
            "attack": record["attack"],
            "id": record["id"],
            "text": record["text"]
        })

    # Process AI (Label 1)
    train_end, val_end = get_split_indices(len(ai_sample))
    for i, record in enumerate(ai_sample):
        if i < train_end:
            split = "train"
        elif i < val_end:
            split = "val"
        else:
            split = "test"
            
        all_rows.append({
            "dataset": record["dataset"],
            "split": split,
            "label": 1,
            "generator": record["model"],
            "domain": record["domain"],
            "attack": record["attack"],
            "id": record["id"],
            "text": record["text"]
        })
        
    return all_rows

# --------------------------------------------------------------------------
# Main Execution
# --------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Sprint 1 Data Preparation — NLP Dataset Partitioning & Manifest")
    print(f"Author: Aryansh Agarwal (23ESKCA021) | Random Seed: {SEED}")
    print("=" * 70)

    # Initialize deterministic RNG
    rng = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load and Buffer
    print(f"\n[1/3] Loading / sampling RAID dataset (Target: {SAMPLES_PER_CLASS:,} per class)...")
    human_buffer, ai_buffer_by_gen = stream_and_buffer_raid(SAMPLES_PER_CLASS)
    
    # 2. Sample and Split
    print("\n[2/3] Balancing and splitting dataset...")
    human_sample, ai_sample = deterministic_sample(human_buffer, ai_buffer_by_gen, rng, SAMPLES_PER_CLASS)
    manifest_rows = construct_splits(human_sample, ai_sample)
    
    # 3. Write Manifest
    print(f"\n[3/3] Writing text manifest to: {MANIFEST_PATH}")
    fieldnames = ["dataset", "split", "label", "generator", "domain", "attack", "id", "text"]
    
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
        
    print(f"      Successfully saved {len(manifest_rows):,} rows.")

    # ----------------------------------------------------------------------
    # Validation & Summary
    # ----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DATASET PARTITION SUMMARY BREAKDOWN")
    print("=" * 70)
    
    # Class distribution
    total = len(manifest_rows)
    human_cnt = sum(1 for r in manifest_rows if r["label"] == 0)
    ai_cnt = sum(1 for r in manifest_rows if r["label"] == 1)
    print(f"Total Records: {total:,}")
    print(f"  * Human (0): {human_cnt:>7,} ({human_cnt/total*100:.1f}%)")
    print(f"  * AI (1):    {ai_cnt:>7,} ({ai_cnt/total*100:.1f}%)")
    
    # Split distribution
    print("\nSplit Distribution:")
    split_counts = Counter(r["split"] for r in manifest_rows)
    for split in ["train", "val", "test"]:
        count = split_counts.get(split, 0)
        split_human = sum(1 for r in manifest_rows if r["split"] == split and r["label"] == 0)
        split_ai = sum(1 for r in manifest_rows if r["split"] == split and r["label"] == 1)
        print(f"  * {split:<10}: {count:>8,} | Human: {split_human:>7,} | AI: {split_ai:>7,}")
        
    # Generator distribution
    print("\nGenerator Distribution (AI Only):")
    gen_counts = Counter(r["generator"] for r in manifest_rows if r["label"] == 1)
    for gen, count in gen_counts.most_common():
        print(f"  * {gen:<25}: {count:>8,}")

    # Validate empty texts
    empty_texts = sum(1 for r in manifest_rows if not r["text"].strip())
    if empty_texts > 0:
        print(f"\n[WARNING] Found {empty_texts} records with empty text!")

    print("=" * 70)
    print("Sprint 1 NLP Data Preparation completed successfully.\n")

if __name__ == "__main__":
    main()
