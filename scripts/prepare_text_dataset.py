"""
prepare_text_dataset.py — Sprint 1 & Sprint 3 Part 2: NLP Dataset Preparation & Unseen Holdout Partitioning

Purpose:
Downloads a deterministic, balanced, and reproducible train/validation/test split
from the Robust AI Detection (RAID) benchmark on Hugging Face, or repartitions an
existing manifest with an honest, out-of-domain holdout split (--holdout-generator / --holdout-domain).

Sprint 3 Part 2 Capabilities:
- Adds --holdout-generator: isolates an entire AI generator (e.g. 'chatgpt' or 'mistral')
  from train and val splits, saving it to a separate unseen test manifest (text_holdout_test.csv)
  paired with clean human controls.
- Adds --holdout-domain: checks and isolates a domain; if the manifest contains only a single
  domain ('abstracts'), reports plainly and recommends generator holdout.
- Adds --input-manifest: allows fast, offline repartitioning directly from an existing manifest.
- Preserves 100% of the default behavior when run without holdout arguments.

Usage:
    python scripts/prepare_text_dataset.py
    python scripts/prepare_text_dataset.py --holdout-generator chatgpt
    python scripts/prepare_text_dataset.py --holdout-generator mistral --holdout-output datasets/splits/text_holdout_test.csv
"""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

# Try importing datasets, fail gracefully when streaming from HF is required
try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

# --------------------------------------------------------------------------
# Configuration & Constants
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
DEFAULT_HOLDOUT_PATH = OUT_DIR / "text_holdout_test.csv"

FIELDNAMES = [
    "dataset",
    "split",
    "label",
    "generator",
    "domain",
    "attack",
    "id",
    "text",
]


# --------------------------------------------------------------------------
# CLI Argument Parser
# --------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare balanced text detection dataset with optional unseen holdout splits."
    )
    parser.add_argument(
        "--holdout-generator",
        type=str,
        default=None,
        help="AI generator to exclude completely from train/val and write to unseen test set (e.g., 'chatgpt').",
    )
    parser.add_argument(
        "--holdout-domain",
        type=str,
        default=None,
        help="Domain to exclude from train/val and write to unseen test set.",
    )
    parser.add_argument(
        "--input-manifest",
        type=str,
        default=None,
        help="Path to existing text manifest CSV to repartition without streaming from Hugging Face.",
    )
    parser.add_argument(
        "--output-manifest",
        type=str,
        default=str(MANIFEST_PATH),
        help=f"Path to write main in-domain manifest CSV (default: {MANIFEST_PATH}).",
    )
    parser.add_argument(
        "--holdout-output",
        type=str,
        default=str(DEFAULT_HOLDOUT_PATH),
        help=f"Path to write unseen holdout test manifest CSV (default: {DEFAULT_HOLDOUT_PATH}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed for deterministic sampling and shuffling (default: {SEED}).",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=SAMPLES_PER_CLASS,
        help=f"Target samples per class when streaming from HF (default: {SAMPLES_PER_CLASS}).",
    )
    return parser.parse_args(argv)


# --------------------------------------------------------------------------
# Helpers & File I/O
# --------------------------------------------------------------------------
def get_split_indices(total: int) -> Tuple[int, int]:
    """Returns the (val_start, test_start) indices for an 80/10/10 split."""
    train_end = int(round(total * TRAIN_FRAC))
    val_end = train_end + int(round(total * VAL_FRAC))
    return train_end, val_end


def read_manifest_csv(path: Path | str) -> List[Dict[str, Any]]:
    """Reads manifest CSV and returns rows with integer labels."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {path}")

    rows: List[Dict[str, Any]] = []
    with open(path, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["label"] = int(r["label"])
            rows.append(r)
    return rows


def write_manifest_csv(
    path: Path | str, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None
) -> None:
    """Writes list of row dictionaries to CSV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = FIELDNAMES

    with open(path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------
# Dataset Loading & Buffering (Streaming from HF)
# --------------------------------------------------------------------------
def stream_and_buffer_raid(target_per_class: int):
    """
    Streams the dataset and buffers a sufficient amount of Human and AI records
    across multiple generators to allow for diverse, balanced sampling.
    """
    if load_dataset is None:
        print(
            "[ERROR] Hugging Face 'datasets' is not installed. Please pip install datasets."
        )
        sys.exit(1)

    print(f"      Connecting to HF Hub ({HF_DATASET_ID}) in streaming mode...")
    try:
        dataset = load_dataset(HF_DATASET_ID, split="train", streaming=True)
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        sys.exit(1)

    human_buffer = []
    ai_buffer_by_gen = defaultdict(list)

    buffer_multiplier = 2
    max_human = target_per_class * buffer_multiplier
    max_per_ai_gen = max(1000, (target_per_class * buffer_multiplier) // 4)

    print("      Buffering records from stream (this may take a minute)...")

    count = 0
    for row in dataset:
        count += 1
        if not row.get("generation") or not str(row["generation"]).strip():
            continue
        if not row.get("model"):
            continue

        model = str(row["model"]).strip().lower()

        record = {
            "id": str(row.get("id", f"idx-{count}")),
            "dataset": "raid",
            "model": model,
            "domain": str(row.get("domain", "unknown")),
            "attack": str(row.get("attack", "none")),
            "text": str(row["generation"]).strip(),
        }

        if model == "human":
            if len(human_buffer) < max_human:
                human_buffer.append(record)
        else:
            if len(ai_buffer_by_gen[model]) < max_per_ai_gen:
                ai_buffer_by_gen[model].append(record)

        ai_total_buffered = sum(len(v) for v in ai_buffer_by_gen.values())
        if len(human_buffer) >= max_human and ai_total_buffered >= (
            target_per_class * buffer_multiplier
        ):
            break

        if count % 50000 == 0:
            print(
                f"        Scanned {count:,} records... (Human: {len(human_buffer):,}, AI: {ai_total_buffered:,})"
            )

    return human_buffer, ai_buffer_by_gen


# --------------------------------------------------------------------------
# Sampling (Sprint 1 Default)
# --------------------------------------------------------------------------
def deterministic_sample(
    human_buffer: list,
    ai_buffer_by_gen: dict,
    rng: random.Random,
    target_per_class: int,
):
    """
    Sorts the buffers deterministically by ID, then uses the RNG to sample
    the target counts. Balances AI sampling evenly across available generators.
    """
    human_buffer.sort(key=lambda x: x["id"])
    rng.shuffle(human_buffer)

    if len(human_buffer) < target_per_class:
        print(
            f"[ERROR] Insufficient human records. Found {len(human_buffer)}, needed {target_per_class}."
        )
        sys.exit(1)

    human_sample = human_buffer[:target_per_class]

    ai_generators = sorted(list(ai_buffer_by_gen.keys()))
    if not ai_generators:
        print("[ERROR] No AI generators found in dataset.")
        sys.exit(1)

    ai_sample = []
    target_per_gen = target_per_class // len(ai_generators)
    remainder = target_per_class % len(ai_generators)

    for i, gen in enumerate(ai_generators):
        gen_list = ai_buffer_by_gen[gen]
        gen_list.sort(key=lambda x: x["id"])
        rng.shuffle(gen_list)

        quota = target_per_gen + (1 if i < remainder else 0)
        ai_sample.extend(gen_list[:quota])

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
        print(
            f"[ERROR] Insufficient AI records. Found {len(ai_sample)}, needed {target_per_class}."
        )
        sys.exit(1)

    return human_sample, ai_sample


# --------------------------------------------------------------------------
# Default Splitting (Sprint 1 Invariant)
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

        all_rows.append(
            {
                "dataset": record["dataset"],
                "split": split,
                "label": 0,
                "generator": record["model"],
                "domain": record["domain"],
                "attack": record["attack"],
                "id": record["id"],
                "text": record["text"],
            }
        )

    # Process AI (Label 1)
    train_end, val_end = get_split_indices(len(ai_sample))
    for i, record in enumerate(ai_sample):
        if i < train_end:
            split = "train"
        elif i < val_end:
            split = "val"
        else:
            split = "test"

        all_rows.append(
            {
                "dataset": record["dataset"],
                "split": split,
                "label": 1,
                "generator": record["model"],
                "domain": record["domain"],
                "attack": record["attack"],
                "id": record["id"],
                "text": record["text"],
            }
        )

    return all_rows


# --------------------------------------------------------------------------
# Sprint 3 Part 2: Unseen Holdout Partitioning
# --------------------------------------------------------------------------
def partition_manifest_holdout(
    rows: List[Dict[str, Any]],
    holdout_generator: Optional[str] = None,
    holdout_domain: Optional[str] = None,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    test_frac: float = TEST_FRAC,
    seed: int = SEED,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Partitions rows into an in-domain dataset and an unseen holdout test dataset.

    Rules & Invariants:
    1. If neither holdout_generator nor holdout_domain is provided, returns (rows, []).
    2. If holdout_domain is specified:
       - Checks available domains in rows.
       - If only 1 domain exists ('abstracts'), raises ValueError with clear explanation
         and recommendation to use --holdout-generator instead.
    3. If holdout_generator is specified:
       - Normalizes name and verifies presence in AI records (label=1).
       - Extracts all AI records matching the generator into holdout_rows with split='test'.
       - Deterministically samples an equal count of human records (label=0) for holdout_rows
         with split='test' to guarantee 50/50 class balance.
       - Stratifies remaining AI generators and remaining Human records across train, val,
         and in-domain test splits according to train_frac / val_frac / test_frac.
       - Guarantees zero leakage: the held-out generator has 0 records in train or val.
    """
    if not holdout_generator and not holdout_domain:
        return rows, []

    rng = random.Random(seed)

    # Domain check
    if holdout_domain:
        target_domain = holdout_domain.strip().lower()
        unique_domains = sorted(
            list(set(str(r.get("domain", "")).strip().lower() for r in rows))
        )
        if len(unique_domains) <= 1:
            raise ValueError(
                f"Cannot hold out domain '{holdout_domain}': the manifest contains only a single domain "
                f"({unique_domains}). Excluding this domain would leave zero training samples. "
                f"Recommendation: hold out an AI generator using --holdout-generator (e.g. 'chatgpt' or 'mistral') instead."
            )
        if target_domain not in unique_domains:
            raise ValueError(
                f"Holdout domain '{holdout_domain}' not found in manifest. Available domains: {unique_domains}"
            )

    if not holdout_generator:
        # If domain holdout on multi-domain dataset was requested
        target_domain = holdout_domain.strip().lower()
        holdout_rows = [
            dict(r, split="test")
            for r in rows
            if str(r.get("domain", "")).strip().lower() == target_domain
        ]
        in_rows = [
            r for r in rows if str(r.get("domain", "")).strip().lower() != target_domain
        ]
        return in_rows, holdout_rows

    # Generator holdout processing
    target_gen = holdout_generator.strip().lower()
    ai_gens = sorted(
        list(
            set(
                str(r.get("generator", "")).strip().lower()
                for r in rows
                if int(r.get("label", 0)) == 1
            )
        )
    )

    if target_gen not in ai_gens:
        raise ValueError(
            f"Holdout generator '{holdout_generator}' not found in manifest AI records. "
            f"Available AI generators: {ai_gens}"
        )

    # 1. Separate Holdout AI rows
    holdout_ai = [
        dict(r, split="test")
        for r in rows
        if int(r.get("label", 0)) == 1
        and str(r.get("generator", "")).strip().lower() == target_gen
    ]
    holdout_count = len(holdout_ai)

    # 2. Separate all Human rows and reserve holdout_count for holdout set
    all_human = [r for r in rows if int(r.get("label", 0)) == 0]
    all_human_sorted = sorted(all_human, key=lambda x: str(x.get("id", "")))
    rng.shuffle(all_human_sorted)

    if len(all_human_sorted) < holdout_count:
        raise ValueError(
            f"Insufficient human records ({len(all_human_sorted)}) to match holdout count ({holdout_count})."
        )

    holdout_human = [dict(r, split="test") for r in all_human_sorted[:holdout_count]]
    remaining_human = all_human_sorted[holdout_count:]

    # 3. Separate remaining AI rows
    remaining_ai = [
        r
        for r in rows
        if int(r.get("label", 0)) == 1
        and str(r.get("generator", "")).strip().lower() != target_gen
    ]

    # Combine holdout set (50% Human, 50% AI)
    holdout_rows = holdout_human + holdout_ai
    holdout_rows.sort(key=lambda x: str(x.get("id", "")))
    rng.shuffle(holdout_rows)

    # 4. Form in-domain splits (train, val, test) with balanced classes
    target_in_per_class = min(len(remaining_human), len(remaining_ai))
    in_human_pool = remaining_human[:target_in_per_class]

    # Partition Human into train, val, test
    train_end_h = int(round(target_in_per_class * train_frac))
    val_end_h = train_end_h + int(round(target_in_per_class * val_frac))

    main_rows: List[Dict[str, Any]] = []
    for i, r in enumerate(in_human_pool):
        if i < train_end_h:
            sp = "train"
        elif i < val_end_h:
            sp = "val"
        else:
            sp = "test"
        main_rows.append(dict(r, split=sp))

    # Partition remaining AI generators proportionally across train, val, test
    # to avoid the Sprint 1 generator-slice allocation flaw
    remaining_ai_by_gen = defaultdict(list)
    for r in remaining_ai:
        gen = str(r.get("generator", "")).strip().lower()
        remaining_ai_by_gen[gen].append(r)

    ai_in_rows: List[Dict[str, Any]] = []
    for gen, gen_records in sorted(remaining_ai_by_gen.items()):
        gen_records.sort(key=lambda x: str(x.get("id", "")))
        rng.shuffle(gen_records)

        tr_end = int(round(len(gen_records) * train_frac))
        va_end = tr_end + int(round(len(gen_records) * val_frac))

        for j, rec in enumerate(gen_records):
            if j < tr_end:
                sp = "train"
            elif j < va_end:
                sp = "val"
            else:
                sp = "test"
            ai_in_rows.append(dict(rec, split=sp))

    # Match AI count to Human count if needed
    if len(ai_in_rows) > target_in_per_class:
        ai_in_rows = ai_in_rows[:target_in_per_class]

    main_rows.extend(ai_in_rows)
    main_rows.sort(key=lambda x: str(x.get("id", "")))
    rng.shuffle(main_rows)

    return main_rows, holdout_rows


# --------------------------------------------------------------------------
# Main Execution
# --------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)

    print("=" * 70)
    print("Sprint 3 Part 2: NLP Dataset Preparation & Unseen Holdout Partitioning")
    print(f"Author: Aryansh Agarwal (23ESKCA021) | Random Seed: {args.seed}")
    print("=" * 70)

    rng = random.Random(args.seed)
    out_dir = Path(args.output_manifest).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    input_manifest_path = Path(args.input_manifest) if args.input_manifest else None
    is_holdout_requested = bool(args.holdout_generator or args.holdout_domain)

    # 1. Acquire raw / base records
    if input_manifest_path and input_manifest_path.is_file():
        print(f"\n[1/3] Loading input manifest: {input_manifest_path}")
        raw_rows = read_manifest_csv(input_manifest_path)
    elif is_holdout_requested and MANIFEST_PATH.is_file():
        print(
            f"\n[1/3] Using existing manifest at {MANIFEST_PATH} for holdout partitioning..."
        )
        raw_rows = read_manifest_csv(MANIFEST_PATH)
    else:
        # Default behavior: stream from Hugging Face
        print(
            f"\n[1/3] Streaming RAID dataset (Target: {args.samples_per_class:,} per class)..."
        )
        human_buf, ai_buf = stream_and_buffer_raid(args.samples_per_class)
        human_samp, ai_samp = deterministic_sample(
            human_buf, ai_buf, rng, args.samples_per_class
        )
        raw_rows = construct_splits(human_samp, ai_samp)

    # 2. Partition with holdout if requested
    print("\n[2/3] Partitioning splits and holdout sets...")
    main_rows, holdout_rows = partition_manifest_holdout(
        raw_rows,
        holdout_generator=args.holdout_generator,
        holdout_domain=args.holdout_domain,
        train_frac=TRAIN_FRAC,
        val_frac=VAL_FRAC,
        test_frac=TEST_FRAC,
        seed=args.seed,
    )

    # 3. Write Manifests
    out_manifest = Path(args.output_manifest)
    print(f"\n[3/3] Writing main manifest to: {out_manifest}")
    write_manifest_csv(out_manifest, main_rows)
    print(f"      Successfully saved {len(main_rows):,} rows.")

    if holdout_rows:
        holdout_file = Path(args.holdout_output)
        print(f"      Writing unseen holdout test manifest to: {holdout_file}")
        write_manifest_csv(holdout_file, holdout_rows)
        print(f"      Successfully saved {len(holdout_rows):,} holdout rows.")

    # ----------------------------------------------------------------------
    # Partition Breakdown Summary
    # ----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DATASET PARTITION SUMMARY BREAKDOWN")
    print("=" * 70)

    total_main = len(main_rows)
    human_cnt = sum(1 for r in main_rows if int(r["label"]) == 0)
    ai_cnt = sum(1 for r in main_rows if int(r["label"]) == 1)
    print(f"Main Manifest Records: {total_main:,}")
    print(
        f"  * Human (0): {human_cnt:>7,} ({human_cnt/total_main*100:.1f}%)"
        if total_main
        else ""
    )
    print(
        f"  * AI (1):    {ai_cnt:>7,} ({ai_cnt/total_main*100:.1f}%)"
        if total_main
        else ""
    )

    print("\nMain Manifest Split Distribution:")
    split_counts = Counter(r["split"] for r in main_rows)
    for sp in ["train", "val", "test"]:
        cnt = split_counts.get(sp, 0)
        h = sum(1 for r in main_rows if r["split"] == sp and int(r["label"]) == 0)
        a = sum(1 for r in main_rows if r["split"] == sp and int(r["label"]) == 1)
        print(f"  * {sp:<10}: {cnt:>8,} | Human: {h:>7,} | AI: {a:>7,}")

    if holdout_rows:
        total_ho = len(holdout_rows)
        ho_h = sum(1 for r in holdout_rows if int(r["label"]) == 0)
        ho_a = sum(1 for r in holdout_rows if int(r["label"]) == 1)
        print("\nUnseen Holdout Test Set:")
        print(f"  * Total Holdout Rows: {total_ho:,} (split='test')")
        print(f"  * Human Control (0):  {ho_h:>7,} ({ho_h/total_ho*100:.1f}%)")
        print(f"  * Held-out AI (1):    {ho_a:>7,} ({ho_a/total_ho*100:.1f}%)")
        ho_gens = Counter(r["generator"] for r in holdout_rows if int(r["label"]) == 1)
        print(f"  * Held-out AI Generator(s): {dict(ho_gens.most_common())}")

    print("=" * 70)
    print("NLP Data Preparation & Holdout Partition completed successfully.\n")


if __name__ == "__main__":
    main()
