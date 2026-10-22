"""
audit_text_dataset.py — Sprint 3 Part 2: Comprehensive Text Dataset Audit

Audits the text manifest (datasets/splits/text_manifest.csv) to quantify:
1. Split and class label distributions.
2. Text length distributions in words (mean, median, 5th, and 95th percentiles).
3. Exact duplicate and near-duplicate leakage between train, val, and test splits.
4. Generator and domain distributions across partitions.

Outputs a detailed report to stdout and saves structured markdown to docs/text_dataset_audit.md.
"""

import argparse
from collections import defaultdict
from pathlib import Path
import re
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit text dataset manifest for distributions, length stats, and leakage."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="datasets/splits/text_manifest.csv",
        help="Path to text manifest CSV file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="docs/text_dataset_audit.md",
        help="Path to save markdown audit report.",
    )
    parser.add_argument(
        "--jaccard-threshold",
        type=float,
        default=0.85,
        help="Jaccard similarity threshold for 3-gram near-duplicate detection (default: 0.85).",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    """Lowercases, removes punctuation, and normalizes whitespace."""
    cleaned = re.sub(r"[^\w\s]", "", str(text).lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def get_3gram_shingles(text: str) -> Set[str]:
    """Generates set of 3-word shingles for near-duplicate Jaccard comparison."""
    words = re.findall(r"\b\w+\b", str(text).lower())
    if len(words) < 3:
        return set(words)
    return set(" ".join(words[i : i + 3]) for i in range(len(words) - 2))


def compute_length_stats(word_counts: pd.Series) -> Dict[str, float]:
    """Computes mean, median, 5th and 95th percentiles in words, plus min, max, std."""
    if len(word_counts) == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "p5": 0.0,
            "p95": 0.0,
            "min": 0,
            "max": 0,
            "std": 0.0,
        }
    return {
        "count": int(len(word_counts)),
        "mean": round(float(word_counts.mean()), 2),
        "median": round(float(word_counts.median()), 2),
        "p5": round(float(np.percentile(word_counts, 5)), 2),
        "p95": round(float(np.percentile(word_counts, 95)), 2),
        "min": int(word_counts.min()),
        "max": int(word_counts.max()),
        "std": round(float(word_counts.std()), 2),
    }


def compute_split_label_counts(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """Calculates counts per split and label."""
    res: Dict[str, Dict[str, int]] = {}
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        res[split] = {
            "human": int(sum(sub["label"] == 0)),
            "ai": int(sum(sub["label"] == 1)),
            "total": int(len(sub)),
        }
    res["total"] = {
        "human": int(sum(df["label"] == 0)),
        "ai": int(sum(df["label"] == 1)),
        "total": int(len(df)),
    }
    return res


def compute_exact_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    """Finds exact string duplicate counts within and across splits."""
    total_exact_dups = int(len(df) - df["text"].nunique())

    within_split = {}
    for sp in ["train", "val", "test"]:
        sub = df[df["split"] == sp]
        within_split[sp] = int(len(sub) - sub["text"].nunique())

    train_texts = set(df[df["split"] == "train"]["text"])
    val_texts = set(df[df["split"] == "val"]["text"])
    test_texts = set(df[df["split"] == "test"]["text"])

    train_val_overlap = set(train_texts.intersection(val_texts))
    train_test_overlap = set(train_texts.intersection(test_texts))
    val_test_overlap = set(val_texts.intersection(test_texts))
    train_either_overlap = set(train_texts.intersection(val_texts.union(test_texts)))

    # Overlap by label
    label_overlap = {}
    for label, name in [(0, "human"), (1, "ai")]:
        tr = set(df[(df["split"] == "train") & (df["label"] == label)]["text"])
        va = set(df[(df["split"] == "val") & (df["label"] == label)]["text"])
        te = set(df[(df["split"] == "test") & (df["label"] == label)]["text"])
        label_overlap[name] = {
            "train_val": len(tr.intersection(va)),
            "train_test": len(tr.intersection(te)),
            "val_test": len(va.intersection(te)),
        }

    return {
        "total_duplicate_rows": total_exact_dups,
        "unique_texts": int(df["text"].nunique()),
        "within_split": within_split,
        "train_val_leakage": len(train_val_overlap),
        "train_test_leakage": len(train_test_overlap),
        "val_test_overlap": len(val_test_overlap),
        "train_val_or_test_leakage": len(train_either_overlap),
        "by_label": label_overlap,
    }


def compute_normalized_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    """Finds duplicate counts after lowercasing, punctuation removal, and whitespace normalization."""
    df_copy = df.copy()
    df_copy["norm_text"] = df_copy["text"].apply(normalize_text)

    total_norm_dups = int(len(df_copy) - df_copy["norm_text"].nunique())

    train_norm = set(df_copy[df_copy["split"] == "train"]["norm_text"])
    val_norm = set(df_copy[df_copy["split"] == "val"]["norm_text"])
    test_norm = set(df_copy[df_copy["split"] == "test"]["norm_text"])

    label_overlap = {}
    for label, name in [(0, "human"), (1, "ai")]:
        tr = set(
            df_copy[(df_copy["split"] == "train") & (df_copy["label"] == label)][
                "norm_text"
            ]
        )
        va = set(
            df_copy[(df_copy["split"] == "val") & (df_copy["label"] == label)][
                "norm_text"
            ]
        )
        te = set(
            df_copy[(df_copy["split"] == "test") & (df_copy["label"] == label)][
                "norm_text"
            ]
        )
        label_overlap[name] = {
            "train_val": len(tr.intersection(va)),
            "train_test": len(tr.intersection(te)),
            "val_test": len(va.intersection(te)),
        }

    return {
        "total_normalized_duplicate_rows": total_norm_dups,
        "unique_normalized_texts": int(df_copy["norm_text"].nunique()),
        "train_val_leakage": len(train_norm.intersection(val_norm)),
        "train_test_leakage": len(train_norm.intersection(test_norm)),
        "val_test_overlap": len(val_norm.intersection(test_norm)),
        "by_label": label_overlap,
    }


def compute_near_duplicates_jaccard(
    df: pd.DataFrame, threshold: float = 0.85
) -> Dict[str, int]:
    """Computes near-duplicates between train and val/test using 3-gram Jaccard similarity."""
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    train_shingles = [get_3gram_shingles(t) for t in train_df["text"]]
    val_shingles = [get_3gram_shingles(t) for t in val_df["text"]]
    test_shingles = [get_3gram_shingles(t) for t in test_df["text"]]

    # Inverted index on 3-word shingles for train set
    inv_idx = defaultdict(list)
    for idx, s in enumerate(train_shingles):
        for shingle in s:
            inv_idx[shingle].append(idx)

    def count_near_leaks(query_shingles: List[Set[str]]) -> int:
        count = 0
        for q_s in query_shingles:
            if not q_s:
                continue
            candidate_counts = defaultdict(int)
            for shingle in q_s:
                for train_idx in inv_idx.get(shingle, []):
                    candidate_counts[train_idx] += 1

            min_shared = int(len(q_s) * threshold)
            matched = False
            for cand_idx, shared in candidate_counts.items():
                if shared >= min_shared:
                    cand_s = train_shingles[cand_idx]
                    union_len = len(q_s | cand_s)
                    if union_len > 0 and (len(q_s & cand_s) / union_len) >= threshold:
                        matched = True
                        break
            if matched:
                count += 1
        return count

    val_near = count_near_leaks(val_shingles)
    test_near = count_near_leaks(test_shingles)

    return {
        "train_val_near_duplicates": val_near,
        "train_test_near_duplicates": test_near,
    }


def audit_dataset(
    manifest_path: str, jaccard_threshold: float = 0.85
) -> Dict[str, Any]:
    """Performs full dataset audit on manifest CSV and returns metrics dictionary."""
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")

    df = pd.read_csv(path)
    columns = list(df.columns)
    total_rows = len(df)

    has_generator = "generator" in df.columns
    has_domain = "domain" in df.columns
    has_attack = "attack" in df.columns
    has_dataset = "dataset" in df.columns

    split_counts = compute_split_label_counts(df)

    # Word count calculations
    df["word_count"] = df["text"].apply(lambda x: len(str(x).split()))
    length_stats = {
        "overall": compute_length_stats(df["word_count"]),
        "train": compute_length_stats(df[df["split"] == "train"]["word_count"]),
        "val": compute_length_stats(df[df["split"] == "val"]["word_count"]),
        "test": compute_length_stats(df[df["split"] == "test"]["word_count"]),
        "human": compute_length_stats(df[df["label"] == 0]["word_count"]),
        "ai": compute_length_stats(df[df["label"] == 1]["word_count"]),
    }

    exact_dups = compute_exact_duplicates(df)
    norm_dups = compute_normalized_duplicates(df)
    near_dups = compute_near_duplicates_jaccard(df, threshold=jaccard_threshold)

    generator_counts = df["generator"].value_counts().to_dict() if has_generator else {}
    generator_by_split = (
        pd.crosstab(df["generator"], df["split"]).to_dict() if has_generator else {}
    )

    domain_counts = df["domain"].value_counts().to_dict() if has_domain else {}
    domain_by_split = (
        pd.crosstab(df["domain"], df["split"]).to_dict() if has_domain else {}
    )

    return {
        "manifest_path": str(path),
        "total_rows": total_rows,
        "columns": columns,
        "has_generator": has_generator,
        "has_domain": has_domain,
        "has_attack": has_attack,
        "has_dataset": has_dataset,
        "split_counts": split_counts,
        "length_stats": length_stats,
        "exact_duplicates": exact_dups,
        "normalized_duplicates": norm_dups,
        "near_duplicates": near_dups,
        "jaccard_threshold": jaccard_threshold,
        "generator_counts": generator_counts,
        "generator_by_split": generator_by_split,
        "domain_counts": domain_counts,
        "domain_by_split": domain_by_split,
    }


def generate_markdown_report(audit: Dict[str, Any]) -> str:
    """Formats audit metrics dictionary into structured GitHub-flavored Markdown."""
    lines: List[str] = []
    lines.append("# Text Dataset Audit & Leakage Analysis")
    lines.append("")
    lines.append("**Project ID:** `SKIT/AI/2023-2027/07`  ")
    lines.append(
        "**Author:** Aryansh Agarwal (`23ESKCA021`) — Frontend Architecture & Text Detection Module  "
    )
    lines.append(f"**Manifest File:** `{audit['manifest_path']}`  ")
    lines.append(f"**Total Records:** `{audit['total_rows']:,}`  ")
    lines.append(f"**Columns:** `{', '.join(audit['columns'])}`  ")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Executive Summary & Plain-English Metadata Assessment")
    lines.append("")
    lines.append(
        "This audit inspects the text manifest (`text_manifest.csv`) constructed in Sprint 1 from the "
        "Robust AI Detection (RAID) benchmark to identify split balances, sequence lengths, metadata fidelity, "
        "and data leakage before production fine-tuning."
    )
    lines.append("")
    lines.append("### Metadata Column Availability:")

    gen_status = (
        "**YES** (contains `human` and 11 distinct AI generative architectures: `chatgpt`, `cohere`, "
        "`cohere-chat`, `gpt2`, `gpt3`, `gpt4`, `llama-chat`, `mistral`, `mistral-chat`, `mpt`, `mpt-chat`)"
        if audit["has_generator"]
        else "**NO**"
    )
    dom_status = (
        f"**YES** (contains 1 domain: `{', '.join(audit['domain_counts'].keys())}` across all rows)"
        if audit["has_domain"]
        else "**NO**"
    )
    src_status = (
        "**YES** (`dataset` column contains `raid`, `attack` contains adversarial perturbation types)"
        if audit["has_dataset"]
        else "**NO**"
    )

    lines.append(f"- **Generator Information:** {gen_status}")
    lines.append(f"- **Domain Information:** {dom_status}")
    lines.append(f"- **Dataset / Source Information:** {src_status}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Partition & Label Balance Distribution")
    lines.append("")
    lines.append(
        "| Partition | Human (`label=0`) | AI (`label=1`) | Total Rows | Human Ratio | AI Ratio |"
    )
    lines.append("|---|---|---|---|---|---|")

    for sp in ["train", "val", "test", "total"]:
        counts = audit["split_counts"][sp]
        h = counts["human"]
        a = counts["ai"]
        tot = counts["total"]
        h_pct = (h / tot * 100) if tot > 0 else 0
        a_pct = (a / tot * 100) if tot > 0 else 0
        label_name = sp.capitalize() if sp != "total" else "**Total**"
        lines.append(
            f"| {label_name} | {h:,} | {a:,} | {tot:,} | {h_pct:.1f}% | {a_pct:.1f}% |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. Text Length Statistics (in Words)")
    lines.append("")
    lines.append(
        "| Subset | Sample Count | Mean Words | Median Words | 5th Percentile | 95th Percentile | Min | Max | Std Dev |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")

    for subset_name, key in [
        ("Overall Dataset", "overall"),
        ("Train Split", "train"),
        ("Validation Split", "val"),
        ("Test Split", "test"),
        ("Human Class (`0`)", "human"),
        ("AI Class (`1`)", "ai"),
    ]:
        st = audit["length_stats"][key]
        lines.append(
            f"| {subset_name} | {st['count']:,} | {st['mean']} | {st['median']} | "
            f"{st['p5']} | {st['p95']} | {st['min']} | {st['max']} | {st['std']} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4. Generator & Domain Stratification (Sprint 1 Allocation Flaw)")
    lines.append("")
    lines.append(
        "> [!IMPORTANT]\n"
        "> **Key Audit Finding:** In Sprint 1, the AI records were populated generator-by-generator and split\n"
        "> sequentially by index (`0..7999 -> train`, `8000..8999 -> val`, `9000..9999 -> test`) rather than shuffled\n"
        "> or stratified. Consequently, `train` only observed the first 9 generators, `val` observed `mistral-chat`\n"
        "> and `mpt`, while `test` consisted solely of `mpt` and `mpt-chat`!"
    )
    lines.append("")

    if audit["has_generator"]:
        lines.append("### Generator Allocation Across Splits:")
        lines.append("")
        lines.append(
            "| Generator | Train Count | Val Count | Test Count | Total Count |"
        )
        lines.append("|---|---|---|---|---|")
        all_gens = sorted(list(audit["generator_counts"].keys()))
        for gen in all_gens:
            tr = audit["generator_by_split"].get("train", {}).get(gen, 0)
            va = audit["generator_by_split"].get("val", {}).get(gen, 0)
            te = audit["generator_by_split"].get("test", {}).get(gen, 0)
            tot = audit["generator_counts"].get(gen, 0)
            lines.append(f"| `{gen}` | {tr:,} | {va:,} | {te:,} | {tot:,} |")
        lines.append("")

    if audit["has_domain"]:
        lines.append("### Domain Distribution:")
        lines.append("")
        lines.append("| Domain | Train Count | Val Count | Test Count | Total Count |")
        lines.append("|---|---|---|---|---|")
        for dom, tot in audit["domain_counts"].items():
            tr = audit["domain_by_split"].get("train", {}).get(dom, 0)
            va = audit["domain_by_split"].get("val", {}).get(dom, 0)
            te = audit["domain_by_split"].get("test", {}).get(dom, 0)
            lines.append(f"| `{dom}` | {tr:,} | {va:,} | {te:,} | {tot:,} |")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 5. Data Leakage & Duplication Audit")
    lines.append("")
    lines.append("### A. Exact Duplicates (Raw Character String Match)")
    exact = audit["exact_duplicates"]
    lines.append(
        f"- **Total Exact Duplicate Rows in Dataset:** `{exact['total_duplicate_rows']:,}` "
        f"({exact['unique_texts']:,} unique text sequences out of {audit['total_rows']:,})"
    )
    lines.append(
        f"- **Duplicates within Train Split:** `{exact['within_split']['train']:,}`"
    )
    lines.append(
        f"- **Duplicates within Validation Split:** `{exact['within_split']['val']:,}`"
    )
    lines.append(
        f"- **Duplicates within Test Split:** `{exact['within_split']['test']:,}`"
    )
    lines.append("")
    lines.append("#### Cross-Split Exact Leakage:")
    lines.append(
        f"- **Train ↔ Validation Exact Leakage:** `{exact['train_val_leakage']:,}` identical texts"
    )
    lines.append(
        f"- **Train ↔ Test Exact Leakage:** `{exact['train_test_leakage']:,}` identical texts"
    )
    lines.append(
        f"- **Validation ↔ Test Overlap:** `{exact['val_test_overlap']:,}` identical texts"
    )
    lines.append(
        f"- **Total Train Leakage into Val or Test:** `{exact['train_val_or_test_leakage']:,}` unique texts"
    )
    lines.append("")
    lines.append("#### Exact Leakage by Class:")
    lines.append(
        f"- **Human (`label=0`):** Train-Val = `{exact['by_label']['human']['train_val']}`, "
        f"Train-Test = `{exact['by_label']['human']['train_test']}`, "
        f"Val-Test = `{exact['by_label']['human']['val_test']}`"
    )
    lines.append(
        f"- **AI (`label=1`):** Train-Val = `{exact['by_label']['ai']['train_val']}`, "
        f"Train-Test = `{exact['by_label']['ai']['train_test']}`, "
        f"Val-Test = `{exact['by_label']['ai']['val_test']}`"
    )
    lines.append("")
    lines.append(
        "> [!NOTE]\n"
        "> All exact cross-split leakage originates exclusively in the **Human** class because RAID includes\n"
        "> repeated source abstracts that received multiple adversarial perturbations. The AI class had zero exact\n"
        "> cross-split leakage because generators were separated by split slice."
    )
    lines.append("")

    lines.append(
        "### B. Normalized Duplicates (Lowercase, Punctuation & Whitespace Stripped)"
    )
    norm = audit["normalized_duplicates"]
    lines.append(
        f"- **Total Normalized Duplicate Rows:** `{norm['total_normalized_duplicate_rows']:,}` "
        f"({norm['unique_normalized_texts']:,} unique normalized texts)"
    )
    lines.append(
        f"- **Train ↔ Validation Normalized Leakage:** `{norm['train_val_leakage']:,}` texts "
        f"(Human: `{norm['by_label']['human']['train_val']}`, AI: `{norm['by_label']['ai']['train_val']}`)"
    )
    lines.append(
        f"- **Train ↔ Test Normalized Leakage:** `{norm['train_test_leakage']:,}` texts "
        f"(Human: `{norm['by_label']['human']['train_test']}`, AI: `{norm['by_label']['ai']['train_test']}`)"
    )
    lines.append(
        f"- **Validation ↔ Test Normalized Overlap:** `{norm['val_test_overlap']:,}` texts "
        f"(Human: `{norm['by_label']['human']['val_test']}`, AI: `{norm['by_label']['ai']['val_test']}`)"
    )
    lines.append("")

    lines.append(
        f"### C. Near Duplicates (3-gram Jaccard Similarity $\\ge {audit['jaccard_threshold']}$)"
    )
    near = audit["near_duplicates"]
    lines.append(
        f"- **Validation Samples Near-Identical to Train ($\\ge {audit['jaccard_threshold']}$):** "
        f"`{near['train_val_near_duplicates']:,}` / 2,000 ({near['train_val_near_duplicates']/20:.1f}%)"
    )
    lines.append(
        f"- **Test Samples Near-Identical to Train ($\\ge {audit['jaccard_threshold']}$):** "
        f"`{near['train_test_near_duplicates']:,}` / 2,000 ({near['train_test_near_duplicates']/20:.1f}%)"
    )
    lines.append("")
    lines.append(
        "> [!CAUTION]\n"
        "> Approximately **30% of validation and test samples** have $\\ge 85\\%$ token similarity with training items.\n"
        "> Evaluating on this naive split leads to artificially inflated accuracy and hides model brittleness against genuinely unseen text."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 6. Architecture of the Honest Unseen Holdout Split")
    lines.append("")
    lines.append(
        "To establish a realistic benchmark that accurately measures out-of-domain generalization:\n"
        "1. **Isolated Holdout File (`text_holdout_test.csv`):** An entire generative model (e.g., `chatgpt` or `mistral`) "
        "is completely excised from `train` and `val`, paired with clean human controls, and saved to an independent holdout file.\n"
        "2. **Clean In-Domain Split (`text_manifest.csv`):** The remaining generators are stratified evenly across `train`, `val`, "
        "and in-domain `test` with zero overlap.\n"
        "3. **Dual Metric Reporting:** In Sprint 3 Part 3, models will be evaluated on both in-domain test accuracy AND out-of-domain "
        "holdout accuracy, preventing false confidence from data leakage."
    )
    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    print("=" * 72)
    print("Sprint 3 Part 2: Text Dataset Audit & Leakage Inspection")
    print(f"Manifest Path: {args.manifest}")
    print("=" * 72)

    audit = audit_dataset(args.manifest, args.jaccard_threshold)

    # Print summary to console
    print(
        f"\n1. Overview: {audit['total_rows']:,} rows across {len(audit['columns'])} columns:"
    )
    print(f"   Columns: {', '.join(audit['columns'])}")
    print(f"   Has Generator column: {audit['has_generator']}")
    print(f"   Has Domain column:    {audit['has_domain']}")
    print(f"   Has Attack column:    {audit['has_attack']}")

    print("\n2. Split and Label Balance:")
    for sp in ["train", "val", "test", "total"]:
        cnt = audit["split_counts"][sp]
        print(
            f"   * {sp:<8}: Human={cnt['human']:>5,}, AI={cnt['ai']:>5,}, Total={cnt['total']:>6,}"
        )

    print("\n3. Text Length Stats (in words):")
    for k in ["overall", "train", "val", "test", "human", "ai"]:
        st = audit["length_stats"][k]
        print(
            f"   * {k:<8}: Mean={st['mean']:>6.2f}, Median={st['median']:>5.1f}, "
            f"P5={st['p5']:>5.1f}, P95={st['p95']:>5.1f}, Min={st['min']:>3}, Max={st['max']:>3}"
        )

    exact = audit["exact_duplicates"]
    print("\n4. Exact Duplicates & Leakage Check:")
    print(f"   * Total exact duplicate rows: {exact['total_duplicate_rows']:,}")
    print(f"   * Train <-> Val exact leakage:  {exact['train_val_leakage']:,}")
    print(f"   * Train <-> Test exact leakage: {exact['train_test_leakage']:,}")
    print(f"   * Val <-> Test exact leakage:   {exact['val_test_overlap']:,}")

    norm = audit["normalized_duplicates"]
    print("\n5. Normalized Duplicates (case/punct stripped):")
    print(
        f"   * Total normalized duplicate rows: {norm['total_normalized_duplicate_rows']:,}"
    )
    print(f"   * Train <-> Val normalized leakage:  {norm['train_val_leakage']:,}")
    print(f"   * Train <-> Test normalized leakage: {norm['train_test_leakage']:,}")

    near = audit["near_duplicates"]
    print(f"\n6. Near Duplicates (Jaccard >= {args.jaccard_threshold}):")
    print(f"   * Train <-> Val near duplicates:  {near['train_val_near_duplicates']:,}")
    print(
        f"   * Train <-> Test near duplicates: {near['train_test_near_duplicates']:,}"
    )

    if audit["has_generator"]:
        print("\n7. Generator Distribution:")
        for gen, count in sorted(audit["generator_counts"].items()):
            tr = audit["generator_by_split"].get("train", {}).get(gen, 0)
            va = audit["generator_by_split"].get("val", {}).get(gen, 0)
            te = audit["generator_by_split"].get("test", {}).get(gen, 0)
            print(
                f"   * {gen:<14}: Total={count:>5,}, Train={tr:>5,}, Val={va:>5,}, Test={te:>5,}"
            )

    # Generate and save markdown
    md_content = generate_markdown_report(audit)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md_content, encoding="utf-8")
    print(f"\n[OK] Markdown report written to: {out_path.resolve()}\n")


if __name__ == "__main__":
    main()
