# Text Dataset Audit & Leakage Analysis

**Project ID:** `SKIT/AI/2023-2027/07`  
**Author:** Aryansh Agarwal (`23ESKCA021`) — Frontend Architecture & Text Detection Module  
**Manifest File:** `datasets\splits\text_manifest.csv`  
**Total Records:** `20,000`  
**Columns:** `dataset, split, label, generator, domain, attack, id, text`  

---

## 1. Executive Summary & Plain-English Metadata Assessment

This audit inspects the text manifest (`text_manifest.csv`) constructed in Sprint 1 from the Robust AI Detection (RAID) benchmark to identify split balances, sequence lengths, metadata fidelity, and data leakage before production fine-tuning.

### Metadata Column Availability:
- **Generator Information:** **YES** (contains `human` and 11 distinct AI generative architectures: `chatgpt`, `cohere`, `cohere-chat`, `gpt2`, `gpt3`, `gpt4`, `llama-chat`, `mistral`, `mistral-chat`, `mpt`, `mpt-chat`)
- **Domain Information:** **YES** (contains 1 domain: `abstracts` across all rows)
- **Dataset / Source Information:** **YES** (`dataset` column contains `raid`, `attack` contains adversarial perturbation types)

---

## 2. Partition & Label Balance Distribution

| Partition | Human (`label=0`) | AI (`label=1`) | Total Rows | Human Ratio | AI Ratio |
|---|---|---|---|---|---|
| Train | 8,000 | 8,000 | 16,000 | 50.0% | 50.0% |
| Val | 1,000 | 1,000 | 2,000 | 50.0% | 50.0% |
| Test | 1,000 | 1,000 | 2,000 | 50.0% | 50.0% |
| **Total** | 10,000 | 10,000 | 20,000 | 50.0% | 50.0% |

---

## 3. Text Length Statistics (in Words)

| Subset | Sample Count | Mean Words | Median Words | 5th Percentile | 95th Percentile | Min | Max | Std Dev |
|---|---|---|---|---|---|---|---|---|
| Overall Dataset | 20,000 | 178.33 | 161.0 | 84.0 | 364.0 | 1 | 466 | 80.66 |
| Train Split | 16,000 | 175.99 | 160.0 | 90.0 | 337.0 | 1 | 464 | 74.47 |
| Validation Split | 2,000 | 224.8 | 188.0 | 96.95 | 420.05 | 1 | 466 | 108.81 |
| Test Split | 2,000 | 150.65 | 135.0 | 66.95 | 293.1 | 2 | 453 | 76.75 |
| Human Class (`0`) | 10,000 | 175.05 | 171.0 | 112.0 | 249.0 | 62 | 312 | 41.05 |
| AI Class (`1`) | 10,000 | 181.62 | 137.0 | 71.0 | 402.0 | 1 | 466 | 106.33 |

---

## 4. Generator & Domain Stratification (Sprint 1 Allocation Flaw)

> [!IMPORTANT]
> **Key Audit Finding:** In Sprint 1, the AI records were populated generator-by-generator and split
> sequentially by index (`0..7999 -> train`, `8000..8999 -> val`, `9000..9999 -> test`) rather than shuffled
> or stratified. Consequently, `train` only observed the first 9 generators, `val` observed `mistral-chat`
> and `mpt`, while `test` consisted solely of `mpt` and `mpt-chat`!

### Generator Allocation Across Splits:

| Generator | Train Count | Val Count | Test Count | Total Count |
|---|---|---|---|---|
| `chatgpt` | 910 | 0 | 0 | 910 |
| `cohere` | 909 | 0 | 0 | 909 |
| `cohere-chat` | 909 | 0 | 0 | 909 |
| `gpt2` | 909 | 0 | 0 | 909 |
| `gpt3` | 909 | 0 | 0 | 909 |
| `gpt4` | 909 | 0 | 0 | 909 |
| `human` | 8,000 | 1,000 | 1,000 | 10,000 |
| `llama-chat` | 909 | 0 | 0 | 909 |
| `mistral` | 909 | 0 | 0 | 909 |
| `mistral-chat` | 727 | 182 | 0 | 909 |
| `mpt` | 0 | 818 | 91 | 909 |
| `mpt-chat` | 0 | 0 | 909 | 909 |

### Domain Distribution:

| Domain | Train Count | Val Count | Test Count | Total Count |
|---|---|---|---|---|
| `abstracts` | 16,000 | 2,000 | 2,000 | 20,000 |

---

## 5. Data Leakage & Duplication Audit

### A. Exact Duplicates (Raw Character String Match)
- **Total Exact Duplicate Rows in Dataset:** `612` (19,388 unique text sequences out of 20,000)
- **Duplicates within Train Split:** `415`
- **Duplicates within Validation Split:** `6`
- **Duplicates within Test Split:** `5`

#### Cross-Split Exact Leakage:
- **Train ↔ Validation Exact Leakage:** `99` identical texts
- **Train ↔ Test Exact Leakage:** `79` identical texts
- **Validation ↔ Test Overlap:** `12` identical texts
- **Total Train Leakage into Val or Test:** `174` unique texts

#### Exact Leakage by Class:
- **Human (`label=0`):** Train-Val = `99`, Train-Test = `79`, Val-Test = `12`
- **AI (`label=1`):** Train-Val = `0`, Train-Test = `0`, Val-Test = `0`

> [!NOTE]
> All exact cross-split leakage originates exclusively in the **Human** class because RAID includes
> repeated source abstracts that received multiple adversarial perturbations. The AI class had zero exact
> cross-split leakage because generators were separated by split slice.

### B. Normalized Duplicates (Lowercase, Punctuation & Whitespace Stripped)
- **Total Normalized Duplicate Rows:** `3,656` (16,344 unique normalized texts)
- **Train ↔ Validation Normalized Leakage:** `422` texts (Human: `408`, AI: `14`)
- **Train ↔ Test Normalized Leakage:** `380` texts (Human: `380`, AI: `0`)
- **Validation ↔ Test Normalized Overlap:** `106` texts (Human: `98`, AI: `8`)

### C. Near Duplicates (3-gram Jaccard Similarity $\ge 0.85$)
- **Validation Samples Near-Identical to Train ($\ge 0.85$):** `618` / 2,000 (30.9%)
- **Test Samples Near-Identical to Train ($\ge 0.85$):** `607` / 2,000 (30.4%)

> [!CAUTION]
> Approximately **30% of validation and test samples** have $\ge 85\%$ token similarity with training items.
> Evaluating on this naive split leads to artificially inflated accuracy and hides model brittleness against genuinely unseen text.

---

## 6. Architecture of the Honest Unseen Holdout Split

To establish a realistic benchmark that accurately measures out-of-domain generalization:
1. **Isolated Holdout File (`text_holdout_test.csv`):** An entire generative model (e.g., `chatgpt` or `mistral`) is completely excised from `train` and `val`, paired with clean human controls, and saved to an independent holdout file.
2. **Clean In-Domain Split (`text_manifest.csv`):** The remaining generators are stratified evenly across `train`, `val`, and in-domain `test` with zero overlap.
3. **Dual Metric Reporting:** In Sprint 3 Part 3, models will be evaluated on both in-domain test accuracy AND out-of-domain holdout accuracy, preventing false confidence from data leakage.
