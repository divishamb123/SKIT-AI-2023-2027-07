# Sprint 1 — Week 2: Image Dataset Preparation & Partitioning

**Student:** Divisha Manak Bohra (`23ESKCA038`)  
**Department:** Computer Science & Engineering (Artificial Intelligence)  
**Institution:** Swami Keshvanand Institute of Technology, Management & Gramothan (SKIT), Jaipur  
**Academic Year:** 2026–27 (Phase-II, 7th Semester)  
**Sprint Window:** 03-August-2026 to 20-September-2026  
**User Story:** *Preparing image data and authentication interface*

---

## 📌 Milestone Overview (Week 2 Submission)

This submission delivers **Task 1 and Task 2** of Form – 2 for Sprint 1:

| S.No. | Form-2 Sprint 1 Task | Implementation | Status |
|---|---|---|---|
| 1 | **Collecting and organising image datasets for real/AI-generated classification** | Curated 132,000 real and synthetic images across CIFAKE and GenImage datasets. | Completed (Week 1) |
| 2 | **Creating balanced train, validation and test splits** | Generated 90k train, 10k val, 20k test, and 12k holdout splits with exact 50/50 class balance (`seed=42`). | Completed & Verified (Week 2) |
| 3 | **Designing login and registration interface components** | Reusable UI primitives (`Input`, `Button`, `AuthCard`, `PasswordRequirements`). | Scheduled (Week 3) |
| 4 | **Building responsive authentication screens and handling representative form states** | Responsive `/login` and `/register` pages with form validation, loading, error, and session states. | Scheduled (Week 3) |

---

## 📁 Week 2 Deliverables & Architecture

| Component | Path | Description |
|---|---|---|
| **Data Preparation Script** | [`scripts/prepare_image_datasets.py`](scripts/prepare_image_datasets.py) | Python pipeline to scan, catalog, and structure real and generative AI image datasets. |
| **Partition Manifest** | [`datasets/splits/manifest.csv`](datasets/splits/manifest.csv) | Partition catalog tracking 132,000 records with label, generator, and split metadata. |
| **Verification Suite** | [`scripts/verify_sprint1_splits.py`](scripts/verify_sprint1_splits.py) | Standalone verification suite checking class distribution, zero leakage, and file existence. |
| **Automated Tests** | [`tests/test_dataset_splits.py`](tests/test_dataset_splits.py) | Pytest suite enforcing schema consistency, class balance, and partition disjointness. |
| **Dependencies** | [`requirements.txt`](requirements.txt) | Environment dependencies for data handling and verification. |
| **Git Configuration** | [`.gitignore`](.gitignore) | Ignore rules to exclude binary caches, build artifacts, and virtual environments. |

---

## 🗂️ Dataset Partitioning & Stratification

All partition metadata is stored in [`datasets/splits/manifest.csv`](datasets/splits/manifest.csv) (132,000 records).

### Split Distribution

| Split | Partition Role | Total Images | Real ($y=0$) | AI Synthetic ($y=1$) | Class Balance | Sources |
|---|---|---|---|---|---|---|
| `train` | Training | 90,000 | 45,000 | 45,000 | 50.0% / 50.0% | CIFAKE (CIFAR-10 / SD-v1.4) |
| `val` | Validation | 10,000 | 5,000 | 5,000 | 50.0% / 50.0% | CIFAKE (CIFAR-10 / SD-v1.4) |
| `test` | In-Domain Test | 20,000 | 10,000 | 10,000 | 50.0% / 50.0% | CIFAKE (CIFAR-10 / SD-v1.4) |
| `holdout` | Out-of-Domain Holdout | 12,000 | 6,000 | 6,000 | 50.0% / 50.0% | GenImage (Real, SD, Midjourney, BigGAN) |
| **Total** | **All Splits** | **132,000** | **66,000** | **66,000** | **50.0% / 50.0%** | **Multi-Generator Corpus** |

### Generators & Sources

- `cifar10`: 60,000 samples (Real baseline)
- `stable-diffusion-1.4`: 60,000 samples (Latent diffusion)
- `real`: 6,000 samples (High-resolution photography)
- `stable-diffusion`: 5,000 samples (Diffusion)
- `midjourney`: 500 samples (Zero-shot holdout)
- `biggan`: 500 samples (Zero-shot holdout)

### Partition Integrity Guarantees

- **Class Balance:** Exact 50.0% real and 50.0% fake across all 4 partitions.
- **Zero Leakage:** Confirmed 0 duplicate file paths across train, val, test, and holdout splits.
- **Reproducibility:** Deterministic split assignment using pseudo-random seed `42`.

---

## 🚀 Execution & Verification Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Dataset Preparation Pipeline
```bash
python3 scripts/prepare_image_datasets.py
```

### 3. Verify Dataset Partitions & Zero Leakage
```bash
python3 scripts/verify_sprint1_splits.py
```

### 4. Run Automated Pytest Suite
```bash
pytest tests/test_dataset_splits.py
```

---

# Sprint 1 — Week 1: Frontend Architecture & Text Dataset Preparation

**Student:** Aryansh Agarwal (`23ESKCA021`)  
**Department:** Computer Science & Engineering (Artificial Intelligence)  
**Institution:** Swami Keshvanand Institute of Technology, Management & Gramothan (SKIT), Jaipur  
**Academic Year:** 2026–27 (Phase-II, 7th Semester)  
**Sprint Window:** 03-August-2026 to 20-September-2026  
**User Story:** *Preparing text data and frontend architecture*

---

## 📌 Milestone Overview (Week 1 Submission)

This submission delivers **Task 1** of Form – 2 for Sprint 1 (Aryansh Agarwal):

> **Form-2 Task 1:** *Implementing core frontend architecture and preparing the Robust AI Detection (RAID) text benchmark.*

---

## 📁 Week 1 Deliverables & Architecture

| Component | Path | Description |
|---|---|---|
| **Frontend Foundation** | [`frontend/`](frontend/) | Next.js 16 (App Router), TypeScript, Tailwind CSS, Axios API client. |
| **UI & Layout** | [`frontend/src/components/`](frontend/src/components/) | Reusable `Navbar`, `Footer`, `UploadDropzone`, `JobStatusCard`, `Button`, `Modal`, `LoadingSpinner`. |
| **Application Pages** | [`frontend/src/app/`](frontend/src/app/) | Initial routes: `/`, `/login`, `/register`, `/dashboard`, `/health`. |
| **Text Data Pipeline** | [`scripts/prepare_text_dataset.py`](scripts/prepare_text_dataset.py) | Streams HF `liamdugan/raid`, balances classes, generates text manifest. |
| **Text Manifest** | [`datasets/splits/text_manifest.csv`](datasets/splits/text_manifest.csv) | Final 20,000-record text split (80/10/10). |

---

## 🗂️ Dataset Sources & Organisation

The text pipeline prepares a deterministically sampled 20,000-record subset from the **Robust AI Detection (RAID)** benchmark (`liamdugan/raid`), producing a balanced dataset:

1. **Human Collection (Real - Label 0)**:
   - 10,000 deterministically sampled human-written records.
2. **AI Collection (Fake - Label 1)**:
   - 10,000 deterministically sampled AI-generated records evenly distributed across 11 distinct generator subtypes (e.g., ChatGPT, Cohere, LLaMA, Mistral, GPT-4).
3. **Partitioning**:
   - Random Seed `42` ensures perfect reproducibility.
   - 80/10/10 stratified split: Train (16,000), Validation (2,000), Test (2,000).
   - Raw HF corpus is streamed and cache is excluded via `.gitignore` to prevent storage bloat.

---

## 🚀 Execution Instructions & Verification

### 1. Verification Results
- **Frontend**: Linting and production build passed cleanly.
- **Dataset**: Validation confirms 50/50 balance across all splits, no empty texts, no duplicate IDs, and identical SHA-256 generation across multiple script runs.

### 2. Run the Text Dataset Pipeline
```bash
# Install text requirements
pip install -r requirements.txt

# Run the pipeline
python3 scripts/prepare_text_dataset.py
```
