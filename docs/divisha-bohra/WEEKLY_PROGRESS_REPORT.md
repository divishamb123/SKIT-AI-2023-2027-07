# Divisha Manak Bohra — Individual Weekly Progress Report

**Name:** Divisha Manak Bohra  
**College ID:** `23ESKCA038`  
**GitHub:** `divishamb123`  
**Role:** Member 1 — Image Detection Modality & Image Interface  
**Repository:** https://github.com/divishamb123/SKIT-AI-2023-2027-07  
**Project ID:** `SKIT/AI/2023-2027/07`  

> This document is the official individual weekly log for Divisha Manak Bohra.  
> All tasks and task descriptions are taken **verbatim** from Form-2 (*Roles & Responsibility of Team Members*).  
> Each weekly entry includes verifiable GitHub commit SHAs as proof of work.  

---

## Sprint 1 — Data Preparation & Interface

**Form-2 User Story:** Preparing image data and authentication interface  
**Form-2 Dates:** 03-Aug-2026 to 20-Sep-2026  
**Form-2 Deliverables:**  
- Collecting and organising image datasets for real/AI-generated classification  
- Creating balanced train, validation and test splits  
- Designing login and registration interface components  
- Building responsive authentication screens and handling representative form states  

---

### Week 1 — 03 September 2026

**Task completed (Form-2 Sprint 1 words):**  
*Collecting and organising image datasets for real/AI-generated classification.*

**Specific work done:**  
- Curated 132,000 real and synthetic images from CIFAKE and GenImage.
- Standardized file paths, binary class labels (`0` for Real, `1` for AI-generated), and generator metadata.
- Generated `datasets/splits/manifest.csv`.

**GitHub Evidence:**

| Commit SHA | Date | Description |
|---|---|---|
| [`77a5f21`](https://github.com/divishamb123/SKIT-AI-2023-2027-07/commit/77a5f21) | 02-Sep-2026 | feat(data): collect and organize image dataset curation pipeline (Sprint 1 - Task 1) |
| [`85990d6`](https://github.com/divishamb123/SKIT-AI-2023-2027-07/commit/85990d6) | 02-Sep-2026 | docs: update README to reflect Week 1 dataset curation scope only |

**Sprint 1 Progress:** 25% (+25%)

---

### Week 2 — 10 September 2026

**Task completed (Form-2 Sprint 1 words):**  
*Creating balanced train, validation and test splits.*

**Specific work done:**  
- Structured deterministic dataset partitions with exact 50/50 balance using `seed=42`:
  - Train: 90,000 images (45k Real / 45k AI)
  - Validation: 10,000 images (5k Real / 5k AI)
  - In-domain Test: 20,000 images (10k Real / 10k AI)
  - Out-of-domain Holdout: 12,000 images (6k Real / 6k AI)
- Built automated verification script `scripts/verify_sprint1_splits.py` and unit tests in `tests/test_dataset_splits.py` proving zero cross-split leakage.

**GitHub Evidence:**

| Commit SHA | Date | Description |
|---|---|---|
| [`1b7fcf1`](https://github.com/divishamb123/SKIT-AI-2023-2027-07/commit/1b7fcf1) | 06-Sep-2026 | feat(data): create balanced train-val-test splits and partition verification suite |
| [`a894c3c`](https://github.com/divishamb123/SKIT-AI-2023-2027-07/commit/a894c3c) | 06-Sep-2026 | docs: restore Week 1 deliverables and retain Week 2 deliverables in README |

**Tests:** 6 passed (`tests/test_dataset_splits.py`)  
**Sprint 1 Progress:** 50% (+25%)

---

### Week 3 — 17 September 2026

**Task completed (Form-2 Sprint 1 words):**  
*Designing login and registration interface components.*

**Specific work done:**  
- Developed reusable Next.js / TypeScript UI primitives in `frontend/src/components/`:
  - `Input.tsx`: High-contrast accessible input field with real-time feedback.
  - `Button.tsx`: Button component supporting loading spinner states and variants.
  - `AuthCard.tsx`: Glassmorphism card container.
  - `PasswordRequirements.tsx`: Dynamic security indicator scoring Length, Uppercase, Numbers, and Symbols.
  - `FormAlert.tsx`: Error and success notifications.

**GitHub Evidence:**

| Commit SHA | Date | Description |
|---|---|---|
| [`c840a26`](https://github.com/divishamb123/SKIT-AI-2023-2027-07/commit/c840a26) | 14-Sep-2026 | feat(auth): implement authentication UI primitives, responsive screens, and session handling |

**Sprint 1 Progress:** 75% (+25%)

---

### Week 4 — 20 September 2026

**Task completed (Form-2 Sprint 1 words):**  
*Building responsive authentication screens and handling representative form states.*

**Specific work done:**  
- Implemented `/login` and `/register` pages with form validation, error banners, and token storage.
- Integrated `AuthContext.tsx` managing user session lifecycle.
- Created `scripts/verify_auth_components.mjs` verifying all 25 form states and UI interactions.

**GitHub Evidence:**

| Commit SHA | Date | Description |
|---|---|---|
| [`5fb6a76`](https://github.com/divishamb123/SKIT-AI-2023-2027-07/commit/5fb6a76) | 14-Sep-2026 | feat(dashboard): build realistic enterprise forensics workbench with multi-modal scanning |
| [`20d7f9e`](https://github.com/divishamb123/SKIT-AI-2023-2027-07/commit/20d7f9e) | 14-Sep-2026 | docs: complete Sprint 1 documentation for Tasks 3 and 4 |

**Sprint 1 Status: ✅ COMPLETE (100%)**

---

## Sprint 2 — Baseline Detection Service

**Form-2 User Story:** Building the image detection module  
**Form-2 Dates:** 21-Sep-2026 to 08-Nov-2026  
**Form-2 Deliverables:**  
- Developing an image inference service with a defined input/output interface  
- Designing batch-processing logic for multiple image inputs  
- Testing image inference using grouped input samples  
- Designing image upload and result interaction components  

---

### Week 1 — 24 September 2026

**Task completed (Form-2 Sprint 2 words):**  
*Developing an image inference service with a defined input/output interface.*

**Specific work done:**  
- Implemented `BaselineImageClassifier` (ResNet-18) with dynamic device detection (`mps`, `cuda`, `cpu`).
- Built preprocessing pipeline converting bytes, Base64, and PIL images to standardized $3 \times 224 \times 224$ tensors with ImageNet normalization.
- Created Pydantic I/O contracts returning verdicts (`REAL` / `AI_GENERATED`), probabilities, and metadata.

**GitHub Evidence:**

| Commit SHA | Date | Description |
|---|---|---|
| [`11713a8`](https://github.com/divishamb123/SKIT-AI-2023-2027-07/commit/11713a8) | 18-Sep-2026 | feat(sprint2): implement Week 1 baseline image inference service and defined I/O interface |
| [`8b3c4d6`](https://github.com/divishamb123/SKIT-AI-2023-2027-07/commit/8b3c4d6) | 18-Sep-2026 | feat: restore shared services, text pipeline, docker orchestration and integrate Week 1 |

**Tests:** 9 passed (`tests/test_image_inference_service.py`)  
**Sprint 2 Progress:** 25% (+25%)

---

### Week 2 — 01 October 2026

**Task completed (Form-2 Sprint 2 words):**  
*Developing an image inference service with a defined input/output interface.*

**Specific work done:**  
- Built FastAPI microservice in `src/image_detector/api.py` and `services/image-service/app/main.py` on port `8004`.
- Implemented 6 security guardrails: HTTP 200 (valid), 400 (empty payload), 413 (>15 MB), 415 (unsupported extension), 415 (magic-byte spoofing), 422 (corrupted/out-of-bounds).
- Configured production Docker container in `services/image-service/Dockerfile`.

**GitHub Evidence:**

| Commit SHA | Date | Description |
|---|---|---|
| [`bd4626d`](https://github.com/divishamb123/SKIT-AI-2023-2027-07/commit/bd4626d) | 26-Sep-2026 | feat(sprint2): implement Week 2 I/O API specification and /api/detect/image with strict validation |
| [`e9b8b30`](https://github.com/divishamb123/SKIT-AI-2023-2027-07/commit/e9b8b30) | 26-Sep-2026 | style: format codebase with black for CI compliance |

**Tests:** 15 passed (`tests/test_image_detection_api.py`)  
**Sprint 2 Progress:** 50% (+25%)

---

### Week 3 — 08 October 2026

**Task completed (Form-2 Sprint 2 words):**  
*Designing batch-processing logic for multiple image inputs.*

**Specific work done:**  
- Implemented batched tensor collation `(B, 3, 224, 224)` and vectorized forward pass yielding ~2.3x throughput speedup over sequential inference.
- Implemented dynamic chunking ($B \le 32$) protecting GPU from OOM exceptions.
- Implemented fault-isolated partial error handling: corrupted files are isolated in the error list without failing valid images.
- Added `POST /api/detect/image/batch` and `POST /api/detect/image/batch/base64` endpoints.

**GitHub Evidence:**

| Commit SHA | Date | Description |
|---|---|---|
| [`64483d2`](https://github.com/divishamb123/SKIT-AI-2023-2027-07/commit/64483d2) | 02-Oct-2026 | feat(sprint2): implement Week 3 batch-processing logic, tensor collation, and batch API |

**Tests:** 11 passed (`tests/test_batch_processing.py`)  
**Sprint 2 Progress:** 70% (+20%)

---

### Week 4 — 15 October 2026

**Task completed (Form-2 Sprint 2 words):**  
*Testing image inference using grouped input samples.*

**Specific work done:**  
- Developed `GroupedImageEvaluator` in `src/image_detector/evaluator.py`.
- Stratified 400 manifest samples across classes (Real vs AI), splits (in-domain CIFAKE vs out-of-domain GenImage), generators (`cifar10`, `stable-diffusion-1.4`, `midjourney`, `biggan`), and batch sizes (1 to 32).
- Measured inference throughput at 176–364 FPS on Apple Silicon GPU (`mps`).
- Documented baseline benchmark in `docs/image_baseline_results.md` and `results/image_baseline_evaluation.json`.

**GitHub Evidence:**

| Commit SHA | Date | Description |
|---|---|---|
| [`6f1e5b5`](https://github.com/divishamb123/SKIT-AI-2023-2027-07/commit/6f1e5b5) | 11-Oct-2026 | feat(sprint2): implement Week 4 grouped sample testing, GroupedImageEvaluator engine & metrics |
| [`d772df6`](https://github.com/divishamb123/SKIT-AI-2023-2027-07/commit/d772df6) | 11-Oct-2026 | style(sprint2): format Week 4 files with Black for CI compliance |

**Tests:** 11 passed (`tests/test_grouped_image_evaluation.py`)  
**Sprint 2 Progress:** 85% (+15%)

---

### Week 5 — 22 October 2026

**Task completed (Form-2 Sprint 2 words):**  
*Designing image upload and result interaction components.*

**Specific work done:**  
- Built `ImageUploadDropzone.tsx` with drag-and-drop, validation, and sample photo loaders.
- Built `VerdictGauge.tsx` with 270-degree SVG radial dial and probability breakdown bar.
- Built `ImageMetadataCard.tsx` with resolution, channels, latency, and clipboard JSON export.
- Built `BatchResultsGallery.tsx` with KPI overview cards, tabs, and export tools.
- Integrated `/image-detection` into Next.js navigation.

**GitHub Evidence:**

| Commit SHA | Date | Description |
|---|---|---|
| [`f1fff28`](https://github.com/divishamb123/SKIT-AI-2023-2027-07/commit/f1fff28) | 16-Oct-2026 | feat(sprint2): implement Week 5 image upload and result interaction components |

**Tests:** 8 passed (`tests/test_image_interaction_components.py`)  
**Sprint 2 Progress:** 92% (+7%)

---

### Week 6 — 29 October 2026

**Task completed (Form-2 Sprint 2 words):**  
*Designing image upload and result interaction components.*

**Specific work done:**  
- Developed `DecisionBoundaryController.tsx` with dynamic threshold slider $\tau \in [0.10, 0.90]$, presets, and margin calculation ($\Delta = |P(\text{AI}) - \tau|$).
- Developed `ForensicEvidenceViewer.tsx` with HTML5 Canvas channel decomposition: RGB, Luminance $Y$ ($0.299R + 0.587G + 0.114B$), Spatial High-Pass filter, and Bayer channels with coordinate probe HUD.
- Developed `ForensicReportModal.tsx` generating SHA-256 evidence fingerprints and printable PDF certificates.

**GitHub Evidence:**

| Commit SHA | Date | Description |
|---|---|---|
| [`c772b88`](https://github.com/divishamb123/SKIT-AI-2023-2027-07/commit/c772b88) | 24-Oct-2026 | feat(sprint2): implement Week 6 forensic evidence inspection, threshold calibration, and audit certification |

**Tests:** 8 passed (`tests/test_image_interaction_week6.py`)  
**Sprint 2 Progress:** 97% (+5%)

---

### Week 7 — 01 November 2026 (Final Sprint 2 Milestone)

**Task completed (Form-2 Sprint 2 words):**  
*All 4 Tasks: Inference service, batch-processing, grouped testing, and interaction components.*

**Specific work done:**  
- Implemented API Gateway synchronous image detection proxy routes (`POST /api/v1/detect/image/direct`, `/direct-batch`, `/direct-base64`) connecting frontend and microservice in `docker-compose.yml`.
- Added frontend API client unit test suite (`frontend/src/lib/__tests__/image-api.test.ts`) testing multipart dispatch, Base64 JSON, and batch collation.
- Created comprehensive master Sprint 2 verification script (`scripts/verify_sprint2_final.py`) executing 8 validation stages.
- Created automated 12-test regression audit suite (`tests/test_sprint2_final_audit.py`).
- Verified all **94 project pytest tests pass 100%**, all **7 Vitest tests pass 100%**, and Next.js builds cleanly with 0 errors.

**Evidence:**  
- Master Verification Script: `scripts/verify_sprint2_final.py` (8/8 checks passed, 100%)
- Audit Test Suite: `tests/test_sprint2_final_audit.py` (12/12 passed)
- Comprehensive Audit Report: `DIVISHA_MANAK_BOHRA_INDIVIDUAL_WORK_AUDIT.md`

**Sprint 2 Status: ✅ COMPLETE (100%)**

---

## Sprint 2 Deliverables Summary

| Form-2 Sprint 2 Deliverable | Implementation Artifacts | Status |
|---|---|---|
| **Developing an image inference service with a defined input/output interface** | `src/image_detector/model.py`, `src/image_detector/preprocessing.py`, `src/image_detector/schemas.py`, `services/image-service/` | **✅ 100% COMPLETE** |
| **Designing batch-processing logic for multiple image inputs** | `predict_batch()` in `src/image_detector/service.py`, tensor collation, dynamic chunking $B \le 32$ | **✅ 100% COMPLETE** |
| **Testing image inference using grouped input samples** | `src/image_detector/evaluator.py`, `docs/image_baseline_results.md`, `results/image_baseline_evaluation.json` | **✅ 100% COMPLETE** |
| **Designing image upload and result interaction components** | `ImageUploadDropzone.tsx`, `VerdictGauge.tsx`, `ImageMetadataCard.tsx`, `BatchResultsGallery.tsx`, `DecisionBoundaryController.tsx`, `ForensicEvidenceViewer.tsx`, `ForensicReportModal.tsx` | **✅ 100% COMPLETE** |

---

*Last updated: 01 November 2026 — Divisha Manak Bohra (23ESKCA038)*
