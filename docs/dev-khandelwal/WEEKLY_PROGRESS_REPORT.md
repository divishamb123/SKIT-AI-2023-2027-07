# Dev Khandelwal - Individual Weekly Progress Report

**Name:** Dev Khandelwal
**College ID:** 23ESKCA035
**GitHub:** Dev-learner-17
**Role:** Team Lead - audio detection service, complete backend, fusion layer
**Repository:** https://github.com/divishamb123/SKIT-AI-2023-2027-07

> This document is the official individual weekly log for Dev Khandelwal.
> All tasks and task descriptions are taken **verbatim** from Form-2 (Roles & Responsibility of Team Members).
> Each entry includes the GitHub commit SHA as verifiable proof of work.

---

## Sprint 1 - Foundation & Data Preparation

**Form-2 User Story:** Develop authentication system and backend setup
**Form-2 Dates:** 03-Aug-2026 to 20-Sep-2026
**Form-2 Deliverable:** Set up Docker stack (PostgreSQL, RabbitMQ, MinIO). Created database schema. Built registration, login and JWT authentication. Built file upload and job tracking pipeline.

---

### Week - 03 September 2026

**Task completed (Form-2 Sprint 1 words):**
Set up Docker stack (PostgreSQL, RabbitMQ, MinIO). Created database schema. Built registration, login and JWT authentication.

**GitHub Evidence:**

| Commit SHA | Date | Description |
|---|---|---|
| [`c0edf0fd`](https://github.com/divishamb123/SKIT-AI-2023-2027-07/commit/c0edf0fd43bea1c355d94c21c9b2ce6c80b67f1b) | 03-Sep-2026 | add backend infrastructure foundation |
| [`314b1b2b`](https://github.com/divishamb123/SKIT-AI-2023-2027-07/commit/314b1b2bd71d90fe420d67c569b4b282a652e42c) | 03-Sep-2026 | add authentication and token management |
| [`12ff4438`](https://github.com/divishamb123/SKIT-AI-2023-2027-07/commit/12ff443806ffa3cd530d1f3ef5085d210a09f4dd) | 03-Sep-2026 | add API Gateway foundation |
| [`b6abf377`](https://github.com/divishamb123/SKIT-AI-2023-2027-07/commit/b6abf37734d0e1202d8169ddd912da78e3ecbf77) | 03-Sep-2026 | add job queue and tracking pipeline |
| [`93cd342f`](https://github.com/divishamb123/SKIT-AI-2023-2027-07/commit/93cd342fbdd366e50b9ba266d5729d98bd8bc30b) | 03-Sep-2026 | add backend verification tests |
| [`f393b60d`](https://github.com/divishamb123/SKIT-AI-2023-2027-07/commit/f393b60df43d467b7ec7106f66877ef7112c6694) | 03-Sep-2026 | add Sprint 1 progress status |

**Branch:** `dev-khandelwal/sprint-1-foundation`

---

### Week - 10 September 2026

**Task completed (Form-2 Sprint 1 words):**
Built file upload and job tracking pipeline. Complete Docker application integration, environment verification, and E2E testing.

**GitHub Evidence:**

| Commit SHA | Date | Description |
|---|---|---|
| [`f67ef9bd`](https://github.com/divishamb123/SKIT-AI-2023-2027-07/commit/f67ef9bd546b2f2816d4357d40e08e051f78945d) | 10-Sep-2026 | complete Docker application integration, environment verification, and E2E testing |
| [`0f28eeff`](https://github.com/divishamb123/SKIT-AI-2023-2027-07/commit/0f28eeff9363449006066baa93b551e05d1fbbaa) | 10-Sep-2026 | sync latest branch state |

**Branch:** `dev-khandelwal/sprint-1-foundation`

---

**Sprint 1 Status: ✅ COMPLETE (100%)**
All Form-2 deliverables for Sprint 1 are implemented and merged into `main`.

---

## Sprint 2 - Baseline Detection Services

**Form-2 User Story:** Develop audio detection service
**Form-2 Dates:** 21-Sep-2026 to 08-Nov-2026
**Form-2 Deliverable:** Built audio preprocessing with librosa and torchaudio. Integrated pre-trained AASIST-L model. Implemented background worker for audio jobs. Recorded baseline EER.

---

### Week 1 - 08 October 2026

**Task completed (Form-2 Sprint 2 words):**
Built audio preprocessing with librosa and torchaudio. Integrated pre-trained AASIST-L model.

**Specific work done:**
- Fixed audio worker variable unpacking bug preventing background worker from running
- Resolved object-key payload contract mismatch between API Gateway and RabbitMQ consumer
- Added `/audio` file upload endpoint to API Gateway (completing Sprint 1 foundation gap)
- Added `QUEUE_AUDIO_DETECTION` queue config
- Added meaningful unit tests for `AudioPreprocessor`
- Documented audio API contract in `docs/dev-khandelwal/AUDIO_API_CONTRACT.md`

**GitHub Evidence:**

| Commit SHA | Date | Description |
|---|---|---|
| [`905f3cb8`](https://github.com/divishamb123/SKIT-AI-2023-2027-07/commit/905f3cb82903c0cb332fd3f961008b7b28f1f0cd) | 24-Sep-2026 | implement AASIST-L baseline, preprocessing, and background worker |
| [`2a27c424`](https://github.com/divishamb123/SKIT-AI-2023-2027-07/commit/2a27c424a9a6868121585186214398d074565850) | 06-Oct-2026 | fix audio worker bugs and add API endpoint for week 1 |

**Branch:** `dev-khandelwal/sprint-2-audio`
**Tests:** 3 passed (`tests/unit/test_audio_service.py`)
**Sprint 2 %:** 77% (+7%)

---

### Week 2 - 15 October 2026

**Task completed (Form-2 Sprint 2 words):**
Integrated pre-trained AASIST-L model.

**Specific work done:**
- Refactored `audio_detector.py` to resolve model paths from environment variables (`AASIST_WEIGHTS_PATH` / `AASIST_CONFIG_PATH`) instead of hardcoded absolute paths
- Implemented graceful degraded-mode: service starts even when weights are absent; `/internal/health` returns HTTP 503 with diagnostic JSON
- Added `is_ready()` guard in queue consumer - marks jobs `FAILED` cleanly instead of crashing
- Added `./models:/models:ro` volume mount to `docker-compose.yml` - audio service now Docker-runnable
- Written operator guide `docs/dev-khandelwal/MODEL_SETUP.md`
- 10 new unit tests for degraded-mode state machine and healthy-mode inference

**GitHub Evidence:**

| Commit SHA | Date | Description |
|---|---|---|
| [`1642630b`](https://github.com/divishamb123/SKIT-AI-2023-2027-07/commit/1642630ba973cedfa7d35763f4b8acc4f265bb6a) | 14-Oct-2026 | Docker-ready AASIST-L with graceful degradation |

**Branch:** `dev-khandelwal/sprint-2-audio`
**Tests:** 13 passed (10 detector + 3 preprocessor)
**Sprint 2 %:** 84% (+7%)

---

### Week 3 - 22 October 2026

**Task completed (Form-2 Sprint 2 words):**
Implemented background worker for audio jobs.

**Specific work done:**
- Added per-job retry with exponential back-off (1s → 2s → 4s, up to 3 retries) in `queue_consumer.py`
- Added inference timeout (30s) via `asyncio.wait_for()` around `predict_sync` call
- Added MinIO download timeout (20s) via `asyncio.wait_for()` around S3 download
- Jobs exhausting all retries permanently marked `FAILED` with descriptive `error_message` in database
- All retry/timeout constants exposed as env vars in `docker-compose.yml`
- 12 new unit tests for retry header parsing, env-var overrides, and exponential back-off formula

**GitHub Evidence:**

| Commit SHA | Date | Description |
|---|---|---|
| [`1b97099e`](https://github.com/divishamb123/SKIT-AI-2023-2027-07/commit/1b97099e5e0ae95c479be129f0f932603dff1d9b) | 16-Oct-2026 | implement background worker for audio jobs with retry and timeout handling |

**Branch:** `dev-khandelwal/sprint-2-audio`
**Tests:** 22 passed (12 retry + 10 detector)
**Sprint 2 %:** 91% (+7%)

---

### Week 4 - 29 October 2026

**Task completed (Form-2 Sprint 2 words):**
Recorded baseline EER.

**Specific work done:**
- Ran `scripts/evaluate_audio_baseline.py` against the full ASVspoof 2019 LA evaluation set (71,237 utterances, 0 failed)
- AASIST-L pretrained baseline EER: **0.992%** — meets the Sprint 3 target of ≤ 1.0%
- EER threshold: 1.7633 (bonafide logit direction)
- Results committed to `results/asvspoof_baseline_metrics.json` and `results/asvspoof_baseline_scores.csv`
- Total inference time: 1,298 s on CPU (throughput: 0.0182 s/sample)

**GitHub Evidence:**

| File | Description |
|---|---|
| [`results/asvspoof_baseline_metrics.json`](https://github.com/divishamb123/SKIT-AI-2023-2027-07/blob/main/results/asvspoof_baseline_metrics.json) | EER = 0.992%, threshold = 1.7633, 71,237 samples |
| [`results/asvspoof_baseline_scores.csv`](https://github.com/divishamb123/SKIT-AI-2023-2027-07/blob/main/results/asvspoof_baseline_scores.csv) | Per-utterance scores (utt_id, label, spoof_logit, bonafide_logit) |
| [`scripts/evaluate_audio_baseline.py`](https://github.com/divishamb123/SKIT-AI-2023-2027-07/blob/main/scripts/evaluate_audio_baseline.py) | Evaluation script used to produce these results |

**Branch:** `dev-khandelwal/sprint-2-audio`
**Sprint 2 %:** 98% (+7%)

---

## Sprint 2 Status: ✅ COMPLETE (98%)

| Form-2 Sprint 2 Deliverable | Status |
|---|---|
| Built audio preprocessing with librosa and torchaudio | ✅ Done - `audio_preprocessor.py` |
| Integrated pre-trained AASIST-L model | ✅ Done - `audio_detector.py`, Docker volume mount |
| Implemented background worker for audio jobs | ✅ Done - `queue_consumer.py` with retry/timeout |
| Recorded baseline EER | ✅ Done - EER 0.992% on 71,237 ASVspoof 2019 LA eval samples |

---

## Verification Links

- **Repository:** https://github.com/divishamb123/SKIT-AI-2023-2027-07
- **Sprint-1 branch:** https://github.com/divishamb123/SKIT-AI-2023-2027-07/tree/dev-khandelwal/sprint-1-foundation
- **Sprint-2 branch:** https://github.com/divishamb123/SKIT-AI-2023-2027-07/tree/dev-khandelwal/sprint-2-audio
- **Individual audit report:** [`DEV_KHANDELWAL_INDIVIDUAL_WORK_AUDIT.md`](https://github.com/divishamb123/SKIT-AI-2023-2027-07/blob/main/DEV_KHANDELWAL_INDIVIDUAL_WORK_AUDIT.md)
- **Audio API contract:** [`docs/dev-khandelwal/AUDIO_API_CONTRACT.md`](https://github.com/divishamb123/SKIT-AI-2023-2027-07/blob/main/docs/dev-khandelwal/AUDIO_API_CONTRACT.md)
- **Model setup guide:** [`docs/dev-khandelwal/MODEL_SETUP.md`](https://github.com/divishamb123/SKIT-AI-2023-2027-07/blob/main/docs/dev-khandelwal/MODEL_SETUP.md)

---

*Last updated: 29 October 2026 - Dev Khandelwal (23ESKCA035)*
