# Dev Khandelwal — Weekly Progress Report

**Name:** Dev Khandelwal
**College ID:** 23ESKCA035
**GitHub:** Dev-learner-17
**Role:** Team Lead — audio detection service, complete backend, fusion layer
**Repository:** https://github.com/divishamb123/SKIT-AI-2023-2027-07

> All task descriptions are taken verbatim from Form-2. Each entry includes a commit SHA as proof of work.

---

## Sprint 1 — Foundation & Data Preparation

**Form-2 dates:** 03-Aug-2026 to 20-Sep-2026
**Form-2 deliverable:** Set up Docker stack (PostgreSQL, RabbitMQ, MinIO). Created database schema. Built registration, login and JWT authentication. Built file upload and job tracking pipeline.

---

### Week of 03 September 2026

**Form-2 task:** Set up Docker stack (PostgreSQL, RabbitMQ, MinIO). Created database schema. Built registration, login and JWT authentication.

| Commit | Date | Description |
|---|---|---|
| [`c0edf0f`](https://github.com/divishamb123/SKIT-AI-2023-2027-07/commit/c0edf0fd43bea1c355d94c21c9b2ce6c80b67f1b) | 03-Sep-2026 | add backend infrastructure foundation |
| [`314b1b2`](https://github.com/divishamb123/SKIT-AI-2023-2027-07/commit/314b1b2bd71d90fe420d67c569b4b282a652e42c) | 03-Sep-2026 | add authentication and token management |
| [`12ff443`](https://github.com/divishamb123/SKIT-AI-2023-2027-07/commit/12ff443806ffa3cd530d1f3ef5085d210a09f4dd) | 03-Sep-2026 | add API Gateway foundation |
| [`b6abf37`](https://github.com/divishamb123/SKIT-AI-2023-2027-07/commit/b6abf37734d0e1202d8169ddd912da78e3ecbf77) | 03-Sep-2026 | add job queue and tracking pipeline |
| [`93cd342`](https://github.com/divishamb123/SKIT-AI-2023-2027-07/commit/93cd342fbdd366e50b9ba266d5729d98bd8bc30b) | 03-Sep-2026 | add backend verification tests |

Branch: `dev-khandelwal/sprint-1-foundation`

---

### Week of 10 September 2026

**Form-2 task:** Built file upload and job tracking pipeline.

| Commit | Date | Description |
|---|---|---|
| [`f67ef9b`](https://github.com/divishamb123/SKIT-AI-2023-2027-07/commit/f67ef9bd546b2f2816d4357d40e08e051f78945d) | 10-Sep-2026 | complete Docker application integration, environment verification, and E2E testing |
| [`0f28eef`](https://github.com/divishamb123/SKIT-AI-2023-2027-07/commit/0f28eeff9363449006066baa93b551e05d1fbbaa) | 10-Sep-2026 | sync latest branch state |

**Sprint 1: COMPLETE (100%)**

---

## Sprint 2 — Baseline Detection Services

**Form-2 dates:** 21-Sep-2026 to 08-Nov-2026
**Form-2 deliverable:** Built audio preprocessing with librosa and torchaudio. Integrated pre-trained AASIST-L model. Implemented background worker for audio jobs. Recorded baseline EER.

---

### Week 1 — 08 October 2026

**Form-2 task:** Built audio preprocessing with librosa and torchaudio. Integrated pre-trained AASIST-L model.

Work done:
- Fixed variable unpacking bug in audio worker that was preventing it from running
- Fixed payload contract mismatch between API Gateway and RabbitMQ consumer
- Added `/audio` file upload endpoint to API Gateway
- Added `QUEUE_AUDIO_DETECTION` config
- Added unit tests for `AudioPreprocessor`
- Documented audio API contract

| Commit | Date | Description |
|---|---|---|
| [`905f3cb`](https://github.com/divishamb123/SKIT-AI-2023-2027-07/commit/905f3cb82903c0cb332fd3f961008b7b28f1f0cd) | 24-Sep-2026 | implement AASIST-L baseline, preprocessing, and background worker |
| [`2a27c42`](https://github.com/divishamb123/SKIT-AI-2023-2027-07/commit/2a27c424a9a6868121585186214398d074565850) | 06-Oct-2026 | fix audio worker bugs and add API endpoint for week 1 |

Tests: 3 passed (`tests/unit/test_audio_service.py`) | Sprint 2: **77%**

---

### Week 2 — 15 October 2026

**Form-2 task:** Integrated pre-trained AASIST-L model.

Work done:
- Refactored `audio_detector.py`: model path now from `AASIST_WEIGHTS_PATH` env var (not hardcoded)
- Added graceful degraded-mode: service starts when weights absent, `/internal/health` returns HTTP 503
- Added `is_ready()` guard in queue consumer — marks jobs FAILED instead of crashing
- Added `./models:/models:ro` volume mount to `docker-compose.yml`
- Wrote `docs/dev-khandelwal/MODEL_SETUP.md`
- 10 new unit tests for degraded-mode and healthy-mode inference

| Commit | Date | Description |
|---|---|---|
| [`1642630`](https://github.com/divishamb123/SKIT-AI-2023-2027-07/commit/1642630ba973cedfa7d35763f4b8acc4f265bb6a) | 14-Oct-2026 | Docker-ready AASIST-L with graceful degradation |

Tests: 13 passed (10 detector + 3 preprocessor) | Sprint 2: **84%**

---

### Week 3 — 22 October 2026

**Form-2 task:** Implemented background worker for audio jobs.

Work done:
- Added exponential back-off retry (1s→2s→4s, MAX_RETRIES=3) in `queue_consumer.py`
- Added inference timeout (30s) via `asyncio.wait_for()`
- Added MinIO download timeout (20s)
- Jobs that exhaust all retries marked FAILED with descriptive error message
- All constants as env vars in `docker-compose.yml`
- 12 unit tests for retry/back-off logic

| Commit | Date | Description |
|---|---|---|
| [`1b97099`](https://github.com/divishamb123/SKIT-AI-2023-2027-07/commit/1b97099e5e0ae95c479be129f0f932603dff1d9b) | 16-Oct-2026 | implement background worker for audio jobs with retry and timeout handling |

Tests: 22 passed (12 retry + 10 detector) | Sprint 2: **91%**

---

### Week 4 — 29 October 2026

**Form-2 task:** Recorded baseline EER.

Work done:
- Ran `scripts/evaluate_audio_baseline.py` on full ASVspoof 2019 LA evaluation set (71,237 utterances)
- AASIST-L baseline EER: **0.992%** — below the 1.0% target
- EER threshold: 1.7633
- Total inference time: 1,298s on CPU (0.0182 s/sample)
- Results committed to `results/asvspoof_baseline_metrics.json` and `results/asvspoof_baseline_scores.csv`

| File | Description |
|---|---|
| [`results/asvspoof_baseline_metrics.json`](https://github.com/divishamb123/SKIT-AI-2023-2027-07/blob/main/results/asvspoof_baseline_metrics.json) | EER = 0.992%, threshold = 1.7633, 71,237 samples |
| [`results/asvspoof_baseline_scores.csv`](https://github.com/divishamb123/SKIT-AI-2023-2027-07/blob/main/results/asvspoof_baseline_scores.csv) | Per-utterance scores |

Sprint 2: **98%**

---

### Week 5 — 05 November 2026

**Form-2 task:** Built audio preprocessing with librosa and torchaudio. Recorded baseline EER.

Work done:
- Created `scripts/verify_sprint2_week5_audio.py` — verification script for all 4 Sprint 2 deliverables
- Ran verification: **21/21 checks PASS**
- Updated `DEV_KHANDELWAL_INDIVIDUAL_WORK_AUDIT.md` and this report

| File | Description |
|---|---|
| [`scripts/verify_sprint2_week5_audio.py`](https://github.com/divishamb123/SKIT-AI-2023-2027-07/blob/main/scripts/verify_sprint2_week5_audio.py) | 21/21 checks PASS |

Sprint 2: **100% — COMPLETE**

---

## Sprint 2 Status: COMPLETE (100%)

| Form-2 Deliverable | Status |
|---|---|
| Built audio preprocessing with librosa and torchaudio | Done — `audio_preprocessor.py` |
| Integrated pre-trained AASIST-L model | Done — `audio_detector.py`, Docker volume mount |
| Implemented background worker for audio jobs | Done — `queue_consumer.py` with retry/timeout |
| Recorded baseline EER | Done — EER 0.992% on 71,237 ASVspoof 2019 LA samples |

---

## Verification Links

- Repository: https://github.com/divishamb123/SKIT-AI-2023-2027-07
- Audit report: [`DEV_KHANDELWAL_INDIVIDUAL_WORK_AUDIT.md`](https://github.com/divishamb123/SKIT-AI-2023-2027-07/blob/main/DEV_KHANDELWAL_INDIVIDUAL_WORK_AUDIT.md)
- Audio API contract: [`docs/dev-khandelwal/AUDIO_API_CONTRACT.md`](https://github.com/divishamb123/SKIT-AI-2023-2027-07/blob/main/docs/dev-khandelwal/AUDIO_API_CONTRACT.md)

---

## Sprint 3 — Accuracy Optimisation

**Form-2 dates:** 09-Nov-2026 to 31-Dec-2026  
**Form-2 deliverable:** Tuned audio decision threshold. Measured EER on ASVspoof 2019 LA against 1.0% target. Tested generalisation on In-the-Wild dataset. Added retry and timeout handling.

---

### Week 1 — 15 November 2026

**Form-2 task:** Tuned audio decision threshold.

Work done:
- Implemented `ThresholdCalibrator` class in `services/audio-service/app/services/threshold_calibrator.py`
- 3 operating profiles tuned from ASVspoof 2019 LA EER evaluation:
  - `balanced` τ = 1.7633 — EER operating point (EER = 0.992%, Sprint 2 baseline)
  - `high_security` τ = 2.4 — reduced false acceptance for access-control use cases
  - `forensic` τ = 0.8 — reduced missed detections for investigative use cases
- Temperature-scaled sigmoid calibration: P(bonafide) = sigmoid((score − τ) / T), T = 1.5
- At decision boundary: P = 0.5 exactly (maximum uncertainty)
- Confidence tier output: HIGH (P ≥ 0.90), MEDIUM (P ≥ 0.70), LOW (P < 0.70)
- 33 unit tests in `tests/unit/test_threshold_calibration.py` — all pass

| File | Description |
|---|---|
| [`services/audio-service/app/services/threshold_calibrator.py`](https://github.com/divishamb123/SKIT-AI-2023-2027-07/blob/main/services/audio-service/app/services/threshold_calibrator.py) | ThresholdCalibrator — 3 operating modes, sigmoid calibration |
| [`tests/unit/test_threshold_calibration.py`](https://github.com/divishamb123/SKIT-AI-2023-2027-07/blob/main/tests/unit/test_threshold_calibration.py) | 33 unit tests (profiles, verdicts, math, tiers, batch, metadata) |
| [`scripts/verify_sprint3_week1_audio.py`](https://github.com/divishamb123/SKIT-AI-2023-2027-07/blob/main/scripts/verify_sprint3_week1_audio.py) | Standalone verifier — 22/22 checks PASS |

Sprint 3: **14%**

---

## Verification Links

- Repository: https://github.com/divishamb123/SKIT-AI-2023-2027-07
- Audit report: [`DEV_KHANDELWAL_INDIVIDUAL_WORK_AUDIT.md`](https://github.com/divishamb123/SKIT-AI-2023-2027-07/blob/main/DEV_KHANDELWAL_INDIVIDUAL_WORK_AUDIT.md)
- Audio API contract: [`docs/dev-khandelwal/AUDIO_API_CONTRACT.md`](https://github.com/divishamb123/SKIT-AI-2023-2027-07/blob/main/docs/dev-khandelwal/AUDIO_API_CONTRACT.md)

*Last updated: 11 November 2026 — Dev Khandelwal (23ESKCA035)*
