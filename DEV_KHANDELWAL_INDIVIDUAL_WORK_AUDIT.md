# Dev Khandelwal (23ESKCA035) — Project Contribution & Sprint Status

**Name:** Dev Khandelwal
**College ID:** 23ESKCA035
**GitHub:** Dev-learner-17 (`devkhandelwal61@gmail.com`)
**Role:** Team Lead — backend engineering, audio detection service, fusion layer
**Repository:** https://github.com/divishamb123/SKIT-AI-2023-2027-07
**Last updated:** 05 November 2026

---

## My Role

As Team Lead I am responsible for:
- Complete backend infrastructure (API gateway, auth, database, object storage, job queue)
- Audio detection service (AASIST-L model, preprocessing, background worker)
- Fusion layer (Sprint 5)

---

## Form-2 Responsibilities

| Sprint | Dates | Deliverable |
|---|---|---|
| Sprint 1 | 03-Aug-2026 to 20-Sep-2026 | Set up Docker stack. Created database schema. Built registration, login and JWT authentication. Built file upload and job tracking pipeline. |
| Sprint 2 | 21-Sep-2026 to 08-Nov-2026 | Built audio preprocessing with librosa and torchaudio. Integrated pre-trained AASIST-L model. Implemented background worker for audio jobs. Recorded baseline EER. |
| Sprint 3 | 09-Nov-2026 to 31-Dec-2026 | Tuned audio decision threshold. Measured EER on ASVspoof 2019 LA against 1.0% target. Tested generalisation on In-the-Wild dataset. Added retry and timeout handling. |
| Sprint 4 | 01-Jan-2027 to 24-Jan-2027 | Generated Mel-spectrogram images for each audio verdict. Stored them in MinIO and returned references through the API. |
| Sprint 5 | 25-Jan-2027 to 14-Feb-2027 | Implemented weighted fusion of text, image and audio scores. Calibrated weights on validation data. |
| Sprint 6 | 15-Feb-2027 to 15-Mar-2027 | Security review. Clean rebuild verification. Installation and API documentation. Final project report. |

---

## Git Identity

All my commits are under:
- **Name:** `Dev-learner-17`
- **Email:** `devkhandelwal61@gmail.com`

---

## Sprint 1 — COMPLETE (100%)

**What I built:**

| Component | File(s) | Commit |
|---|---|---|
| DB models (users, jobs, reports) | `shared/db/models/` | `c0edf0f` |
| Alembic migration | `shared/db/alembic/versions/001_initial_migration.py` | `c0edf0f` |
| Auth service (register, login, JWT, bcrypt, Redis) | `services/auth-service/` | `314b1b2` |
| API Gateway (upload, MinIO, RabbitMQ, rate limiting) | `services/api-gateway/` | `12ff443` |
| Job queue and tracking | `services/api-gateway/app/api/jobs.py` | `b6abf37` |
| Docker Compose stack | `docker-compose.yml` | `f67ef9b` |
| CI pipeline | `.github/workflows/ci.yml` | `93cd342` |
| E2E verification scripts | `tests/e2e/` | `93cd342` |

**Tests:** Auth unit tests (`tests/unit/test_auth.py`, `test_security.py`) pass.

---

## Sprint 2 — COMPLETE (100%)

**What I built:**

| Form-2 Deliverable | File(s) | Commit | Status |
|---|---|---|---|
| Audio preprocessing (librosa + torchaudio) | `services/audio-service/app/services/audio_preprocessor.py` | `905f3cb` | Done |
| AASIST-L model integration | `services/audio-service/app/services/audio_detector.py`, `app/models/aasist.py` | `1642630` | Done |
| Background worker (retry + timeout) | `services/audio-service/app/services/queue_consumer.py` | `1b97099` | Done |
| Baseline EER measurement | `results/asvspoof_baseline_metrics.json`, `results/asvspoof_baseline_scores.csv` | `905f3cb` / `e72cafa` | Done |

**Key results:**
- AASIST-L EER = **0.992%** on 71,237 ASVspoof 2019 LA eval utterances (target: ≤ 1.0%)
- Worker: MAX_RETRIES=3, exponential back-off 1s→2s→4s, inference timeout 30s, download timeout 20s
- Degraded mode: service starts without model weights, returns HTTP 503, marks jobs FAILED cleanly

**Tests:** 22 unit tests pass (12 retry + 10 detector).

---

## Sprint 3 — NOT STARTED

Begins 09-Nov-2026.

---

## Sprint 4 — NOT STARTED

Begins 01-Jan-2027.

---

## Sprint 5 — NOT STARTED

Begins 25-Jan-2027.

---

## Sprint 6 — NOT STARTED

Begins 15-Feb-2027.

---

## Known Limitations / Honest Notes

- The image consumer (`services/stub-consumer/`) was a placeholder (asyncio.sleep stub) during Sprint 1–2. It has been replaced with a real implementation that calls the image-service for detection.
- The text detection model (`microsoft/deberta-v3-base`) is currently loaded with a **randomly initialized classifier head** — it has not been fine-tuned yet. Fine-tuning is Sprint 3 work for Aryansh.
- E2E tests (`tests/e2e/`) require the full Docker Compose stack to be running. They have not been run in CI because the GitHub Actions runner doesn't have Docker Compose with all services.
- AASIST-L weights are not in the repository (85 MB, gitignored). The service starts in degraded mode without them.

---

## Verification

Sprint 2 verification script: `python scripts/verify_sprint2_week5_audio.py`
Result: 21/21 checks PASS.
