# Dev Khandelwal (23ESKCA035) — Individual Project Contribution & Sprint Status Report

## 1. Executive Summary
This report presents a comprehensive, evidence-based forensic audit of Dev Khandelwal's individual contributions to the SKIT-AI-2023-2027-07 project as of **06 October 2026**. The audit cross-references official documentation (Form 1, Form 2, Blueprint) with actual Git history and repository codebase. Dev Khandelwal has successfully completed the backend foundation (Sprint 1) and has made significant code contributions to the audio detection service (Sprint 2) and early spectrogram visualization (Sprint 4). However, critical integration bugs and missing Docker configurations currently prevent the audio service from running successfully in the `main` branch. 

## 2. My Official Role
**Team Lead**. Responsible for backend engineering and audio signal processing.

## 3. My Form-1 Responsibilities
According to Form 1:
*   **Audio detection service** (AASIST-L)
*   **Complete backend** (API gateway, authentication, database, object storage, job queue)
*   **Fusion layer**

## 4. My Form-2 Responsibilities
According to Form 2:
*   **Sprint 1 (03-Aug-2026 to 20-Sep-2026):** Set up Docker stack (PostgreSQL, RabbitMQ, MinIO). Created database schema. Built registration, login and JWT authentication. Built file upload and job tracking pipeline.
*   **Sprint 2 (21-Sep-2026 to 08-Nov-2026):** Built audio preprocessing with librosa and torchaudio. Integrated pre-trained AASIST-L model. Implemented background worker for audio jobs. Recorded baseline EER.
*   **Sprint 3 (09-Nov-2026 to 31-Dec-2026):** Tuned audio decision threshold. Measured EER on ASVspoof 2019 LA against 1.0% target. Tested generalisation on In-the-Wild dataset. Added retry and timeout handling.
*   **Sprint 4 (01-Jan-2027 to 24-Jan-2027):** Generated Mel-spectrogram images for each audio verdict. Stored them in MinIO and returned references through the API.
*   **Sprint 5 (25-Jan-2027 to 14-Feb-2027):** Implemented weighted fusion of text, image and audio scores. Calibrated weights on validation data. Compared fused score against individual models.
*   **Sprint 6 (15-Feb-2027 to 15-Mar-2027):** Completed security review and file validation. Verified clean rebuild from repository. Wrote installation and API documentation. Compiled final project report.

## 5. Git Identity
All contributions have been consistently committed under:
*   **Name:** `Dev-learner-17`
*   **Email:** `devkhandelwal61@gmail.com`

There are no commits under the explicit name "Dev Khandelwal", but authorship attribution maps clearly to `Dev-learner-17`.

## 6. Branches Used
*   `dev-khandelwal/sprint-1-foundation`
*   `dev-khandelwal/sprint-2-audio`
*   `main` (All work is currently merged/pushed here)

## 7. Commit History
You have made **15 commits** across the repository.
*   Sprint 1 foundation commits were made between Sept 3 and Sept 10 and formally merged into `main` on Sept 12 (`e5766ac813`).
*   Sprint 2 audio commits were made on Sept 24 and pushed directly to `main` without a pull-request merge commit.

## 8. Sprint 1 Work
**Status:** **[COMPLETE]**
*   **Database/Alembic/Models:** Implemented in `shared/db` (Commit `c0edf0fd43`).
*   **Authentication/JWT:** Implemented in `services/auth-service` (Commit `314b1b2bd7`).
*   **API Gateway:** Implemented in `services/api-gateway` (Commit `12ff443806`). File upload is complete for images, but an `/audio` endpoint is currently missing.
*   **MinIO/RabbitMQ/Docker:** Implemented across `docker-compose.yml` and Python services (Commit `f67ef9bd54`).
*   **Verification tests:** Implemented in `tests/e2e` (Commit `93cd342fbd`).

## 9. Sprint 2 Work
**Status:** **[PARTIALLY COMPLETE / INTEGRATED BUT NEEDS VERIFICATION]**
*   **Preprocessing:** Implemented via `audio_preprocessor.py` (librosa/torchaudio).
*   **AASIST-L Integration:** Model architecture is implemented (`aasist.py`), but the `audio-service/Dockerfile` does not copy or mount the pretrained weights (`/models/aasist_l_v1.0/AASIST-L.pth`). The service currently crashes in Docker.
*   **Background Worker:** Code exists (`queue_consumer.py`), but there is a critical bug: it attempts to unpack two values (`waveform, meta = ...`) from `AudioPreprocessor.process_bytes`, which only returns a single tensor.
*   **Baseline EER:** Evaluated and recorded at 0.99% (`results/asvspoof_baseline_metrics.json`).

## 10. Sprint 3 Status
**Status:** **[NOT STARTED]**
*   **Threshold Tuning:** Hardcoded thresholds currently exist in `audio_detector.py`. No tuning logic is implemented.
*   **In-the-wild testing:** No evidence found.
*   **Retry/Timeout handling:** Missing from `queue_consumer.py`.

## 11. Sprint 4 Status
**Status:** **[PARTIALLY COMPLETE]** *(Ahead of schedule)*
*   **Mel-spectrogram generation:** Implemented early in `spectrogram_visualizer.py` and actively called by the queue consumer.
*   **MinIO Storage:** Successfully uploads `melspec.png` to MinIO.
*   **API References:** Not yet returned through the API Gateway (the `jobs.py` API only returns basic job status, not the explanation artifacts).

## 12. Sprint 5 Status
**Status:** **[NOT STARTED]**
*   Fusion layer not implemented.

## 13. Sprint 6 Status
**Status:** **[NOT STARTED]**
*   Only Sprint 1 documentation exists in `docs/dev-khandelwal/SPRINT_1_STATUS.md`.

## 14. Current Overall Percentage
*   **Sprint 1:** 95%
*   **Sprint 2:** 70%
*   **Sprint 3:** 0%
*   **Sprint 4:** 80% (Completed early)
*   **Sprint 5:** 0%
*   **Sprint 6:** 5% (Initial docs only)
*   **Overall Individual Completion:** **~41%**

## 15. GitHub Contribution Analysis
**The "2565 Lines" Issue**
The contribution count of exactly **2565 additions and 45 deletions** corresponds precisely to the sum of your commits made exclusively on the `dev-khandelwal/sprint-1-foundation` branch (Sept 3 – Sept 10). 

## 16. Why the 2565-line figure is misleading/incomplete
GitHub's contribution graph often lags, relies on default branch merges, or reflects specific Pull Request statistics. Because your Sprint 2 work (`dev-khandelwal/sprint-2-audio`) was committed directly/fast-forwarded to `main` on September 24 without a formal Pull Request merge commit, any check performed prior to Sept 24, or any check scoped only to the Sprint 1 PR, would only yield 2565 lines.

**Actual Git Contribution on `main`:**
*   Total Additions: **4,678 lines**
*   Total Deletions: **286 lines**

## 17. Current Main-Branch Status
Your Sprint 1 and Sprint 2 code is completely merged into `main`. However, because of missing Docker volume mappings for model weights, missing `/audio` API endpoints, and a variable unpacking bug, the audio pipeline does not function end-to-end on `main` out-of-the-box.

## 18. Remaining Work
1. Fix the audio worker bugs.
2. Mount the pretrained AASIST-L weights in Docker.
3. Expose the `/audio` upload endpoint in the API Gateway.
4. Execute Sprint 3 (timeout/retries, tuning).
5. Expose spectrogram URLs via API (Sprint 4).
6. Build the fusion module (Sprint 5).

## 19. Risks / Missing Evidence
*   **Missing Model Weights:** AASIST-L pretrained weights are referenced by absolute path but not included in the Docker context or volume mounts.
*   **Test Scaffolding:** `tests/unit/test_audio_service.py` contains only 1 placeholder test. Real audio tests are missing.
*   **Integration Mismatch:** API Gateway expects `object_key` for images, but `queue_consumer.py` expects `input_object_key` for audio.

## 20. Recommended Next Actions
**Priority 1 — Do immediately**
*   **Task:** Fix variable unpacking bug in `queue_consumer.py`.
*   **Why:** The worker will crash immediately when processing an audio file.
*   **Exact Files:** `services/audio-service/app/services/queue_consumer.py` (Line 50: `waveform, meta = ...` should just be `waveform = ...`).

**Priority 2 — Before Sprint 2 ends (Nov 8)**
*   **Task:** Fix Docker volumes and add `/audio` API endpoint.
*   **Why:** Without these, the audio service cannot be accessed or run via Docker Compose.
*   **Exact Files:** `docker-compose.yml` (add `- ./models:/models` to `audio-service`), `services/api-gateway/app/api/detect.py` (add `@router.post("/audio")`).

**Priority 3 — Sprint 3**
*   **Task:** Implement retries, timeouts, and threshold tuning.
*   **Exact Files:** `audio_detector.py` and `queue_consumer.py`.

## 21. Evidence Appendix
*   **Database Foundation:** `c0edf0fd43` (Main: YES) - `shared/db/models`
*   **Auth Service:** `314b1b2bd7` (Main: YES) - `services/auth-service/app/api/auth.py`
*   **API Gateway:** `12ff443806` (Main: YES) - `services/api-gateway/app/main.py`
*   **AASIST-L Architecture:** `905f3cb829` (Main: YES) - `services/audio-service/app/models/aasist.py`
*   **Baseline EER Evidence:** `905f3cb829` (Main: YES) - `results/asvspoof_baseline_metrics.json`
*   **Spectrogram Generation:** `905f3cb829` (Main: YES) - `services/audio-service/app/services/spectrogram_visualizer.py`

============================================================
## FINAL VERDICT

| Metric | Status |
| :--- | :--- |
| Sprint 1 | 95% |
| Sprint 2 | 70% |
| Sprint 3 | 0% |
| Sprint 4 | 80% (Completed early) |
| Sprint 5 | 0% |
| Sprint 6 | 5% |
| Overall individual completion | ~41% |
| My commits | 15 |
| My additions | 4,678 |
| My deletions | 286 |
| My work currently in main | YES (All commits) |
| Unmerged work | None |
| Current active sprint | Sprint 2 (Ends 08-Nov-2026) |
| Current sprint status | On Track (but requires bug fixes) |
| Most urgent task | Fix `queue_consumer.py` unpacking bug |
| Main missing evidence | Pretrained weights mount in Docker |

**Conclusion:**
Based strictly on Form 1, Form 2, the Functional Requirements, the Engineering Blueprint, and the actual Git history/code, Dev Khandelwal is currently at approximately 41% completion. Sprint 1 is 95%, Sprint 2 is 70%, and the remaining work is largely on track for future sprints, although Sprint 4 (spectrograms) has been implemented significantly ahead of schedule. The immediate priority must be fixing the critical integration bugs preventing the audio service from running cleanly in Docker.


### Week 1 — Weekly Submission

**Date:** 08 October 2026
**Planned milestone:** Fix audio worker bugs and add API endpoint for Week 1.
**Actual work completed:** 
- Resolved object_key payload contract mismatch between API Gateway and RabbitMQ consumer.
- Fixed a critical unpacking bug in queue_consumer.py where it expected two variables from process_bytes.
- Implemented the /audio file upload endpoint in API Gateway (detect.py), completing the final Sprint 1 foundation gap for audio.
- Documented the audio API contract.
- Added meaningful unit tests for AudioPreprocessor.
**Files changed:** 
- services/api-gateway/app/api/detect.py
- services/api-gateway/app/core/config.py
- services/audio-service/app/services/queue_consumer.py
- 	ests/unit/test_audio_service.py
- docs/dev-khandelwal/AUDIO_API_CONTRACT.md
**Tests executed:** 	ests/unit/test_audio_service.py using pytest.
**Test results:** 3 passed.
**Commit SHA:** a27c42**Branch:** dev-khandelwal/sprint-2-audio
**GitHub push status:** Success.
**Sprint 1 percentage:** 100%
**Sprint 2 percentage:** 77% (+7%)
**Overall percentage:** 48% (+7%)
**Remaining work:** 
- Mount the pretrained AASIST-L weights in Docker for Sprint 2.
- Execute Sprint 3 (timeout/retries, tuning).
- Expose spectrogram URLs via API (Sprint 4).
- Build the fusion module (Sprint 5).
**Next week's planned work:** Make the AASIST-L audio service and worker executable via Docker by fixing the model mounts.
