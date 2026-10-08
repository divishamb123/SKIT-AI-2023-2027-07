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


# # #   W e e k   1   � �    W e e k l y   S u b m i s s i o n  
  
 * * D a t e : * *   0 8   O c t o b e r   2 0 2 6  
 * * P l a n n e d   m i l e s t o n e : * *   F i x   a u d i o   w o r k e r   b u g s   a n d   a d d   A P I   e n d p o i n t   f o r   W e e k   1 .  
 * * A c t u a l   w o r k   c o m p l e t e d : * *    
 -   R e s o l v e d   o b j e c t _ k e y   p a y l o a d   c o n t r a c t   m i s m a t c h   b e t w e e n   A P I   G a t e w a y   a n d   R a b b i t M Q   c o n s u m e r .  
 -   F i x e d   a   c r i t i c a l   u n p a c k i n g   b u g   i n   q u e u e _ c o n s u m e r . p y   w h e r e   i t   e x p e c t e d   t w o   v a r i a b l e s   f r o m   p r o c e s s _ b y t e s .  
 -   I m p l e m e n t e d   t h e   ` / a u d i o `   f i l e   u p l o a d   e n d p o i n t   i n   A P I   G a t e w a y   ( ` d e t e c t . p y ` ) ,   c o m p l e t i n g   t h e   f i n a l   S p r i n t   1   f o u n d a t i o n   g a p   f o r   a u d i o .  
 -   D o c u m e n t e d   t h e   a u d i o   A P I   c o n t r a c t .  
 -   A d d e d   m e a n i n g f u l   u n i t   t e s t s   f o r   ` A u d i o P r e p r o c e s s o r ` .  
 * * F i l e s   c h a n g e d : * *    
 -   ` s e r v i c e s / a p i - g a t e w a y / a p p / a p i / d e t e c t . p y `  
 -   ` s e r v i c e s / a p i - g a t e w a y / a p p / c o r e / c o n f i g . p y `  
 -   ` s e r v i c e s / a u d i o - s e r v i c e / a p p / s e r v i c e s / q u e u e _ c o n s u m e r . p y `  
 -   ` t e s t s / u n i t / t e s t _ a u d i o _ s e r v i c e . p y `  
 -   ` d o c s / d e v - k h a n d e l w a l / A U D I O _ A P I _ C O N T R A C T . m d `  
 * * T e s t s   e x e c u t e d : * *   ` t e s t s / u n i t / t e s t _ a u d i o _ s e r v i c e . p y `   u s i n g   p y t e s t .  
 * * T e s t   r e s u l t s : * *   3   p a s s e d .  
 * * C o m m i t   S H A : * *   ` 2 a 2 7 c 4 2 `  
 * * B r a n c h : * *   ` d e v - k h a n d e l w a l / s p r i n t - 2 - a u d i o `  
 * * G i t H u b   p u s h   s t a t u s : * *   S u c c e s s .  
 * * S p r i n t   1   p e r c e n t a g e : * *   1 0 0 %  
 * * S p r i n t   2   p e r c e n t a g e : * *   7 7 %   ( + 7 % )  
 * * O v e r a l l   p e r c e n t a g e : * *   4 8 %   ( + 7 % )  
 * * R e m a i n i n g   w o r k : * *    
 -   M o u n t   t h e   p r e t r a i n e d   A A S I S T - L   w e i g h t s   i n   D o c k e r   f o r   S p r i n t   2 .  
 -   E x e c u t e   S p r i n t   3   ( t i m e o u t / r e t r i e s ,   t u n i n g ) .  
 -   E x p o s e   s p e c t r o g r a m   U R L s   v i a   A P I   ( S p r i n t   4 ) .  
 -   B u i l d   t h e   f u s i o n   m o d u l e   ( S p r i n t   5 ) .  
 * * N e x t   w e e k ' s   p l a n n e d   w o r k : * *   M a k e   t h e   A A S I S T - L   a u d i o   s e r v i c e   a n d   w o r k e r   e x e c u t a b l e   v i a   D o c k e r   b y   f i x i n g   t h e   m o d e l   m o u n t s .  
 

### Week-1 Finalization
**Week 1 completion date:** 08 October 2026
**Implementation commit SHA(s):** a27c42**Report commit SHA:** \c33d349**PR number:** #2
**PR URL:** https://github.com/divishamb123/SKIT-AI-2023-2027-07/pull/2
**Merge commit SHA:** \aebd8d**Merge date:** 06 October 2026
**Source branch:** \dev-khandelwal/sprint-2-audio**Target branch:** \main**Tests:** Executed unit tests for AudioPreprocessor via pytest (3 passed, 0 failed).
**Final verified Sprint-1 percentage:** 100%
**Final verified Sprint-2 percentage:** 77%
**Overall percentage:** 48% (Calculation: Sprint 1 [100%], Sprint 2 [77%], Sprint 4 [80%]. Others [0-5%])
**Production-code additions:** +84 lines
**Test-code additions:** +36 lines
**Documentation additions:** +35 lines
**Total additions/deletions:** 155 additions / 3 deletions (excluding report updates)
**Remaining Sprint-2 work:**
- Mount the pretrained AASIST-L weights in Docker configuration to enable container execution.
- End-to-end testing of the complete audio pipeline running under Docker Compose.
**Future sprint work:**
- Sprint 3: Threshold tuning, ASVspoof 2019 LA evaluation, In-the-Wild generalisation testing, Retry/timeout handling.
- Sprint 4: Final API endpoint integration for MinIO Mel-spectrogram references (currently partially complete).
- Sprint 5: Weighted text/image/audio fusion, Weight calibration.
- Sprint 6: Security review, Final installation documentation, Final report.
**Exact Week-2 plan:** Make the AASIST-L audio service and worker completely executable via Docker. I will focus on fixing the Docker configuration for the AASIST-L pretrained weights, modifying docker-compose.yml and model loading to properly load the model without hardcoded absolute paths, and ensure the container starts successfully.
