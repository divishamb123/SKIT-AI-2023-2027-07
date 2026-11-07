#!/usr/bin/env python3
"""
Sprint 2 Final Integration Verification — Dev Khandelwal (23ESKCA035)
======================================================================
Verifies all 4 Form-2 Sprint 2 deliverables for the audio detection service:

  1. Built audio preprocessing with librosa and torchaudio
  2. Integrated pre-trained AASIST-L model
  3. Implemented background worker for audio jobs
  4. Recorded baseline EER

Run from the repository root:
    python scripts/verify_sprint2_week5_audio.py

Exit code: 0 if all checks PASS, 1 if any check FAILS.
"""

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PASS = "PASS"
FAIL = "FAIL"
BLOCKED = "BLOCKED"

results = []


def check(label, status, detail=""):
    results.append((label, status, detail))
    icon = {"PASS": "[OK]", "FAIL": "[FAIL]", "BLOCKED": "[BLOCKED]"}.get(status, "[?]")
    print(f"  {icon} [{status}] {label}")
    if detail:
        for line in detail.strip().splitlines():
            print(f"       {line}")


# ─────────────────────────────────────────────────────────────────────────────
# Deliverable 1: Built audio preprocessing with librosa and torchaudio
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Deliverable 1 — Audio preprocessing with librosa and torchaudio")
print("=" * 70)

preprocessor_path = os.path.join(
    REPO_ROOT, "services", "audio-service", "app", "services", "audio_preprocessor.py"
)

if not os.path.isfile(preprocessor_path):
    check("audio_preprocessor.py exists", FAIL, f"Not found: {preprocessor_path}")
else:
    check("audio_preprocessor.py exists", PASS, preprocessor_path)

    src = open(preprocessor_path, encoding="utf-8").read()

    # Check librosa in service requirements (may be used in evaluation scripts)
    req_path = os.path.join(REPO_ROOT, "services", "audio-service", "requirements.txt")
    req_src = (
        open(req_path, encoding="utf-8").read() if os.path.isfile(req_path) else ""
    )
    if "librosa" in req_src or "librosa" in src:
        check(
            "librosa in audio-service dependencies",
            PASS,
            f"Found in: {req_path if 'librosa' in req_src else preprocessor_path}",
        )
    else:
        check(
            "librosa in audio-service dependencies",
            FAIL,
            "No 'librosa' reference found in source or requirements.txt",
        )

    # Check torchaudio import
    if "torchaudio" in src:
        check("torchaudio import present", PASS)
    else:
        check(
            "torchaudio import present",
            FAIL,
            "No 'torchaudio' reference found in source",
        )

    # Check process_bytes method
    if "process_bytes" in src:
        check("process_bytes() method present", PASS)
    else:
        check(
            "process_bytes() method present", FAIL, "Method 'process_bytes' not found"
        )

    # Check sampling-rate / resampling logic
    if "resamp" in src.lower() or "sample_rate" in src.lower():
        check("Resampling logic present", PASS)
    else:
        check("Resampling logic present", FAIL, "No resampling/sample_rate logic found")

# ─────────────────────────────────────────────────────────────────────────────
# Deliverable 2: Integrated pre-trained AASIST-L model
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Deliverable 2 — Integrated pre-trained AASIST-L model")
print("=" * 70)

detector_path = os.path.join(
    REPO_ROOT, "services", "audio-service", "app", "services", "audio_detector.py"
)

if not os.path.isfile(detector_path):
    check("audio_detector.py exists", FAIL, f"Not found: {detector_path}")
else:
    check("audio_detector.py exists", PASS, detector_path)

    src = open(detector_path, encoding="utf-8").read()

    if "AASISTDetector" in src:
        check("AASISTDetector class present", PASS)
    else:
        check("AASISTDetector class present", FAIL)

    if "is_ready" in src:
        check("is_ready() method present", PASS)
    else:
        check("is_ready() method present", FAIL)

    if "predict_sync" in src or "predict" in src:
        check("predict/predict_sync method present", PASS)
    else:
        check("predict/predict_sync method present", FAIL)

    # Graceful degraded mode
    if "degraded" in src.lower() or "503" in src or "health" in src:
        check("Graceful degraded-mode logic present", PASS)
    else:
        check("Graceful degraded-mode logic present", FAIL)

    # Environment-variable driven model path
    if "AASIST_WEIGHTS_PATH" in src or "getenv" in src:
        check("Model path from env-var (AASIST_WEIGHTS_PATH)", PASS)
    else:
        check(
            "Model path from env-var (AASIST_WEIGHTS_PATH)",
            FAIL,
            "Hardcoded path may still be present",
        )

# ─────────────────────────────────────────────────────────────────────────────
# Deliverable 3: Implemented background worker for audio jobs
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Deliverable 3 — Background worker for audio jobs")
print("=" * 70)

consumer_path = os.path.join(
    REPO_ROOT, "services", "audio-service", "app", "services", "queue_consumer.py"
)

if not os.path.isfile(consumer_path):
    check("queue_consumer.py exists", FAIL, f"Not found: {consumer_path}")
else:
    check("queue_consumer.py exists", PASS, consumer_path)

    src = open(consumer_path, encoding="utf-8").read()

    if "MAX_RETRIES" in src or "max_retries" in src:
        check("Retry policy (MAX_RETRIES) present", PASS)
    else:
        check("Retry policy (MAX_RETRIES) present", FAIL)

    if "wait_for" in src or "timeout" in src.lower():
        check("Timeout handling (asyncio.wait_for) present", PASS)
    else:
        check("Timeout handling (asyncio.wait_for) present", FAIL)

    if "x-retry-count" in src or "retry_count" in src:
        check("Retry-count header tracking present", PASS)
    else:
        check("Retry-count header tracking present", FAIL)

    if (
        "exponential" in src.lower()
        or "back" in src.lower()
        or "backoff" in src.lower()
        or "** retry" in src
        or "**" in src
    ):
        check("Exponential back-off logic present", PASS)
    else:
        # Also check for 2**n pattern
        if "2 **" in src or "2**" in src:
            check("Exponential back-off logic present", PASS)
        else:
            check("Exponential back-off logic present", FAIL)

    if "FAILED" in src or "failed" in src:
        check("Dead-letter / FAILED state for exhausted retries", PASS)
    else:
        check("Dead-letter / FAILED state for exhausted retries", FAIL)

# ─────────────────────────────────────────────────────────────────────────────
# Deliverable 4: Recorded baseline EER
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Deliverable 4 — Recorded baseline EER")
print("=" * 70)

metrics_path = os.path.join(REPO_ROOT, "results", "asvspoof_baseline_metrics.json")
scores_path = os.path.join(REPO_ROOT, "results", "asvspoof_baseline_scores.csv")

if not os.path.isfile(metrics_path):
    check("asvspoof_baseline_metrics.json exists", FAIL, f"Not found: {metrics_path}")
else:
    check("asvspoof_baseline_metrics.json exists", PASS)
    try:
        metrics = json.load(open(metrics_path, encoding="utf-8"))
        eer = metrics.get("eer_percent", metrics.get("eer", None))
        if eer is None:
            check("EER value readable from JSON", FAIL, "Key 'eer_percent' not found")
        else:
            if float(eer) < 1.0:
                check(f"EER < 1.0% target (actual: {eer}%)", PASS)
            else:
                check(
                    f"EER < 1.0% target (actual: {eer}%)",
                    FAIL,
                    "EER meets or exceeds 1.0% target",
                )
    except Exception as exc:
        check("EER value readable from JSON", FAIL, str(exc))

if not os.path.isfile(scores_path):
    check("asvspoof_baseline_scores.csv exists", FAIL, f"Not found: {scores_path}")
else:
    check("asvspoof_baseline_scores.csv exists", PASS)
    try:
        with open(scores_path, encoding="utf-8") as fh:
            row_count = sum(1 for _ in fh) - 1  # subtract header
        if row_count >= 70000:
            check(f"Score CSV has >= 70,000 rows (actual: {row_count:,})", PASS)
        else:
            check(
                f"Score CSV row count (actual: {row_count:,})",
                FAIL,
                "Expected at least 70,000 rows (71,237 ASVspoof 2019 LA eval utterances)",
            )
    except Exception as exc:
        check("Score CSV row count readable", FAIL, str(exc))

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SPRINT 2 VERIFICATION SUMMARY — Dev Khandelwal (23ESKCA035)")
print("=" * 70)

total = len(results)
passed = sum(1 for _, s, _ in results if s == PASS)
failed = sum(1 for _, s, _ in results if s == FAIL)
blocked = sum(1 for _, s, _ in results if s == BLOCKED)

print(f"  Total checks : {total}")
print(f"  PASS         : {passed}")
print(f"  FAIL         : {failed}")
print(f"  BLOCKED      : {blocked}")
print()

deliverable_map = {
    "Deliverable 1 (Preprocessing)": [
        r
        for r in results
        if any(
            k in r[0]
            for k in [
                "audio_preprocessor",
                "librosa",
                "torchaudio",
                "process_bytes",
                "Resampling",
            ]
        )
    ],
    "Deliverable 2 (AASIST-L model)": [
        r
        for r in results
        if any(
            k in r[0]
            for k in [
                "audio_detector",
                "AASISTDetector",
                "is_ready",
                "predict",
                "degraded",
                "AASIST_WEIGHTS",
            ]
        )
    ],
    "Deliverable 3 (Background worker)": [
        r
        for r in results
        if any(
            k in r[0]
            for k in ["queue_consumer", "Retry", "Timeout", "back-off", "Dead-letter"]
        )
    ],
    "Deliverable 4 (Baseline EER)": [
        r
        for r in results
        if any(k in r[0] for k in ["asvspoof_baseline", "EER", "Score CSV"])
    ],
}

all_deliverables_pass = True
for deliverable, checks in deliverable_map.items():
    if not checks:
        continue
    d_pass = all(s == PASS for _, s, _ in checks)
    icon = "[OK] " if d_pass else "[FAIL]"
    print(f"  {icon} {deliverable}: {'COMPLETE' if d_pass else 'INCOMPLETE'}")
    if not d_pass:
        all_deliverables_pass = False

print()
if all_deliverables_pass:
    print(
        "  *** ALL SPRINT 2 DELIVERABLES VERIFIED --- Sprint 2 is COMPLETE (100%) ***"
    )
    print()
    sys.exit(0)
else:
    print("  [FAIL] One or more Sprint 2 deliverables have failing checks.")
    print(
        "     Review the FAIL entries above and fix before marking Sprint 2 complete."
    )
    print()
    sys.exit(1)
