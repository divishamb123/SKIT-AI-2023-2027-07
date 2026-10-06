"""
Phase 5 – Pretrained AASIST-L ASVspoof 2019 LA Baseline EER
============================================================

Usage (Google Colab):
    python scripts/phase5_asvspoof_baseline.py \
        --dataset_root /content/AASIST/data/ASVspoof2019_LA

The LA root must contain:
    ASVspoof2019_LA_cm_protocols/
        ASVspoof2019.LA.cm.eval.trl.txt
    ASVspoof2019_LA_eval/
        flac/
            LA_E_*.flac

Score direction (AASIST-L):
    Model returns (last_hidden, logits).
    Official AASIST scoring uses a single scalar score:
    score = logits[:, 1]  (bona-fide logit)

    Official EER API: compute_eer(target_scores, nontarget_scores)
        target     = score[label == 1]
        nontarget  = score[label == 0]
    Returns (eer_fraction, threshold) – multiply by 100 for percent.
"""

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List, Tuple


import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# PYTHONPATH: project root = one directory above 'scripts/'
# Works for both:
#   /content/AASIST/scripts/phase5_asvspoof_baseline.py  -> /content/AASIST
#   E:/AASIST/scripts/phase5_asvspoof_baseline.py        -> E:/AASIST
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from audio.model.aasist_loader import AASISTLoader
from audio.preprocessing.audio_preprocessor import AudioPreprocessor
from aasist_official.evaluation import (
    compute_eer,
)  # (target_scores, nontarget_scores) -> (eer_fraction, threshold)

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def read_protocol(protocol_path: Path) -> Tuple[int, List[str], List[int]]:
    """Parse ASVspoof 2019 LA evaluation protocol.

    Protocol columns (space-separated):
        0: SPEAKER_ID
        1: UTTERANCE_ID   <- used as the file stem
        2: SYSTEM_ID
        3: ATTACK_TYPE
        4: KEY             ('bonafide' or 'spoof')

    Returns (num_entries, utterance_ids, labels)
        label: 1 = bonafide, 0 = spoof
    """
    ids: List[str] = []
    labels: List[int] = []
    with open(protocol_path, "r", encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                raise ValueError(
                    f"Protocol line {lineno} has fewer than 5 fields: {line!r}"
                )
            utt_id = parts[1]
            key = parts[4].lower()
            ids.append(utt_id)
            labels.append(1 if key == "bonafide" else 0)
    return len(ids), ids, labels


def verify_dataset(
    eval_flac_dir: Path, utt_ids: List[str]
) -> Tuple[int, List[str], List[str]]:
    """Verify that each protocol ID has a readable FLAC file.

    Returns (total_checked, matched_ids, missing_or_corrupt_ids).
    """
    matched: List[str] = []
    missing: List[str] = []
    for uid in utt_ids:
        audio_path = eval_flac_dir / f"{uid}.flac"
        if audio_path.is_file():
            try:
                sf.info(str(audio_path))  # quick header read – raises on corrupt files
                matched.append(uid)
            except Exception:
                missing.append(uid)
        else:
            missing.append(uid)
    return len(matched) + len(missing), matched, missing


def cross_check_eer(bonafide_scores: np.ndarray, spoof_scores: np.ndarray) -> float:
    """Cross-check EER using sklearn if available; fall back to official implementation.
    Always returns EER as a percentage.
    """
    try:
        from sklearn.metrics import roc_curve

        all_scores = np.concatenate([bonafide_scores, spoof_scores])
        all_labels = np.concatenate(
            [np.ones(len(bonafide_scores)), np.zeros(len(spoof_scores))]
        )
        fpr, tpr, _ = roc_curve(all_labels, all_scores, pos_label=1)
        fnr = 1.0 - tpr
        idx = np.nanargmin(np.abs(fpr - fnr))
        return float((fpr[idx] + fnr[idx]) / 2.0 * 100.0)
    except Exception:
        eer_frac, _ = compute_eer(bonafide_scores, spoof_scores)
        return float(eer_frac * 100.0)


# ---------------------------------------------------------------------------
# Batch inference helper
# ---------------------------------------------------------------------------


def run_batch(
    batch_ids: List[str],
    eval_flac_dir: Path,
    protocol_id_to_label: dict,
    model: torch.nn.Module,
    device: torch.device,
    all_scores: list,
    errors: list,
) -> None:
    """Load, preprocess, and infer a single batch; append results in-place.

    AudioPreprocessor.process_file returns [1, 64600].
    Model.forward() does x = x.unsqueeze(1) internally, so it expects [B, 64600].
    We squeeze the channel dim (dim=0) before stacking to get [B, 64600].
    """
    batch_wavs = []
    valid_ids = []
    for b_uid in batch_ids:
        audio_path = eval_flac_dir / f"{b_uid}.flac"
        try:
            proc = AudioPreprocessor.process_file(str(audio_path))  # [1, 64600]
            batch_wavs.append(proc.float().squeeze(0))  # [64600]
            valid_ids.append(b_uid)
        except Exception as exc:
            errors.append({"utt_id": b_uid, "error": str(exc)})

    if not batch_wavs:
        return

    try:
        batch_tensor = torch.stack(batch_wavs).to(device)  # [B, 64600]
        with torch.no_grad():
            _, batch_logits = model(
                batch_tensor
            )  # last_hidden discarded; logits [B, 2]
        for i, b_uid in enumerate(valid_ids):
            spoof_logit = batch_logits[i, 0].item()  # col 0 = spoof
            bonafide_logit = batch_logits[i, 1].item()  # col 1 = bonafide
            label = protocol_id_to_label[b_uid]
            all_scores.append([b_uid, label, spoof_logit, bonafide_logit])
    except Exception as exc:
        for b_uid in valid_ids:
            errors.append({"utt_id": b_uid, "error": str(exc)})


# ---------------------------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 5 - ASVspoof 2019 LA baseline EER using pretrained AASIST-L"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help=(
            "LA root directory (e.g. /content/AASIST/data/ASVspoof2019_LA). "
            "Must contain ASVspoof2019_LA_cm_protocols/ and ASVspoof2019_LA_eval/."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="GPU batch size (default: 64).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for smoke-test utterance selection (default: 42).",
    )
    args = parser.parse_args()

    # -------------------------------------------------------------------
    # Prepare output directories
    # -------------------------------------------------------------------
    results_dir = _PROJECT_ROOT / "results"
    evidence_dir = _PROJECT_ROOT / "evidence" / "baseline"
    results_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------
    # Record environment
    # -------------------------------------------------------------------
    env_info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "project_root": str(_PROJECT_ROOT),
    }
    (evidence_dir / "phase5_environment.txt").write_text(json.dumps(env_info, indent=2))

    # -------------------------------------------------------------------
    # Checkpoint hash (pre-evaluation)
    # -------------------------------------------------------------------
    ckpt_path = (
        _PROJECT_ROOT / "checkpoints" / "pretrained" / "aasist-l" / "AASIST-L.pth"
    )
    if not ckpt_path.is_file():
        sys.exit(f"[FATAL] Checkpoint not found: {ckpt_path}")
    pre_hash = sha256_file(ckpt_path)
    hash_log = evidence_dir / "phase5_checkpoint_hash.txt"
    hash_log.write_text(f"pre_hash={pre_hash}\n")

    # -------------------------------------------------------------------
    # Preprocessing source-code self-check — runs before any real inference.
    #
    # A previous evaluation run (recorded as EER=6.863%) was later found to
    # have used a stale build of AudioPreprocessor that zero-padded short
    # audio instead of tile-repeating it, which does not match official
    # AASIST data_utils.pad(). That stale build is not present in this
    # codebase, but this check exists so a future accidental regression
    # (e.g. extracting the wrong zip) fails loudly instead of silently
    # producing a wrong EER.
    #
    # This is a STATIC TEXT check of the actual source file on disk — it
    # does not fabricate or run any synthetic audio through the pipeline.
    # -------------------------------------------------------------------
    preproc_path = _PROJECT_ROOT / "audio" / "preprocessing" / "audio_preprocessor.py"
    if not preproc_path.is_file():
        sys.exit(f"[FATAL] audio_preprocessor.py not found: {preproc_path}")

    _preproc_src = preproc_path.read_text()
    _uses_zero_pad = (
        "zero" in _preproc_src.lower()
        and "pad" in _preproc_src.lower()
        and (
            "torch.zeros" in _preproc_src
            or "np.zeros" in _preproc_src
            or "F.pad" in _preproc_src
        )
    )
    _uses_repeat = (".repeat(" in _preproc_src) or ("tile" in _preproc_src.lower())

    (evidence_dir / "phase5_preprocessing_selfcheck.txt").write_text(
        f"preprocessor_path={preproc_path}\n"
        f"contains_zero_fill_padding_call={_uses_zero_pad}\n"
        f"contains_repeat_or_tile_padding={_uses_repeat}\n"
        f"expected: contains_zero_fill_padding_call=False, contains_repeat_or_tile_padding=True\n"
    )

    if _uses_zero_pad or not _uses_repeat:
        sys.exit(
            "[FATAL] Preprocessing source-code check failed: "
            "audio/preprocessing/audio_preprocessor.py does not look like the "
            "tile-repeat implementation (matches a zero-padding pattern instead, "
            "or no repeat/tile call found). This does not match official AASIST "
            "data_utils.pad(). Refusing to run the baseline evaluation with a "
            "mismatched preprocessor. Check that you extracted the correct "
            "project zip and did not overwrite this file."
        )
    print(
        "[Preprocessing self-check OK] audio_preprocessor.py source uses repeat/tile "
        "padding, no zero-fill padding call found."
    )

    # -------------------------------------------------------------------
    # Dataset layout
    # -------------------------------------------------------------------
    la_root = Path(args.dataset_root)
    protocol_dir = la_root / "ASVspoof2019_LA_cm_protocols"
    eval_dir = la_root / "ASVspoof2019_LA_eval"
    eval_flac_dir = eval_dir / "flac"
    protocol_path = protocol_dir / "ASVspoof2019.LA.cm.eval.trl.txt"

    for p, desc in [
        (protocol_dir, "Protocol directory"),
        (protocol_path, "Protocol file"),
        (eval_dir, "Eval directory"),
        (eval_flac_dir, "Eval FLAC directory"),
    ]:
        if not p.exists():
            sys.exit(f"[FATAL] {desc} not found: {p}")

    # -------------------------------------------------------------------
    # Parse protocol
    # -------------------------------------------------------------------
    total_entries, protocol_ids, protocol_labels = read_protocol(protocol_path)
    protocol_id_to_label = {uid: lbl for uid, lbl in zip(protocol_ids, protocol_labels)}

    # -------------------------------------------------------------------
    # Dataset verification – must pass before any inference
    # -------------------------------------------------------------------
    total_files, matched_ids, missing_ids = verify_dataset(eval_flac_dir, protocol_ids)
    # O(N) duplicate detection
    id_counts = Counter(protocol_ids)
    dup_ids = [uid for uid, cnt in id_counts.items() if cnt > 1]

    verification_report = {
        "protocol_entries": total_entries,
        "audio_files_matched": len(matched_ids),
        "missing_or_corrupt": missing_ids[:50],  # cap to keep log readable
        "missing_count": len(missing_ids),
        "duplicate_ids": dup_ids,
    }
    (evidence_dir / "phase5_dataset_verification.txt").write_text(
        json.dumps(verification_report, indent=2)
    )

    if missing_ids:
        sys.exit(
            f"[FATAL] Dataset verification failed - {len(missing_ids)} missing/corrupt files. "
            "See evidence/baseline/phase5_dataset_verification.txt"
        )

    # -------------------------------------------------------------------
    # Load model – CUDA required
    # -------------------------------------------------------------------
    if not torch.cuda.is_available():
        sys.exit(
            "[FATAL] CUDA is not available. Phase 5 evaluation requires a GPU (Tesla T4). "
            "Run this script in Google Colab with a GPU runtime."
        )
    device = torch.device("cuda")

    config_path = _PROJECT_ROOT / "aasist_official" / "config" / "AASIST-L.conf"
    if not config_path.is_file():
        sys.exit(f"[FATAL] AASIST-L config not found: {config_path}")

    model, _ = AASISTLoader.load_model(str(config_path), str(ckpt_path), device=device)
    model.eval()

    # -------------------------------------------------------------------
    # Smoke test – 5 random utterances (seed 42)
    # -------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    smoke_ids = rng.choice(matched_ids, size=5, replace=False).tolist()
    smoke_results = []
    for uid in smoke_ids:
        audio_path = eval_flac_dir / f"{uid}.flac"
        proc = AudioPreprocessor.process_file(str(audio_path))  # shape [1, 64600]
        # AudioPreprocessor returns [1, 64600]. The model's forward() does
        # x = x.unsqueeze(1) internally, so it expects [B, 64600] — NOT [B, 1, 64600].
        # proc is already [1, 64600] == [B=1, 64600]; send it directly.
        tensor = proc.float().to(device)  # shape [1, 64600]
        assert tensor.shape == (1, 64600), f"Unexpected tensor shape: {tensor.shape}"
        with torch.no_grad():
            _, logits = model(tensor)  # last_hidden, logits; discard last_hidden
        assert logits.shape == (1, 2), f"Unexpected logits shape: {logits.shape}"
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            sys.exit(f"[FATAL] Smoke-test NaN/Inf logits for {uid}")
        smoke_results.append(
            {
                "utt_id": uid,
                "label": "bonafide" if protocol_id_to_label[uid] == 1 else "spoof",
                "tensor_shape": list(tensor.shape),
                "logits_shape": list(logits.shape),
                "spoof_logit": logits[0, 0].item(),
                "bonafide_logit": logits[0, 1].item(),
            }
        )

    (evidence_dir / "phase5_smoke_test.txt").write_text(
        json.dumps(
            {"seed": args.seed, "selected_ids": smoke_ids, "results": smoke_results},
            indent=2,
        )
    )
    print("[Smoke test OK] 5 utterances processed without error.")

    # -------------------------------------------------------------------
    # Full evaluation – batch processing
    #
    # All batches (including the final partial batch) are built once from
    # matched_ids and each batch is processed exactly once.
    # -------------------------------------------------------------------
    all_scores: list = []  # each entry: [utt_id, label, spoof_logit, bonafide_logit]
    errors: list = []
    start = time.time()

    batches = [
        matched_ids[i : i + args.batch_size]
        for i in range(0, len(matched_ids), args.batch_size)
    ]

    for batch in tqdm(batches, desc="Evaluating"):
        run_batch(
            batch,
            eval_flac_dir,
            protocol_id_to_label,
            model,
            device,
            all_scores,
            errors,
        )

    total_time = time.time() - start

    if not all_scores:
        sys.exit("[FATAL] No scores collected - all files may have failed. See errors.")

    # -------------------------------------------------------------------
    # Save raw scores CSV
    # -------------------------------------------------------------------
    scores_path = results_dir / "asvspoof_baseline_scores.csv"
    with open(scores_path, "w", encoding="utf-8") as fh:
        fh.write("utt_id,label,spoof_logit,bonafide_logit\n")
        for uid, lbl, sp_logit, bf_logit in all_scores:
            fh.write(f"{uid},{lbl},{sp_logit:.6f},{bf_logit:.6f}\n")

    # -------------------------------------------------------------------
    # EER calculation
    #
    # Score direction:
    #   Official AASIST methodology uses a single scalar score per utterance:
    #   score = logits[:, 1] (bonafide confidence)
    #
    #   We split this single score array by the protocol label:
    #   target_scores     = score where label == 1 (bonafide)
    #   nontarget_scores  = score where label == 0 (spoof)
    #
    # compute_eer(target_scores, nontarget_scores) returns (eer_fraction, threshold).
    # Multiply fraction by 100 to get EER percent.
    # -------------------------------------------------------------------
    # row[3] is bonafide_logit, row[1] is label
    bonafide_scores = np.array(
        [row[3] for row in all_scores if row[1] == 1], dtype=np.float64
    )
    spoof_scores = np.array(
        [row[3] for row in all_scores if row[1] == 0], dtype=np.float64
    )

    eer_frac, eer_threshold = compute_eer(bonafide_scores, spoof_scores)
    eer_percent = float(eer_frac * 100.0)

    # Cross-check
    eer_cross_percent = cross_check_eer(bonafide_scores, spoof_scores)

    # -------------------------------------------------------------------
    # Save metrics
    # -------------------------------------------------------------------
    metrics = {
        "eer_percent": eer_percent,
        "eer_threshold": float(eer_threshold),
        "eer_cross_check_percent": eer_cross_percent,
        "score_direction": "single score = logits[:,1]; target=bonafide(1), nontarget=spoof(0)",
        "total_samples": len(all_scores),
        "failed_samples": len(errors),
        "batch_size": args.batch_size,
        "total_inference_time_sec": total_time,
        "throughput_sec_per_sample": total_time / max(1, len(all_scores)),
    }
    (results_dir / "asvspoof_baseline_metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )

    # -------------------------------------------------------------------
    # Write evidence/baseline/phase5_eer.txt (required output)
    # -------------------------------------------------------------------
    eer_evidence = (
        f"EER (official compute_eer)  : {eer_percent:.4f}%\n"
        f"EER threshold               : {eer_threshold:.6f}\n"
        f"EER cross-check (sklearn)   : {eer_cross_percent:.4f}%\n"
        f"Score direction             : Official single scalar score (logits[:, 1])\n"
        f"                              target_scores    = score[label == 1]\n"
        f"                              nontarget_scores = score[label == 0]\n"
        f"Total scored samples        : {len(all_scores)}\n"
        f"Failed samples              : {len(errors)}\n"
    )
    (evidence_dir / "phase5_eer.txt").write_text(eer_evidence)

    # -------------------------------------------------------------------
    # Record errors
    # -------------------------------------------------------------------
    if errors:
        err_path = results_dir / "asvspoof_baseline_errors.csv"
        with open(err_path, "w", encoding="utf-8") as fh:
            fh.write("utt_id,error\n")
            for e in errors:
                fh.write(f"{e['utt_id']},{e['error'].replace(',', ';')}\n")
        (evidence_dir / "phase5_evaluation.log").write_text(
            f"Evaluation completed with {len(errors)} file errors. See asvspoof_baseline_errors.csv\n"
        )
    else:
        (evidence_dir / "phase5_evaluation.log").write_text(
            "Evaluation completed without errors.\n"
        )

    # -------------------------------------------------------------------
    # Checkpoint integrity (post-evaluation)
    # -------------------------------------------------------------------
    post_hash = sha256_file(ckpt_path)
    with open(hash_log, "a", encoding="utf-8") as fh:
        fh.write(f"post_hash={post_hash}\n")
    if pre_hash != post_hash:
        sys.exit("[FATAL] Checkpoint SHA-256 changed during evaluation!")

    # -------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------
    print("=" * 60)
    print("Phase 5 Baseline Evaluation Complete")
    print("=" * 60)
    print(f"  Dataset root              : {la_root}")
    print(f"  Protocol entries          : {total_entries}")
    print(f"  Audio files matched       : {len(matched_ids)}")
    print(f"  Missing / corrupt files   : {len(missing_ids)}")
    print(f"  Duplicate IDs             : {len(dup_ids)}")
    print(f"  Smoke-test IDs (seed={args.seed}): {smoke_ids}")
    print(f"  Batch size                : {args.batch_size}")
    print(f"  Total inference time (s)  : {total_time:.2f}")
    print(
        f"  EER                       : {eer_percent:.3f}%  (threshold={eer_threshold:.4f})"
    )
    print(f"  EER cross-check           : {eer_cross_percent:.3f}%")
    print(f"  Results saved to          : {results_dir}")
    print(f"  Evidence saved to         : {evidence_dir}")
    if errors:
        print(f"  WARNING: {len(errors)} files failed - experiment marked incomplete.")
    else:
        print("  All files processed successfully.")


if __name__ == "__main__":
    main()
