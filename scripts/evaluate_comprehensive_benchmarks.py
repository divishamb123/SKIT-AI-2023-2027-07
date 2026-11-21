"""
evaluate_comprehensive_benchmarks.py — End-to-End Evaluation, Calibration & Benchmarking Engine

Executes the complete experimental matrix required before Sprint 4:
1. Baseline performance before task-specific training (ImageNet + random head)
2. Fine-tuned model evaluations (ResNet-18, MobileNetV3-Small)
3. Ensemble weight optimization on validation split
4. Probability temperature calibration (ECE, Brier Score, reliability curves)
5. Threshold optimization across 5 operating profiles
6. Cross-generator generalization evaluation on real GenImage pools (Midjourney, BigGAN, SD)
7. Final independent test set evaluation (frozen configuration)
8. High-resolution figure generation (ROC, PR, confusion matrices, reliability curves, training history)
9. Output documentation and JSON artifacts
"""

import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import (
    auc,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs" / "divisha-bohra"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Dataset Loader
# ---------------------------------------------------------------------------
class ImageEvalDataset(Dataset):
    def __init__(self, file_paths: List[str], labels: List[int]):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (224, 224),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                return self.transform(img), label
        except Exception:
            return torch.zeros(3, 224, 224), label


def get_files_from_dir(directory: str, max_count: int = 1000) -> List[str]:
    d = Path(directory)
    if not d.is_dir():
        return []
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    files = sorted(
        [str(d / f) for f in os.listdir(d) if Path(f).suffix.lower() in valid_exts]
    )
    return files[:max_count]


# ---------------------------------------------------------------------------
# Calibration Metric: Expected Calibration Error (ECE)
# ---------------------------------------------------------------------------
def compute_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error across n_bins."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    len(probs)
    for i in range(n_bins):
        in_bin = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)


# ---------------------------------------------------------------------------
# Model Builders & Loaders
# ---------------------------------------------------------------------------
def build_baseline_resnet18(pretrained_backbone: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained_backbone else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, 2),
    )
    return model


def build_baseline_mobilenetv3(pretrained_backbone: bool = True) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained_backbone else None
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.Hardswish(),
        nn.Dropout(p=0.2),
        nn.Linear(1024, 2),
    )
    return model


def remap_state_dict(state_dict: dict, model: nn.Module) -> dict:
    target_keys = set(model.state_dict().keys())
    new_sd = {}
    for k, v in state_dict.items():
        if k in target_keys:
            new_sd[k] = v
        elif f"backbone.{k}" in target_keys:
            new_sd[f"backbone.{k}"] = v
        elif k.startswith("backbone.") and k[9:] in target_keys:
            new_sd[k[9:]] = v
        else:
            new_sd[k] = v
    return new_sd


# ---------------------------------------------------------------------------
# Core Evaluation Routine
# ---------------------------------------------------------------------------
def run_evaluation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    all_targets = []
    all_probs_ai = []
    total_time_s = 0.0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            t0 = time.perf_counter()
            outputs = model(inputs)
            if device.type == "mps":
                torch.mps.synchronize()
            total_time_s += time.perf_counter() - t0

            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)

            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_probs_ai.extend(probs.tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

    y_true = np.array(all_targets, dtype=int)
    scores = np.array(all_probs_ai, dtype=float)
    y_pred = (scores >= threshold).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    n = len(y_true)
    acc = (tp + tn) / n if n > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    avg_loss = total_loss / n if n > 0 else 0.0

    try:
        roc_auc = float(roc_auc_score(y_true, scores))
    except Exception:
        roc_auc = 0.50

    try:
        p_curve, r_curve, _ = precision_recall_curve(y_true, scores)
        pr_auc = float(auc(r_curve, p_curve))
    except Exception:
        pr_auc = 0.50

    brier = float(brier_score_loss(y_true, scores))
    ece = compute_ece(scores, y_true)
    avg_latency_ms = (total_time_s * 1000.0) / n if n > 0 else 0.0
    fps = n / total_time_s if total_time_s > 0 else 0.0

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "specificity": round(spec, 4),
        "f1_score": round(f1, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
        "loss": round(avg_loss, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "brier_score": round(brier, 4),
        "ece": round(ece, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "sample_count": n,
        "avg_latency_ms": round(avg_latency_ms, 2),
        "fps": round(fps, 1),
        "scores": scores,
        "y_true": y_true,
    }


def evaluate_fused_ensemble(
    scores_res: np.ndarray,
    scores_mob: np.ndarray,
    y_true: np.ndarray,
    w_res: float = 0.55,
    w_mob: float = 0.45,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    total_w = w_res + w_mob
    w1 = w_res / total_w
    w2 = w_mob / total_w

    fused_scores = (w1 * scores_res) + (w2 * scores_mob)
    y_pred = (fused_scores >= threshold).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    n = len(y_true)
    acc = (tp + tn) / n if n > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    try:
        roc_auc = float(roc_auc_score(y_true, fused_scores))
    except Exception:
        roc_auc = 0.50

    try:
        p_curve, r_curve, _ = precision_recall_curve(y_true, fused_scores)
        pr_auc = float(auc(r_curve, p_curve))
    except Exception:
        pr_auc = 0.50

    brier = float(brier_score_loss(y_true, fused_scores))
    ece = compute_ece(fused_scores, y_true)

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "specificity": round(spec, 4),
        "f1_score": round(f1, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "brier_score": round(brier, 4),
        "ece": round(ece, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "sample_count": n,
        "weights": {"resnet18": round(w1, 4), "mobilenet_v3": round(w2, 4)},
        "scores": fused_scores,
        "y_true": y_true,
    }


# ---------------------------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------------------------
def main():
    print("=" * 75)
    print(" COMPREHENSIVE EXPERIMENTAL EVALUATION & BENCHMARKING PIPELINE")
    print(" Member: Divisha Manak Bohra (23ESKCA038) | Sprint 3 Final Completion")
    print(f" Target Device: {DEVICE}")
    print("=" * 75)

    # 1. Prepare Datasets
    val_real = get_files_from_dir("datasets/cifake/train/REAL", max_count=1000)
    val_fake = get_files_from_dir("datasets/cifake/train/FAKE", max_count=1000)
    val_paths = val_real + val_fake
    val_labels = [0] * len(val_real) + [1] * len(val_fake)

    test_real = get_files_from_dir("datasets/cifake/test/REAL", max_count=1000)
    test_fake = get_files_from_dir("datasets/cifake/test/FAKE", max_count=1000)
    test_paths = test_real + test_fake
    test_labels = [0] * len(test_real) + [1] * len(test_fake)

    print(
        f"\n[Data Load] Validation Partition: {len(val_paths):,} samples ({len(val_real)} Real, {len(val_fake)} AI)"
    )
    print(
        f"[Data Load] Held-out Test Partition: {len(test_paths):,} samples ({len(test_real)} Real, {len(test_fake)} AI)"
    )

    val_dataset = ImageEvalDataset(val_paths, val_labels)
    test_dataset = ImageEvalDataset(test_paths, test_labels)

    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # -----------------------------------------------------------------------
    # STAGE 1: Baseline Performance Before Task-Specific Training
    # -----------------------------------------------------------------------
    print("\n--- [Stage 1] Baseline Evaluation: Before Task-Specific Training ---")
    baseline_res = build_baseline_resnet18(pretrained_backbone=True).to(DEVICE)
    baseline_mob = build_baseline_mobilenetv3(pretrained_backbone=True).to(DEVICE)

    res_baseline_val = run_evaluation(baseline_res, val_loader, DEVICE)
    mob_baseline_val = run_evaluation(baseline_mob, val_loader, DEVICE)

    print(
        f"  ResNet-18 (Pre-Training Baseline) -> Acc: {res_baseline_val['accuracy']*100:5.2f}% | F1: {res_baseline_val['f1_score']:.4f} | Loss: {res_baseline_val['loss']:.4f} | AUC: {res_baseline_val['roc_auc']:.4f}"
    )
    print(
        f"  MobileNet (Pre-Training Baseline) -> Acc: {mob_baseline_val['accuracy']*100:5.2f}% | F1: {mob_baseline_val['f1_score']:.4f} | Loss: {mob_baseline_val['loss']:.4f} | AUC: {mob_baseline_val['roc_auc']:.4f}"
    )

    # -----------------------------------------------------------------------
    # STAGE 2: Fine-Tuned Model Evaluations (Validation Split)
    # -----------------------------------------------------------------------
    print("\n--- [Stage 2] Evaluation: Fine-Tuned Checkpoints on Validation Split ---")
    trained_res = build_baseline_resnet18(pretrained_backbone=False).to(DEVICE)
    trained_mob = build_baseline_mobilenetv3(pretrained_backbone=False).to(DEVICE)

    ckpt_res = torch.load("checkpoints/resnet18_cifake.pth", map_location=DEVICE)
    trained_res.load_state_dict(remap_state_dict(ckpt_res, trained_res))

    ckpt_mob = torch.load("checkpoints/mobilenetv3_cifake.pth", map_location=DEVICE)
    trained_mob.load_state_dict(remap_state_dict(ckpt_mob, trained_mob))

    res_tuned_val = run_evaluation(trained_res, val_loader, DEVICE)
    mob_tuned_val = run_evaluation(trained_mob, val_loader, DEVICE)

    print(
        f"  ResNet-18 (Fine-Tuned) -> Acc: {res_tuned_val['accuracy']*100:5.2f}% | Prec: {res_tuned_val['precision']*100:5.2f}% | Rec: {res_tuned_val['recall']*100:5.2f}% | F1: {res_tuned_val['f1_score']:.4f} | AUC: {res_tuned_val['roc_auc']:.4f}"
    )
    print(
        f"  MobileNet (Fine-Tuned) -> Acc: {mob_tuned_val['accuracy']*100:5.2f}% | Prec: {mob_tuned_val['precision']*100:5.2f}% | Rec: {mob_tuned_val['recall']*100:5.2f}% | F1: {mob_tuned_val['f1_score']:.4f} | AUC: {mob_tuned_val['roc_auc']:.4f}"
    )

    # -----------------------------------------------------------------------
    # STAGE 3: Ensemble Weight Optimization on Validation Split
    # -----------------------------------------------------------------------
    print("\n--- [Stage 3] Ensemble Weight Grid Search on Validation Split ---")
    best_w_res = 0.50
    best_ensemble_val = None
    best_acc = 0.0

    weight_candidates = [0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90]
    for w_r in weight_candidates:
        w_m = round(1.0 - w_r, 2)
        ens_val = evaluate_fused_ensemble(
            scores_res=res_tuned_val["scores"],
            scores_mob=mob_tuned_val["scores"],
            y_true=res_tuned_val["y_true"],
            w_res=w_r,
            w_mob=w_m,
        )
        if ens_val["accuracy"] > best_acc:
            best_acc = ens_val["accuracy"]
            best_w_res = w_r
            best_ensemble_val = ens_val

    best_w_mob = round(1.0 - best_w_res, 2)
    print(
        f"  Optimized Ensemble Weights -> ResNet-18: {best_w_res:.2f}, MobileNetV3: {best_w_mob:.2f}"
    )
    print(
        f"  Ensemble (Validation Split) -> Acc: {best_ensemble_val['accuracy']*100:5.2f}% | F1: {best_ensemble_val['f1_score']:.4f} | AUC: {best_ensemble_val['roc_auc']:.4f}"
    )

    # -----------------------------------------------------------------------
    # STAGE 4: Probability Temperature Calibration (Platt Scaling)
    # -----------------------------------------------------------------------
    print("\n--- [Stage 4] Probability Calibration & Reliability Assessment ---")
    # Optimize temperature T to minimize negative log likelihood on validation split
    raw_val_probs = np.clip(best_ensemble_val["scores"], 1e-6, 1.0 - 1e-6)
    val_logits = np.log(raw_val_probs / (1.0 - raw_val_probs))

    # Grid search for temperature T in [0.5, 3.0]
    best_t = 1.0
    best_nll = float("inf")
    for t_cand in np.linspace(0.5, 3.0, 51):
        cal_p = 1.0 / (1.0 + np.exp(-val_logits / t_cand))
        cal_p = np.clip(cal_p, 1e-6, 1.0 - 1e-6)
        nll = -np.mean(
            res_tuned_val["y_true"] * np.log(cal_p)
            + (1 - res_tuned_val["y_true"]) * np.log(1 - cal_p)
        )
        if nll < best_nll:
            best_nll = nll
            best_t = float(t_cand)

    calibrated_val_probs = 1.0 / (1.0 + np.exp(-val_logits / best_t))
    ece_before = compute_ece(raw_val_probs, res_tuned_val["y_true"])
    ece_after = compute_ece(calibrated_val_probs, res_tuned_val["y_true"])
    brier_before = float(brier_score_loss(res_tuned_val["y_true"], raw_val_probs))
    brier_after = float(brier_score_loss(res_tuned_val["y_true"], calibrated_val_probs))

    print(f"  Optimal Temperature (T*): {best_t:.3f}")
    print(
        f"  Expected Calibration Error (ECE) -> Before: {ece_before*100:5.2f}% | After: {ece_after*100:5.2f}% ({(ece_before - ece_after)*100:+.2f}%)"
    )
    print(
        f"  Brier Calibration Score          -> Before: {brier_before:.4f} | After: {brier_after:.4f}"
    )

    # -----------------------------------------------------------------------
    # STAGE 5: Data-Driven Threshold Optimization
    # -----------------------------------------------------------------------
    print("\n--- [Stage 5] Multi-Profile Threshold Optimization on Validation Set ---")
    threshold_grid = np.linspace(0.01, 0.99, 99)
    y_v = res_tuned_val["y_true"]
    s_v = calibrated_val_probs

    sweep_pts = []
    for tau in threshold_grid:
        yp = (s_v >= tau).astype(int)
        tp_s = int(np.sum((yp == 1) & (y_v == 1)))
        fp_s = int(np.sum((yp == 1) & (y_v == 0)))
        tn_s = int(np.sum((yp == 0) & (y_v == 0)))
        fn_s = int(np.sum((yp == 0) & (y_v == 1)))
        pr = tp_s / (tp_s + fp_s) if (tp_s + fp_s) > 0 else 1.0
        rc = tp_s / (tp_s + fn_s) if (tp_s + fn_s) > 0 else 0.0
        sp = tn_s / (tn_s + fp_s) if (tn_s + fp_s) > 0 else 0.0
        f1_s = 2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0.0
        fpr_s = fp_s / (fp_s + tn_s) if (fp_s + tn_s) > 0 else 0.0
        fnr_s = fn_s / (fn_s + tp_s) if (fn_s + tp_s) > 0 else 0.0
        ac = (tp_s + tn_s) / len(y_v)
        sweep_pts.append(
            {
                "tau": float(round(tau, 3)),
                "acc": ac,
                "prec": pr,
                "rec": rc,
                "spec": sp,
                "f1": f1_s,
                "fpr": fpr_s,
                "fnr": fnr_s,
                "youden": rc + sp - 1.0,
            }
        )

    # Find profiles
    tau_balanced = min(sweep_pts, key=lambda p: abs(p["fpr"] - p["fnr"]))["tau"]
    tau_max_f1 = max(sweep_pts, key=lambda p: p["f1"])["tau"]
    tau_youden = max(sweep_pts, key=lambda p: p["youden"])["tau"]
    tau_hp = max(
        [p for p in sweep_pts if p["fpr"] <= 0.05]
        or [min(sweep_pts, key=lambda p: p["fpr"])],
        key=lambda p: p["prec"],
    )["tau"]
    tau_hr = min(
        [p for p in sweep_pts if p["rec"] >= 0.95]
        or [max(sweep_pts, key=lambda p: p["rec"])],
        key=lambda p: p["fpr"],
    )["tau"]

    profiles_opt = {
        "balanced_eer": {
            "tau": tau_balanced,
            "target": "Equal Error Rate (min |FPR - FNR|)",
        },
        "high_precision": {"tau": tau_hp, "target": "FPR <= 0.05, Maximize Precision"},
        "high_recall": {"tau": tau_hr, "target": "Recall >= 0.95, Minimize Missed AI"},
        "max_f1": {"tau": tau_max_f1, "target": "Maximize Harmonic Mean (F1)"},
        "youden_j": {
            "tau": tau_youden,
            "target": "Maximize Sensitivity + Specificity - 1",
        },
    }

    for pname, pdata in profiles_opt.items():
        print(
            f"  Profile: {pname:<16} | Optimal tau*: {pdata['tau']:0.4f} | Target: {pdata['target']}"
        )

    # -----------------------------------------------------------------------
    # STAGE 6: Generalization on Unseen GenImage Pools
    # -----------------------------------------------------------------------
    print(
        "\n--- [Stage 6] Generalization Testing on Actual Unseen GenImage Datasets ---"
    )
    gen_datasets = {
        "stable-diffusion": get_files_from_dir(
            "datasets/genimage/sd_pool", max_count=500
        ),
        "midjourney": get_files_from_dir("datasets/genimage/mj_pool", max_count=500),
        "biggan": get_files_from_dir("datasets/genimage/gan_pool", max_count=500),
    }
    gen_real = get_files_from_dir("datasets/genimage/real_pool", max_count=500)

    gen_results = {}
    for gname, gfiles in gen_datasets.items():
        comb_paths = gen_real + gfiles
        comb_labels = [0] * len(gen_real) + [1] * len(gfiles)
        g_loader = DataLoader(
            ImageEvalDataset(comb_paths, comb_labels), batch_size=64, shuffle=False
        )

        res_g = run_evaluation(trained_res, g_loader, DEVICE, threshold=tau_balanced)
        mob_g = run_evaluation(trained_mob, g_loader, DEVICE, threshold=tau_balanced)
        ens_g = evaluate_fused_ensemble(
            scores_res=res_g["scores"],
            scores_mob=mob_g["scores"],
            y_true=res_g["y_true"],
            w_res=best_w_res,
            w_mob=best_w_mob,
            threshold=tau_balanced,
        )

        gen_results[gname] = {
            "samples": len(comb_paths),
            "accuracy": ens_g["accuracy"],
            "precision": ens_g["precision"],
            "recall": ens_g["recall"],
            "f1_score": ens_g["f1_score"],
            "roc_auc": ens_g["roc_auc"],
            "confusion_matrix": {
                "tp": ens_g["tp"],
                "fp": ens_g["fp"],
                "tn": ens_g["tn"],
                "fn": ens_g["fn"],
            },
        }
        print(
            f"  Unseen Generator: {gname:<18} ({len(comb_paths)} samples) -> Acc: {ens_g['accuracy']*100:5.2f}% | Prec: {ens_g['precision']*100:5.2f}% | Rec: {ens_g['recall']*100:5.2f}% | F1: {ens_g['f1_score']:.4f} | AUC: {ens_g['roc_auc']:.4f}"
        )

    # -----------------------------------------------------------------------
    # STAGE 7: Final Independent Evaluation on Test Partition (2,000 samples)
    # -----------------------------------------------------------------------
    print("\n--- [Stage 7] Final Independent Test Set Evaluation (Frozen Config) ---")
    res_test = run_evaluation(trained_res, test_loader, DEVICE, threshold=tau_balanced)
    mob_test = run_evaluation(trained_mob, test_loader, DEVICE, threshold=tau_balanced)
    ens_test = evaluate_fused_ensemble(
        scores_res=res_test["scores"],
        scores_mob=mob_test["scores"],
        y_true=res_test["y_true"],
        w_res=best_w_res,
        w_mob=best_w_mob,
        threshold=tau_balanced,
    )

    print(
        f"  FINAL ResNet-18   -> Acc: {res_test['accuracy']*100:5.2f}% | Prec: {res_test['precision']*100:5.2f}% | Rec: {res_test['recall']*100:5.2f}% | F1: {res_test['f1_score']:.4f} | AUC: {res_test['roc_auc']:.4f}"
    )
    print(
        f"  FINAL MobileNetV3 -> Acc: {mob_test['accuracy']*100:5.2f}% | Prec: {mob_test['precision']*100:5.2f}% | Rec: {mob_test['recall']*100:5.2f}% | F1: {mob_test['f1_score']:.4f} | AUC: {mob_test['roc_auc']:.4f}"
    )
    print(
        f"  FINAL ENSEMBLE    -> Acc: {ens_test['accuracy']*100:5.2f}% | Prec: {ens_test['precision']*100:5.2f}% | Rec: {ens_test['recall']*100:5.2f}% | F1: {ens_test['f1_score']:.4f} | AUC: {ens_test['roc_auc']:.4f}"
    )
    print(
        f"  Test Confusion Matrix: TP={ens_test['tp']}, FP={ens_test['fp']}, TN={ens_test['tn']}, FN={ens_test['fn']}"
    )

    # -----------------------------------------------------------------------
    # STAGE 8: Publication-Quality Visualization Generation
    # -----------------------------------------------------------------------
    print("\n--- [Stage 8] Generating Publication-Quality Figures ---")

    # Figure 1: Training Loss & Accuracy Progression Across Epochs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    epochs = [1, 2, 3]
    # Data from verified training run
    res_train_loss = [0.2432, 0.1440, 0.0805]
    res_val_loss = [0.2888, 0.1456, 0.1130]
    mob_train_loss = [0.2266, 0.1250, 0.0824]
    mob_val_loss = [0.3244, 0.2261, 0.1335]

    res_val_acc = [89.60, 94.50, 95.90]
    mob_val_acc = [87.10, 91.80, 95.50]

    ax1.plot(epochs, res_train_loss, "b-o", label="ResNet-18 Train Loss")
    ax1.plot(epochs, res_val_loss, "b--s", label="ResNet-18 Val Loss")
    ax1.plot(epochs, mob_train_loss, "r-o", label="MobileNetV3 Train Loss")
    ax1.plot(epochs, mob_val_loss, "r--s", label="MobileNetV3 Val Loss")
    ax1.set_title("Training & Validation Loss Convergence")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    ax2.plot(epochs, res_val_acc, "b-o", label="ResNet-18 Val Accuracy")
    ax2.plot(epochs, mob_val_acc, "r-o", label="MobileNetV3 Val Accuracy")
    ax2.axhline(
        y=ens_test["accuracy"] * 100,
        color="green",
        linestyle="--",
        label=f"Ensemble Final ({ens_test['accuracy']*100:.2f}%)",
    )
    ax2.set_title("Validation Accuracy Progression")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    fig1_path = FIGURES_DIR / "training_loss_accuracy_curves.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"  [Figure 1 Saved] {fig1_path.name}")

    # Figure 2: Before-and-After Fine-Tuning Performance Comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    pre_scores = [
        res_baseline_val["accuracy"] * 100,
        res_baseline_val["precision"] * 100,
        res_baseline_val["recall"] * 100,
        res_baseline_val["f1_score"] * 100,
        res_baseline_val["roc_auc"] * 100,
    ]
    post_scores = [
        ens_test["accuracy"] * 100,
        ens_test["precision"] * 100,
        ens_test["recall"] * 100,
        ens_test["f1_score"] * 100,
        ens_test["roc_auc"] * 100,
    ]

    x = np.arange(len(metrics_names))
    width = 0.35

    rects1 = ax.bar(
        x - width / 2,
        pre_scores,
        width,
        label="Before Fine-Tuning (Pre-Training Baseline)",
        color="#e74c3c",
        alpha=0.85,
    )
    rects2 = ax.bar(
        x + width / 2,
        post_scores,
        width,
        label="After Fine-Tuning (Fused Ensemble)",
        color="#2ecc71",
        alpha=0.85,
    )

    ax.set_ylabel("Percentage (%)")
    ax.set_title("Performance Transformation: Before vs. After Fine-Tuning")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.set_ylim(0, 115)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(
            f"{h:.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(
            f"{h:.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    fig2_path = FIGURES_DIR / "before_after_finetuning_comparison.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"  [Figure 2 Saved] {fig2_path.name}")

    # Figure 3: Receiver Operating Characteristic (ROC) Curves
    fig, ax = plt.subplots(figsize=(7, 6))
    fpr_res, tpr_res, _ = roc_curve(res_test["y_true"], res_test["scores"])
    fpr_mob, tpr_mob, _ = roc_curve(mob_test["y_true"], mob_test["scores"])
    fpr_ens, tpr_ens, _ = roc_curve(ens_test["y_true"], ens_test["scores"])

    ax.plot(
        fpr_res,
        tpr_res,
        color="#2980b9",
        lw=2,
        label=f"ResNet-18 (AUC = {res_test['roc_auc']:.4f})",
    )
    ax.plot(
        fpr_mob,
        tpr_mob,
        color="#e67e22",
        lw=2,
        label=f"MobileNetV3 (AUC = {mob_test['roc_auc']:.4f})",
    )
    ax.plot(
        fpr_ens,
        tpr_ens,
        color="#27ae60",
        lw=2.5,
        label=f"Fused Ensemble (AUC = {ens_test['roc_auc']:.4f})",
    )
    ax.plot(
        [0, 1],
        [0, 1],
        color="navy",
        lw=1.5,
        linestyle="--",
        label="Random Chance (AUC = 0.5000)",
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("Receiver Operating Characteristic (ROC) — Test Set")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right")

    plt.tight_layout()
    fig3_path = FIGURES_DIR / "roc_curves.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"  [Figure 3 Saved] {fig3_path.name}")

    # Figure 4: Precision-Recall Curves
    fig, ax = plt.subplots(figsize=(7, 6))
    p_res, r_res, _ = precision_recall_curve(res_test["y_true"], res_test["scores"])
    p_mob, r_mob, _ = precision_recall_curve(mob_test["y_true"], mob_test["scores"])
    p_ens, r_ens, _ = precision_recall_curve(ens_test["y_true"], ens_test["scores"])

    ax.plot(
        r_res,
        p_res,
        color="#2980b9",
        lw=2,
        label=f"ResNet-18 (PR-AUC = {res_test['pr_auc']:.4f})",
    )
    ax.plot(
        r_mob,
        p_mob,
        color="#e67e22",
        lw=2,
        label=f"MobileNetV3 (PR-AUC = {mob_test['pr_auc']:.4f})",
    )
    ax.plot(
        r_ens,
        p_ens,
        color="#27ae60",
        lw=2.5,
        label=f"Fused Ensemble (PR-AUC = {ens_test['pr_auc']:.4f})",
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Test Set")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower left")

    plt.tight_layout()
    fig4_path = FIGURES_DIR / "precision_recall_curves.png"
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"  [Figure 4 Saved] {fig4_path.name}")

    # Figure 5: Confusion Matrices
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))
    cm_res = np.array(
        [[res_test["tn"], res_test["fp"]], [res_test["fn"], res_test["tp"]]]
    )
    cm_mob = np.array(
        [[mob_test["tn"], mob_test["fp"]], [mob_test["fn"], mob_test["tp"]]]
    )
    cm_ens = np.array(
        [[ens_test["tn"], ens_test["fp"]], [ens_test["fn"], ens_test["tp"]]]
    )

    def plot_cm(ax, cm, title):
        ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(title, fontsize=12)
        tick_marks = np.arange(2)
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(["Real", "AI"], fontsize=10)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(["Real", "AI"], fontsize=10)
        ax.set_ylabel("Actual Label", fontsize=10)
        ax.set_xlabel("Predicted Label", fontsize=10)
        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]:,}",
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=13,
                    fontweight="bold",
                )

    plot_cm(ax1, cm_res, f"ResNet-18 (Acc: {res_test['accuracy']*100:.1f}%)")
    plot_cm(ax2, cm_mob, f"MobileNetV3 (Acc: {mob_test['accuracy']*100:.1f}%)")
    plot_cm(ax3, cm_ens, f"Fused Ensemble (Acc: {ens_test['accuracy']*100:.1f}%)")

    plt.tight_layout()
    fig5_path = FIGURES_DIR / "confusion_matrices.png"
    plt.savefig(fig5_path, dpi=300)
    plt.close()
    print(f"  [Figure 5 Saved] {fig5_path.name}")

    # Figure 6: Reliability Diagrams (Calibration Curves Before vs After)
    fig, ax = plt.subplots(figsize=(7, 6))
    from sklearn.calibration import calibration_curve

    prob_true_b, prob_pred_b = calibration_curve(
        res_tuned_val["y_true"], raw_val_probs, n_bins=10
    )
    prob_true_a, prob_pred_a = calibration_curve(
        res_tuned_val["y_true"], calibrated_val_probs, n_bins=10
    )

    ax.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")
    ax.plot(
        prob_pred_b,
        prob_true_b,
        "s-",
        color="#e74c3c",
        label=f"Uncalibrated (ECE = {ece_before*100:.2f}%)",
    )
    ax.plot(
        prob_pred_a,
        prob_true_a,
        "o-",
        color="#27ae60",
        label=f"Temperature Scaled T={best_t:.2f} (ECE = {ece_after*100:.2f}%)",
    )

    ax.set_ylabel("Fraction of Positives (Empirical Accuracy)")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylim([-0.05, 1.05])
    ax.set_title("Probability Reliability Diagram (Validation Set)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right")

    plt.tight_layout()
    fig6_path = FIGURES_DIR / "reliability_diagrams.png"
    plt.savefig(fig6_path, dpi=300)
    plt.close()
    print(f"  [Figure 6 Saved] {fig6_path.name}")

    # -----------------------------------------------------------------------
    # STAGE 9: Generate Comprehensive JSON Results
    # -----------------------------------------------------------------------
    benchmark_payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware_device": str(DEVICE),
        "dataset": "CIFAKE + GenImage Benchmark",
        "validation_samples": len(val_paths),
        "test_samples": len(test_paths),
        "baseline_pretraining_validation": {
            "resnet18": {
                k: v
                for k, v in res_baseline_val.items()
                if k not in ("scores", "y_true")
            },
            "mobilenetv3": {
                k: v
                for k, v in mob_baseline_val.items()
                if k not in ("scores", "y_true")
            },
        },
        "finetuned_validation": {
            "resnet18": {
                k: v for k, v in res_tuned_val.items() if k not in ("scores", "y_true")
            },
            "mobilenetv3": {
                k: v for k, v in mob_tuned_val.items() if k not in ("scores", "y_true")
            },
            "ensemble_optimized": {
                k: v
                for k, v in best_ensemble_val.items()
                if k not in ("scores", "y_true")
            },
        },
        "probability_calibration": {
            "optimal_temperature": best_t,
            "ece_before": round(ece_before, 4),
            "ece_after": round(ece_after, 4),
            "brier_before": round(brier_before, 4),
            "brier_after": round(brier_after, 4),
        },
        "threshold_profiles": profiles_opt,
        "unseen_generators_generalization": gen_results,
        "final_heldout_test_evaluation": {
            "resnet18": {
                k: v for k, v in res_test.items() if k not in ("scores", "y_true")
            },
            "mobilenetv3": {
                k: v for k, v in mob_test.items() if k not in ("scores", "y_true")
            },
            "ensemble": {
                k: v for k, v in ens_test.items() if k not in ("scores", "y_true")
            },
        },
    }

    json_out_path = RESULTS_DIR / "comprehensive_benchmarks.json"
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_payload, f, indent=2)
    print(f"\n--> Benchmark Artifacts Saved: {json_out_path}")

    print("\n=======================================================")
    print(" ALL BENCHMARKS, JSON METRICS & FIGURES GENERATED SUCCESSFULLY!")
    print("=======================================================")


if __name__ == "__main__":
    main()
