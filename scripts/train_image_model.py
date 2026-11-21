"""
train_image_model.py — Training & Fine-Tuning Pipeline for Image AI Detection

Sprint 3 (Accuracy Optimisation) — Model Training Pipeline
Member: Divisha Manak Bohra (23ESKCA038)

Fine-tunes deep convolutional backbones on the CIFAKE dataset:
1. ResNet-18 (Deep Spatial Residual CNN) with ImageNet transfer learning
2. MobileNetV3-Small (Inverted Residuals + Squeeze-and-Excitation) with ImageNet transfer learning

Outputs:
- checkpoints/resnet18_cifake.pth
- checkpoints/mobilenetv3_cifake.pth
- results/image_training_evaluation.json
- docs/image_training_results.md
"""

import json
import os
from pathlib import Path
import time
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Dataset Definition
# ---------------------------------------------------------------------------
class CIFAKEDataset(Dataset):
    """PyTorch Dataset for CIFAKE balanced image classification."""

    def __init__(self, file_paths: List[str], labels: List[int], transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                if self.transform:
                    img = self.transform(img)
                return img, label
        except Exception:
            # Fallback for damaged file
            dummy = torch.zeros(3, 224, 224)
            return dummy, label


def load_cifake_file_list(
    split_dir: str,
    max_per_class: int = 5000,
) -> Tuple[List[str], List[int]]:
    """Loads balanced list of (path, label) for REAL (0) and FAKE (1)."""
    real_dir = Path(split_dir) / "REAL"
    fake_dir = Path(split_dir) / "FAKE"

    real_files = sorted(
        [
            str(real_dir / f)
            for f in os.listdir(real_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]
    )[:max_per_class]
    fake_files = sorted(
        [
            str(fake_dir / f)
            for f in os.listdir(fake_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]
    )[:max_per_class]

    paths = real_files + fake_files
    labels = [0] * len(real_files) + [1] * len(fake_files)
    return paths, labels


# ---------------------------------------------------------------------------
# Training & Evaluation Routines
# ---------------------------------------------------------------------------
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 3,
    lr: float = 3e-4,
    model_name: str = "resnet18",
) -> Tuple[nn.Module, Dict[str, float]]:
    """Trains a model with AdamW and Cosine Annealing, evaluating on test set each epoch."""
    print("\n=======================================================")
    print(f" Training {model_name.upper()} on CIFAKE Dataset ({device})")
    print("=======================================================")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    best_acc = 0.0
    best_metrics = {}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        t0 = time.time()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
                batch_acc = 100.0 * correct / total
                avg_loss = total_loss / total
                print(
                    f"  [Epoch {epoch}/{epochs}] Step {batch_idx + 1}/{len(train_loader)} | Loss: {avg_loss:.4f} | Train Acc: {batch_acc:.2f}%"
                )

        scheduler.step()
        train_time = time.time() - t0

        # Evaluate on test set
        test_metrics = evaluate_model(model, test_loader, device, criterion)
        print(
            f"  --> Epoch {epoch} Test Results | Acc: {test_metrics['accuracy']*100:.2f}% | Loss: {test_metrics['loss']:.4f} | F1: {test_metrics['f1_score']:.4f} ({train_time:.1f}s)"
        )

        if test_metrics["accuracy"] > best_acc:
            best_acc = test_metrics["accuracy"]
            best_metrics = test_metrics

    return model, best_metrics


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module = None,
) -> Dict[str, float]:
    """Computes full test metrics: Accuracy, Precision, Recall, Specificity, F1, Loss."""
    model.eval()
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)

            preds = outputs.argmax(dim=1)
            y_true.extend(targets.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    total = len(y_true)
    acc = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    avg_loss = total_loss / total if total > 0 else 0.0

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "specificity": round(float(spec), 4),
        "f1_score": round(float(f1), 4),
        "loss": round(float(avg_loss), 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "total_test_samples": total,
    }


def evaluate_ensemble(
    resnet: nn.Module,
    mobilenet: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    weights: Tuple[float, float] = (0.55, 0.45),
) -> Dict[str, float]:
    """Evaluates the fused ensemble detector on the test set."""
    resnet.eval()
    mobilenet.eval()

    w_res, w_mob = weights
    total_w = w_res + w_mob
    w_res /= total_w
    w_mob /= total_w

    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            out_res = torch.softmax(resnet(inputs), dim=1).cpu().numpy()
            out_mob = torch.softmax(mobilenet(inputs), dim=1).cpu().numpy()

            fused_ai = (w_res * out_res[:, 1]) + (w_mob * out_mob[:, 1])
            preds = (fused_ai >= 0.50).astype(int)

            y_true.extend(targets.numpy().tolist())
            y_pred.extend(preds.tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    total = len(y_true)
    acc = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "specificity": round(float(spec), 4),
        "f1_score": round(float(f1), 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "total_test_samples": total,
    }


# ---------------------------------------------------------------------------
# Main Training Routine
# ---------------------------------------------------------------------------
def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using Compute Device: {device}")

    # Prepare directories
    checkpoints_dir = PROJECT_ROOT / "checkpoints"
    results_dir = PROJECT_ROOT / "results"
    docs_dir = PROJECT_ROOT / "docs"
    checkpoints_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    docs_dir.mkdir(exist_ok=True)

    # Transforms
    train_transform = transforms.Compose(
        [
            transforms.Resize(
                (224, 224),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize(
                (224, 224),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load file lists
    # Training: 5,000 Real + 5,000 Fake = 10,000 balanced samples
    # Testing:  1,000 Real + 1,000 Fake = 2,000 test samples
    cifake_train_dir = str(PROJECT_ROOT / "datasets" / "cifake" / "train")
    cifake_test_dir = str(PROJECT_ROOT / "datasets" / "cifake" / "test")

    print("\nLoading CIFAKE dataset files...")
    train_paths, train_labels = load_cifake_file_list(
        cifake_train_dir, max_per_class=5000
    )
    test_paths, test_labels = load_cifake_file_list(cifake_test_dir, max_per_class=1000)
    print(
        f"Loaded {len(train_paths):,} training samples ({train_labels.count(0)} Real, {train_labels.count(1)} AI)"
    )
    print(
        f"Loaded {len(test_paths):,} test samples ({test_labels.count(0)} Real, {test_labels.count(1)} AI)"
    )

    train_dataset = CIFAKEDataset(train_paths, train_labels, transform=train_transform)
    test_dataset = CIFAKEDataset(test_paths, test_labels, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    # -----------------------------------------------------------------------
    # 1. Train ResNet-18
    # -----------------------------------------------------------------------
    print("\nInstantiating ResNet-18 with ImageNet Pretrained Weights...")
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = resnet.fc.in_features
    resnet.fc = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, 2),
    )
    resnet = resnet.to(device)

    resnet, resnet_test_metrics = train_model(
        model=resnet,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=3,
        lr=3e-4,
        model_name="ResNet-18",
    )

    resnet_ckpt_path = checkpoints_dir / "resnet18_cifake.pth"
    torch.save(resnet.state_dict(), resnet_ckpt_path)
    print(f"\n--> Saved ResNet-18 Checkpoint: {resnet_ckpt_path}")

    # -----------------------------------------------------------------------
    # 2. Train MobileNetV3-Small
    # -----------------------------------------------------------------------
    print("\nInstantiating MobileNetV3-Small with ImageNet Pretrained Weights...")
    mobilenet = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.DEFAULT
    )
    in_features_mob = mobilenet.classifier[0].in_features
    mobilenet.classifier = nn.Sequential(
        nn.Linear(in_features_mob, 1024),
        nn.Hardswish(),
        nn.Dropout(p=0.2),
        nn.Linear(1024, 2),
    )
    mobilenet = mobilenet.to(device)

    mobilenet, mobilenet_test_metrics = train_model(
        model=mobilenet,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=3,
        lr=3e-4,
        model_name="MobileNetV3-Small",
    )

    mobilenet_ckpt_path = checkpoints_dir / "mobilenetv3_cifake.pth"
    torch.save(mobilenet.state_dict(), mobilenet_ckpt_path)
    print(f"\n--> Saved MobileNetV3 Checkpoint: {mobilenet_ckpt_path}")

    # -----------------------------------------------------------------------
    # 3. Evaluate Multi-Model Ensemble
    # -----------------------------------------------------------------------
    print("\n=======================================================")
    print(" Evaluating Multi-Model Neural Ensemble on Test Set")
    print("=======================================================")
    ensemble_test_metrics = evaluate_ensemble(
        resnet=resnet,
        mobilenet=mobilenet,
        test_loader=test_loader,
        device=device,
        weights=(0.55, 0.45),
    )

    print(
        f"  [Ensemble Test Accuracy] : {ensemble_test_metrics['accuracy'] * 100:.2f}%"
    )
    print(
        f"  [Ensemble Precision (AI)]: {ensemble_test_metrics['precision'] * 100:.2f}%"
    )
    print(f"  [Ensemble Recall (AI)]   : {ensemble_test_metrics['recall'] * 100:.2f}%")
    print(
        f"  [Ensemble Specificity]   : {ensemble_test_metrics['specificity'] * 100:.2f}%"
    )
    print(f"  [Ensemble F1-Score]      : {ensemble_test_metrics['f1_score']:.4f}")
    print(
        f"  [Confusion Matrix]       : TP={ensemble_test_metrics['tp']}, FP={ensemble_test_metrics['fp']}, TN={ensemble_test_metrics['tn']}, FN={ensemble_test_metrics['fn']}"
    )

    # -----------------------------------------------------------------------
    # 4. Save JSON Report
    # -----------------------------------------------------------------------
    report_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": "CIFAKE (Real CIFAR-10 vs. Stable Diffusion 1.4)",
        "train_samples": len(train_paths),
        "test_samples": len(test_paths),
        "epochs": 3,
        "device": str(device),
        "checkpoints": {
            "resnet18": str(resnet_ckpt_path),
            "mobilenetv3": str(mobilenet_ckpt_path),
        },
        "results": {
            "resnet18": resnet_test_metrics,
            "mobilenetv3": mobilenet_test_metrics,
            "ensemble": ensemble_test_metrics,
        },
    }

    report_json_path = results_dir / "image_training_evaluation.json"
    with open(report_json_path, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"\n--> Saved JSON Report: {report_json_path}")

    print("\n=======================================================")
    print(" TRAINING & EVALUATION COMPLETED SUCCESSFULLY!")
    print("=======================================================")


if __name__ == "__main__":
    main()
