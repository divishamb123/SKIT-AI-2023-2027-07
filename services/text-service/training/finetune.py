"""
finetune.py — Sprint 3 Part 1: Fine-Tuning Pipeline for Text Model (DeBERTa-v3)

A reproducible, device-agnostic CLI pipeline to fine-tune `microsoft/deberta-v3-base`
(or compatible transformers sequence classifiers) for binary AI-vs-Human text detection.

Key Invariants:
- Loads partitioned splits from datasets/splits/text_manifest.csv (or specified path).
- Explicit label mappings: id2label={0: 'human', 1: 'ai'}, label2id={'human': 0, 'ai': 1}.
- Evaluation metrics: accuracy and macro F1 computed via compute_metrics.
- Device agnostic: supports CUDA, MPS, and CPU. Forces fp16=False when CUDA is absent.
- Best model selection: loads best checkpoint at end based on macro F1.
- Output serialization: saves fine-tuned model weights, tokenizer, and metrics.json.
- Quick smoke testing: supports --max-samples for CPU verification without full training.
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    set_seed,
)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

ID2LABEL = {0: "human", 1: "ai"}
LABEL2ID = {"human": 0, "ai": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune DeBERTa-v3 for AI vs. Human text detection."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="microsoft/deberta-v3-base",
        help="Pretrained Hugging Face model identifier or local checkpoint path.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="services/text-service/outputs/deberta-v3-base-finetuned",
        help="Directory to save fine-tuned model, tokenizer, and metrics.json.",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default="datasets/splits/text_manifest.csv",
        help="Path to the unified text manifest CSV file.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Per-device batch size for training and evaluation.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Initial learning rate for AdamW optimizer.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum token sequence length.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for regularization.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional upper bound on dataset samples for quick smoke testing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze transformer backbone (train classifier head only, ideal for fast CPU smoke tests).",
    )
    return parser.parse_args()


def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """Computes accuracy and macro F1 score for evaluation predictions."""
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    if isinstance(logits, tuple):
        logits = logits[0]

    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
    }


def resolve_manifest_path(manifest_arg: str) -> Path:
    """Resolves manifest path relative to script, CWD, or repo root."""
    candidate = Path(manifest_arg)
    if candidate.is_file():
        return candidate.resolve()

    # Try resolving relative to repository root
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    root_candidate = repo_root / manifest_arg
    if root_candidate.is_file():
        return root_candidate.resolve()

    raise FileNotFoundError(
        f"Could not locate text manifest at '{manifest_arg}' or '{root_candidate}'."
    )


def load_and_preprocess_splits(
    manifest_path: Path, max_samples: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads manifest CSV, validates schema, filters train/val splits, and optionally caps sample count."""
    logger.info("Loading text manifest from: %s", manifest_path)
    df = pd.read_csv(manifest_path)

    required_cols = {"split", "label", "text"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Manifest missing required columns: {missing}. Available: {list(df.columns)}"
        )

    # Clean missing values
    df = df.dropna(subset=["text", "label"]).copy()
    df["label"] = df["label"].astype(int)

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()

    if train_df.empty or val_df.empty:
        raise ValueError(
            f"Manifest must contain non-empty 'train' and 'val' splits. "
            f"Found {len(train_df)} train rows and {len(val_df)} val rows."
        )

    if max_samples is not None and max_samples > 0:
        train_count = min(len(train_df), max_samples)
        val_count = min(len(val_df), max(16, max_samples // 2))
        logger.info(
            "Quick smoke testing enabled (--max-samples %d): capping train to %d, val to %d",
            max_samples,
            train_count,
            val_count,
        )
        train_df = train_df.iloc[:train_count].copy()
        val_df = val_df.iloc[:val_count].copy()

    logger.info("Split counts: train=%d, val=%d", len(train_df), len(val_df))
    return train_df, val_df


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_file = resolve_manifest_path(args.manifest_path)
    train_df, val_df = load_and_preprocess_splits(manifest_file, args.max_samples)

    # Device detection and precision policy
    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    device = "cuda" if has_cuda else ("mps" if has_mps else "cpu")
    use_fp16 = has_cuda  # Force fp16=False on CPU / MPS

    logger.info("Device selected: %s (fp16=%s)", device, use_fp16)
    logger.info("Initializing tokenizer: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)

    logger.info("Tokenizing datasets (max_length=%d)...", args.max_length)

    def tokenize_fn(batch: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
        )

    train_ds = Dataset.from_pandas(train_df[["text", "label"]])
    val_ds = Dataset.from_pandas(val_df[["text", "label"]])

    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    logger.info(
        "Loading model: %s with id2label=%s, label2id=%s",
        args.model_name,
        ID2LABEL,
        LABEL2ID,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # For CPU smoke tests or when explicitly requested, freeze the transformer backbone
    # so backward passes only update the classifier head, running in seconds on CPU.
    if args.freeze_backbone or (args.max_samples is not None and not has_cuda):
        logger.info("Freezing transformer backbone for fast CPU smoke testing...")
        base_encoder = getattr(model, "deberta", None) or getattr(
            model, "base_model", None
        )
        if base_encoder is not None:
            for param in base_encoder.parameters():
                param.requires_grad = False

    training_kwargs: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "fp16": use_fp16,
        "dataloader_pin_memory": False if not has_cuda else True,
        "logging_steps": 10 if args.max_samples is None else 1,
        "seed": args.seed,
        "report_to": "none",
    }

    try:
        training_args = TrainingArguments(eval_strategy="epoch", **training_kwargs)
    except TypeError:
        training_args = TrainingArguments(
            evaluation_strategy="epoch", **training_kwargs
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training loop (epochs=%d)...", args.epochs)
    train_result = trainer.train()

    logger.info("Running final evaluation on validation set...")
    eval_metrics = trainer.evaluate()

    logger.info("Saving best model and tokenizer to: %s", output_dir)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    training_loss = (
        float(train_result.training_loss)
        if train_result.training_loss is not None
        else float(train_result.metrics.get("train_loss", 0.0))
    )

    metrics_payload = {
        "train_loss": training_loss,
        "eval_loss": float(eval_metrics.get("eval_loss", 0.0)),
        "eval_accuracy": float(eval_metrics.get("eval_accuracy", 0.0)),
        "eval_macro_f1": float(eval_metrics.get("eval_macro_f1", 0.0)),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": args.model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_samples": args.max_samples,
        "max_length": args.max_length,
        "device": device,
    }

    metrics_file = output_dir / "metrics.json"
    logger.info("Writing metrics to: %s", metrics_file)
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    logger.info("Fine-tuning pipeline execution finished successfully.")
    print("\n--- Training & Evaluation Summary ---")
    print(f"Training Loss:  {training_loss:.4f}")
    print(f"Eval Loss:      {metrics_payload['eval_loss']:.4f}")
    print(f"Eval Accuracy:  {metrics_payload['eval_accuracy']:.4f}")
    print(f"Eval Macro F1:  {metrics_payload['eval_macro_f1']:.4f}")
    print(f"Saved To:       {output_dir}")
    print(f"Metrics File:   {metrics_file}\n")


if __name__ == "__main__":
    main()
