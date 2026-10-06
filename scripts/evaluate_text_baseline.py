import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm


def main():
    project_root = Path(__file__).resolve().parent.parent
    manifest_path = project_root / "datasets" / "splits" / "text_manifest.csv"
    output_path = project_root / "docs" / "text_baseline_results.md"

    if not manifest_path.exists():
        print(f"Manifest not found at {manifest_path}")
        sys.exit(1)

    print("Loading test dataset...")
    df = pd.read_csv(manifest_path)
    test_df = df[df["split"] == "test"].copy()

    print(f"Test set size: {len(test_df)}")
    if len(test_df) == 0:
        print("No test data found.")
        sys.exit(1)

    print("Subsampling 200 items for baseline eval (CPU performance)...")
    human_df = test_df[test_df["label"] == 0].head(100)
    ai_df = test_df[test_df["label"] == 1].head(100)
    test_df = pd.concat([human_df, ai_df])

    model_name = "microsoft/deberta-v3-base"
    print(f"Loading pretrained model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    y_true = []
    y_pred = []

    print("Running inference...")
    # Process in batches to speed up
    batch_size = 16
    texts = test_df["text"].tolist()
    labels = test_df["label"].tolist()  # 0 for human, 1 for AI

    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i : i + batch_size]
        batch_labels = labels[i : i + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().tolist()

        y_true.extend(batch_labels)
        y_pred.extend(preds)

    print("Calculating metrics...")
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    report = classification_report(y_true, y_pred, target_names=["Human (0)", "AI (1)"])
    print(report)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Text Detection Baseline Results\n\n")
        f.write("**Model:** `microsoft/deberta-v3-base` (PRETRAINED ONLY)\n\n")
        f.write(
            "This document records the baseline accuracy and metrics for the completely untrained (pretrained only) DeBERTa-v3-base model on the `test` split of our RAID-based text dataset. Because the model has not yet been fine-tuned for the binary AI vs. Human text classification task, its performance is expected to be near random chance (~50%). Fine-tuning is explicitly scheduled for Sprint 3.\n\n"
        )
        f.write("## Metrics\n")
        f.write(f"- **Accuracy:** {acc:.4f}\n")
        f.write(f"- **Precision:** {prec:.4f}\n")
        f.write(f"- **Recall:** {rec:.4f}\n")
        f.write(f"- **F1 Score:** {f1:.4f}\n\n")
        f.write("## Classification Report\n")
        f.write("```text\n")
        f.write(report)
        f.write("\n```\n")

    print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
