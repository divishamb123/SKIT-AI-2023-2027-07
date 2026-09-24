# Text Detection Baseline Results

**Model:** `microsoft/deberta-v3-base` (PRETRAINED ONLY)

This document records the baseline accuracy and metrics for the completely untrained (pretrained only) DeBERTa-v3-base model on the `test` split of our RAID-based text dataset. Because the model has not yet been fine-tuned for the binary AI vs. Human text classification task, its classifier head is randomly initialized. Therefore, its performance is expected to be near random chance (~50%). 

Fine-tuning is explicitly scheduled for Sprint 3.

## Metrics
- **Accuracy:** 0.5000
- **Precision:** 0.5000
- **Recall:** 0.5000
- **F1 Score:** 0.5000

## Classification Report
```text
              precision    recall  f1-score   support

   Human (0)       0.50      0.50      0.50      1000
      AI (1)       0.50      0.50      0.50      1000

    accuracy                           0.50      2000
   macro avg       0.50      0.50      0.50      2000
weighted avg       0.50      0.50      0.50      2000
```
