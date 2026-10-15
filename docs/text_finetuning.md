# Text Model Fine-Tuning Pipeline & Serving Guide

**Project ID:** `SKIT/AI/2023-2027/07`  
**Modality:** Text Detection  
**Model Architecture:** `microsoft/deberta-v3-base`  
**Dataset Benchmark:** Robust AI Detection (RAID) Benchmark  

---

## 📌 1. Overview & Forensic Objective

The text detection module classifies submitted textual content as either **Human Written** (`0`) or **AI Generated** (`1`). While Sprint 2 established the microservice baseline using pretrained `microsoft/deberta-v3-base`, Sprint 3 introduces the fine-tuning pipeline to specialize DeBERTa's bidirectional disentangled attention representations on adversarial and diverse generative AI patterns.

### Label Specification
- `0`: `human` (Human written literature, news, academic abstracts, recipes, poems)
- `1`: `ai` (AI generated text across 11 generative models: ChatGPT, GPT-4, Claude, LLaMA, Mistral, Cohere, etc.)

---

## 🗂️ 2. Dataset Organisation & Inputs

The pipeline trains on the deterministic dataset split generated in Sprint 1:
- **Location:** [`datasets/splits/text_manifest.csv`](../datasets/splits/text_manifest.csv)
- **Partitions:**
  - `train`: 16,000 samples (50% human / 50% AI)
  - `val`: 2,000 samples (50% human / 50% AI)
  - `test`: 2,000 samples (50% human / 50% AI, held out for final benchmark)
- **Zero Leakage:** Strictly disjoint splits generated with fixed seed `42`.

---

## 🚀 3. Execution Environments

### Mode A: Google Colab GPU (Full Training — Recommended)
Full training of DeBERTa-v3 on 18,000 train+val samples requires GPU acceleration (NVIDIA T4, V100, or A100).
1. Open the interactive Colab notebook: [`notebooks/text_finetune_colab.ipynb`](../notebooks/text_finetune_colab.ipynb).
2. Set runtime type to **T4 GPU** (`Runtime > Change runtime type > T4 GPU`).
3. Execute the cells to clone the repository, install dependencies from `services/text-service/training/requirements-train.txt`, and run `finetune.py`.
4. Download the generated `deberta_v3_finetuned.zip` archive containing the fine-tuned weights, tokenizer, and `metrics.json`.

### Mode B: Local CPU Smoke-Testing
For CI/CD and developer verification without running hours of compute on a laptop:
```bash
python services/text-service/training/finetune.py \
    --max-samples 64 \
    --epochs 1 \
    --batch-size 4 \
    --max-length 64 \
    --output-dir services/text-service/outputs/smoke-test
```
This executes the entire pipeline (manifest parsing, tokenization, PyTorch forward/backward pass, validation evaluation, metric calculation, checkpoint serialization, and `metrics.json` export) in under two minutes on CPU.

---

## ⚙️ 4. Hyperparameter Choices & Rationale

| Hyperparameter | Default | Design Rationale |
|---|---|---|
| `--model-name` | `microsoft/deberta-v3-base` | State-of-the-art disentangled attention mechanism superior to standard BERT/RoBERTa for forensic linguistic pattern discovery. |
| `--epochs` | `3` | Empirically sufficient for convergence on 16k samples without catastrophic forgetting or overfitting. |
| `--batch-size` | `8` | Fits comfortably within 16 GB Colab T4 GPU VRAM while maintaining stable AdamW gradient estimates. |
| `--learning-rate` | `2e-5` | Standard fine-tuning rate for DeBERTa with linear decay and AdamW optimizer. |
| `--max-length` | `512` | Captures long document context up to DeBERTa's default positional embedding limit. |
| `--weight-decay` | `0.01` | L2 regularization applied to non-bias/LayerNorm parameters to prevent overfitting. |
| `metric_for_best_model` | `macro_f1` | Macro F1 balances performance evenly across both classes regardless of minor partition fluctuations. |

---

## 📊 5. Output Artifacts & Metrics Specification

Upon training completion, the output directory (`--output-dir`) contains:
- `model.safetensors` (or `pytorch_model.bin`): fine-tuned model weights.
- `config.json`: architecture configuration with explicit `id2label` and `label2id`.
- `tokenizer.json`, `spm.model`, `special_tokens_map.json`: tokenizer assets.
- `metrics.json`: structured evaluation results matching the schema:

```json
{
  "train_loss": 0.3421,
  "eval_loss": 0.2845,
  "eval_accuracy": 0.8950,
  "eval_macro_f1": 0.8948,
  "timestamp": "2026-10-15T19:30:00.000000+00:00",
  "model_name": "microsoft/deberta-v3-base",
  "epochs": 3,
  "batch_size": 8,
  "learning_rate": 0.00002,
  "max_samples": null,
  "max_length": 512,
  "device": "cuda"
}
```

---

## 🔄 6. Hand-off to the Inference Microservice

The standalone inference service (`services/text-service`) dynamically checks for a local fine-tuned model:

1. **Extract Model Artifacts:**
   Extract your trained model folder into `services/text-service/models/deberta-v3-base-finetuned/`.

2. **Configure Environment:**
   In `services/text-service/.env` (see [`services/text-service/.env.example`](../services/text-service/.env.example)):
   ```env
   MODEL_PATH=models/deberta-v3-base-finetuned
   ```

3. **Fallback Resiliency:**
   If `MODEL_PATH` is empty, unset, or points to an invalid directory, `TextDetector` in [`services/text-service/app/core/model.py`](../services/text-service/app/core/model.py) automatically falls back to `microsoft/deberta-v3-base` (pretrained baseline) without crashing.

4. **Service Verification:**
   Run the test suite to verify loading and inference contract:
   ```bash
   pytest services/text-service/tests/test_predict.py
   ```
