# Cross-Generator Generalization Evaluation Report

**Sprint:** Sprint 3 (Accuracy Optimisation) — Week 2 Deliverable  
**Member:** Divisha Manak Bohra (`23ESKCA038`) — Image Detection Modality & Image Interface  
**Project ID:** `SKIT/AI/2023-2027/07`  
**Ensemble Version:** `ensemble-resnet18-mobilenetv3-v1.0`  
**Device:** `mps`  
**Decision Threshold (τ):** `0.5`  
**Timestamp:** `2026-11-16T09:33:06.328040+00:00`  

---

## 1. Executive Summary

This report evaluates the multi-model ensemble detector's ability to generalise from in-domain generators (seen during training) to out-of-domain / unseen generators. A smaller generalization gap indicates stronger cross-domain robustness.

| Metric | Value |
|---|---|
| Total Samples Evaluated | **60** |
| Overall Accuracy | **48.33%** |
| Overall F1 Score | **0.3673** |
| In-Domain Accuracy | **35.00%** |
| Out-of-Domain Accuracy | **55.00%** |
| **Generalization Gap (Δ)** | **-20.00%** |
| Gap Severity | **SIGNIFICANT** |
| Total Latency | **1013.8 ms** |

---

## 2. Per-Generator Breakdown

| Generator | Domain | Samples | Accuracy | Precision | Recall | F1 | Robustness | Confidence | Throughput |
|---|---|---|---|---|---|---|---|---|---|
| **biggan** | out_of_domain | 10 | 50.0% | 0.0% | 0.0% | 0.000 | 1.000 | 0.775 | 28.9 FPS |
| **cifar10** | in_domain | 10 | 50.0% | 0.0% | 0.0% | 0.000 | 1.000 | 0.729 | 62.3 FPS |
| **midjourney** | out_of_domain | 10 | 50.0% | 0.0% | 0.0% | 0.000 | 1.000 | 0.736 | 76.8 FPS |
| **real** | out_of_domain | 10 | 50.0% | 50.0% | 100.0% | 0.667 | 1.000 | 0.752 | 80.0 FPS |
| **stable-diffusion** | out_of_domain | 10 | 70.0% | 100.0% | 40.0% | 0.571 | 0.800 | 0.599 | 81.3 FPS |
| **stable-diffusion-1.4** | in_domain | 10 | 20.0% | 28.6% | 40.0% | 0.333 | 0.700 | 0.668 | 77.4 FPS |

---

## 3. Domain-Level Analysis

### 3.1 In-Domain Performance

**Generators:** cifar10, stable-diffusion-1.4  
**Total Samples:** 20  

| Metric | Value |
|---|---|
| Accuracy | 35.00% |
| Precision | 28.57% |
| Recall | 20.00% |
| Specificity | 50.00% |
| F1 Score | 0.2353 |
| Mean Confidence | 0.6989 |
| Mean Robustness | 0.8500 |

### 3.2 Out-of-Domain / Unseen Generator Performance

**Generators:** biggan, midjourney, real, stable-diffusion  
**Total Samples:** 40  

| Metric | Value |
|---|---|
| Accuracy | 55.00% |
| Precision | 58.33% |
| Recall | 35.00% |
| Specificity | 75.00% |
| F1 Score | 0.4375 |
| Mean Confidence | 0.7157 |
| Mean Robustness | 0.9500 |

---

## 4. Generalization Gap Analysis

| Metric | In-Domain | Out-of-Domain | Gap (Δ) |
|---|---|---|---|
| Accuracy | 35.00% | 55.00% | -20.00% |
| F1 Score | 0.2353 | 0.4375 | -0.2022 |
| Precision | 28.57% | 58.33% | -29.76% |
| Recall | 20.00% | 35.00% | -15.00% |

**Severity Assessment:** SIGNIFICANT  

---

## 5. Per-Generator Confusion Matrices

### biggan (out_of_domain)
```text
                 Predicted Real    Predicted AI
Actual Real          5                0         
Actual AI            5                0         
```

### cifar10 (in_domain)
```text
                 Predicted Real    Predicted AI
Actual Real          5                0         
Actual AI            5                0         
```

### midjourney (out_of_domain)
```text
                 Predicted Real    Predicted AI
Actual Real          5                0         
Actual AI            5                0         
```

### real (out_of_domain)
```text
                 Predicted Real    Predicted AI
Actual Real          0                5         
Actual AI            0                5         
```

### stable-diffusion (out_of_domain)
```text
                 Predicted Real    Predicted AI
Actual Real          5                0         
Actual AI            3                2         
```

### stable-diffusion-1.4 (in_domain)
```text
                 Predicted Real    Predicted AI
Actual Real          0                5         
Actual AI            3                2         
```

---

## 6. Engineering Notes & Analysis

Baseline cross-generator generalization evaluation using untrained ensemble weights. With random initialization (pre-Sprint 3 fine-tuning), accuracy is expected at ~50% (random chance) across all generators. The generalization gap is minimal at this stage, confirming that the evaluation pipeline correctly computes and reports metrics. After fine-tuning, this pipeline will reveal real performance differences between in-domain and out-of-domain generators.
