# Image Detection Baseline Evaluation Report — Grouped Input Samples

**Sprint:** Sprint 2 (Baseline Detection Service) — Week 4 Deliverable  
**Member:** Divisha Manak Bohra (`23ESKCA038`) — Image Detection Modality & Image Interface  
**Project ID:** `SKIT/AI/2023-2027/07`  
**Model Version:** `baseline-resnet18-v1.0`  
**Device Utilized:** `mps`  
**Timestamp:** `2026-10-11T11:53:28.054976+00:00`  

---

## 1. Executive Summary & Overall Baseline Performance

This evaluation assesses the baseline **ResNet-18** image classifier across representative grouped sample partitions from `manifest.csv`. Because the baseline model uses untrained random classifier head weights prior to Sprint 3 fine-tuning, its raw binary classification accuracy is expected to hover around random chance (~50%), while serving as the verified baseline reference point.

| Metric | Value |
|---|---|
| Total Samples Evaluated | **400** |
| Failed / Inaccessible | **0** |
| Overall Accuracy | **62.75%** |
| Precision (AI) | **62.66%** |
| Recall (AI) | **100.00%** |
| Specificity (Real) | **0.67%** |
| F1-Score | **0.7704** |
| Mean Confidence | **0.9903** |
| Total Latency | **2270.7 ms** |
| Avg Latency / Image | **5.68 ms** |
| Throughput | **176.2 FPS** |

### Confusion Matrix (Overall)
```text
                 Predicted Real    Predicted AI
Actual Real          1                149       
Actual AI            0                250       
```

---

## 2. Grouped Breakdown Analysis

### Grouped By Class

| Group Name | Samples | Accuracy | Precision | Recall | F1 | Mean Conf | Latency (ms) | Throughput (FPS) |
|---|---|---|---|---|---|---|---|---|
| **Real (Class 0)** | 150 | 0.7% | 0.0% | 0.0% | 0.000 | 0.985 | 565.1 | 265.4 |
| **AI-Generated (Class 1)** | 250 | 100.0% | 100.0% | 100.0% | 1.000 | 0.993 | 1597.6 | 156.5 |

### Grouped By Split

| Group Name | Samples | Accuracy | Precision | Recall | F1 | Mean Conf | Latency (ms) | Throughput (FPS) |
|---|---|---|---|---|---|---|---|---|
| **split:holdout** | 200 | 75.0% | 75.0% | 100.0% | 0.857 | 0.989 | 1565.9 | 127.7 |
| **split:test** | 200 | 50.5% | 50.2% | 100.0% | 0.669 | 0.991 | 548.9 | 364.3 |

### Grouped By Generator

| Group Name | Samples | Accuracy | Precision | Recall | F1 | Mean Conf | Latency (ms) | Throughput (FPS) |
|---|---|---|---|---|---|---|---|---|
| **generator:biggan** | 50 | 100.0% | 100.0% | 100.0% | 1.000 | 0.991 | 196.3 | 254.7 |
| **generator:cifar10** | 100 | 1.0% | 0.0% | 0.0% | 0.000 | 0.986 | 296.5 | 337.2 |
| **generator:midjourney** | 50 | 100.0% | 100.0% | 100.0% | 1.000 | 0.994 | 824.0 | 60.7 |
| **generator:real** | 50 | 0.0% | 0.0% | 0.0% | 0.000 | 0.984 | 213.3 | 234.5 |
| **generator:stable-diffusion** | 50 | 100.0% | 100.0% | 100.0% | 1.000 | 1.000 | 337.0 | 148.4 |
| **generator:stable-diffusion-1.4** | 100 | 100.0% | 100.0% | 100.0% | 1.000 | 0.997 | 264.6 | 377.9 |

### Batch Size Scaling

| Group Name | Samples | Accuracy | Precision | Recall | F1 | Mean Conf | Latency (ms) | Throughput (FPS) |
|---|---|---|---|---|---|---|---|---|
| **Group Size 1** | 64 | 0.0% | 0.0% | 0.0% | 0.000 | 0.988 | 294.4 | 217.4 |
| **Group Size 4** | 64 | 0.0% | 0.0% | 0.0% | 0.000 | 0.988 | 179.7 | 356.2 |
| **Group Size 8** | 64 | 0.0% | 0.0% | 0.0% | 0.000 | 0.988 | 175.4 | 364.8 |
| **Group Size 16** | 64 | 0.0% | 0.0% | 0.0% | 0.000 | 0.988 | 178.8 | 357.8 |
| **Group Size 32** | 64 | 0.0% | 0.0% | 0.0% | 0.000 | 0.988 | 182.6 | 350.4 |

---

## 3. Engineering Notes & Analysis

Baseline ResNet-18 classifier evaluated with initial weights on both in-domain (CIFAKE test) and out-of-domain (GenImage holdout) grouped partitions. Throughput and scaling confirmed across batch sizes up to 32. Supervised fine-tuning and ensemble tuning scheduled for Sprint 3.
