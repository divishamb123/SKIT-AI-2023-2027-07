# AASIST-L Model Setup Guide

**Author:** Dev Khandelwal (23ESKCA035)  
**Sprint:** 2 — Week 2  
**Last updated:** 15 October 2026

---

## Overview

The audio detection service uses the **AASIST-L** pretrained model (published by
NAVER Corp.) for audio deepfake detection.  The pretrained weights are **not**
checked into the repository (`.gitignore` excludes `*.pth` / `*.pt` files).
They must be placed on the host machine and volume-mounted into the container at
runtime.

---

## Quick Start

### 1 — Download the pretrained weights

Download the AASIST-L checkpoint from the official repository:

```
https://github.com/clovaai/aasist/releases
```

You need two files:

| File | Purpose |
|------|---------|
| `AASIST-L.pth` | PyTorch model weights |
| `AASIST-L.conf` | Model architecture config (JSON) |

### 2 — Place files in the host `./models/` directory

```
SKIT-AI-2023-2027-07/
└── models/
    └── aasist_l_v1.0/
        ├── AASIST-L.pth    ← pretrained weights (~84 MB)
        └── AASIST-L.conf   ← model config JSON
```

### 3 — Start the stack

```bash
docker compose up --build
```

The `audio-service` Docker Compose definition mounts `./models` as a read-only
volume at `/models` inside the container:

```yaml
volumes:
  - ./models:/models:ro
```

---

## Graceful Degradation (No Weights)

If the weight file is absent, the audio-service **does not crash**.  Instead:

- Container starts successfully.
- `/internal/health` returns **HTTP 503** with:
  ```json
  {
    "status": "degraded",
    "model_loaded": false,
    "weights_path": "/models/aasist_l_v1.0/AASIST-L.pth",
    "error": "Weight file not found: ..."
  }
  ```
- Any RabbitMQ messages received while in degraded mode will have their database
  job status set to `"failed"` with an `error_message` describing the missing
  weights.

---

## Custom Weight Paths (Environment Variables)

If you want to store weights at a different host location, override the default
container paths via environment variables in `.env` or `docker-compose.yml`:

```env
AASIST_WEIGHTS_PATH=/custom/path/weights.pth
AASIST_CONFIG_PATH=/custom/path/config.conf
```

---

## Verifying the Model is Loaded

After `docker compose up`, run:

```bash
curl http://localhost:8003/internal/health
```

Expected response when weights are present:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "weights_path": "/models/aasist_l_v1.0/AASIST-L.pth",
  "device": "cpu"
}
```

---

## CI / Development Without Weights

In CI environments and local development without weights, the service starts in
degraded mode and the health endpoint returns 503.  The audio *pipeline tests*
(preprocessing, shape validation) still pass independently of the model weights.

---

## Security Note

Do **not** commit pretrained weights to the repository.  The `.gitignore` already
excludes `*.pth` and `*.pt` files.  Weights are large binary files and may be
subject to licence restrictions from NAVER Corp.
