# AASIST-L Model Setup Guide

**Author:** Dev Khandelwal (23ESKCA035)
**Service:** `services/audio-service`

---

## What the model needs

The audio service uses the **AASIST-L** pretrained model (from the official NAVER AASIST repository). The weights file is NOT committed to Git (it's 85 MB, gitignored).

You need to place the weights at:
```
./models/aasist_l_v1.0/AASIST-L.pth
./models/aasist_l_v1.0/AASIST-L.conf
```

The Docker Compose file mounts `./models` as a read-only volume:
```yaml
volumes:
  - ./models:/models:ro
```

---

## Downloading the weights

Download from the official AASIST release:
```
https://github.com/clovaai/aasist/releases
```

Place the `.pth` and `.conf` files under `./models/aasist_l_v1.0/`.

---

## Degraded mode (no weights)

If the weights are absent, the service **starts anyway** in degraded mode:
- `/internal/health` returns HTTP 503 with:
  ```json
  {"status": "degraded", "reason": "model weights not found"}
  ```
- Jobs submitted to the queue are marked `FAILED` with `error_message: "model not ready"`

This prevents the container from crash-looping during development without the large weights file.

---

## Running with Docker Compose

```bash
# 1. Place weights
mkdir -p models/aasist_l_v1.0
cp /path/to/AASIST-L.pth models/aasist_l_v1.0/
cp /path/to/AASIST-L.conf models/aasist_l_v1.0/

# 2. Start services
cp .env.example .env
# edit .env to set passwords
docker-compose up --build audio-service
```

---

## Running without Docker (dev/CI mode)

The service uses environment variables to find the weights:
```bash
export AASIST_WEIGHTS_PATH=/absolute/path/to/AASIST-L.pth
export AASIST_CONFIG_PATH=/absolute/path/to/AASIST-L.conf
```

If these env vars are not set or the files don't exist, the service starts in degraded mode.
