# Audio Detection API Contract

**Service:** API Gateway (`services/api-gateway`)
**Author:** Dev Khandelwal (23ESKCA035)

---

## POST /api/v1/detect/audio

Upload an audio file for deepfake detection.

**Auth:** JWT Bearer token required

**Request:** `multipart/form-data`

| Field | Type | Constraints |
|---|---|---|
| `file` | binary | Max 20 MB. Allowed types: `audio/wav`, `audio/x-wav`, `audio/flac`, `audio/ogg`, `audio/mpeg`, `audio/mp3` |

**Response 202 Accepted:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "message": "Audio successfully uploaded and queued for detection."
}
```

**What happens after 202:**
1. File is stored in MinIO under a UUID-based key
2. A job record is created in PostgreSQL with status `queued`
3. A message is published to RabbitMQ queue `forensics.audio.detect`
4. The audio-service worker picks up the job, runs AASIST-L inference, and updates the job status to `completed` or `failed`

**Check job result:** `GET /api/v1/jobs/{job_id}`

**Error responses:**

| Code | Meaning |
|---|---|
| 401 | Missing or invalid JWT |
| 413 | File exceeds 20 MB |
| 415 | Unsupported MIME type |
| 429 | Rate limit exceeded (20 requests/minute) |
| 500 | MinIO upload or RabbitMQ publish failed |

---

## GET /api/v1/jobs/{job_id}

Returns current status of a detection job.

**Auth:** JWT Bearer token required (must own the job)

**Response 200:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "modality": "audio",
  "file_name": "sample.wav",
  "created_at": "2026-10-15T10:30:00Z",
  "error_message": null
}
```
