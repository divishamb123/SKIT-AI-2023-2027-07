# Audio API Contract

This document defines the REST API contract for the audio detection service implemented in the API Gateway.

## Endpoints

### 1. Upload Audio for Detection
**Endpoint**: `POST /api/v1/detect/audio`

**Description**: Accepts an audio file (e.g., `.wav`, `.flac`, `.mp3`), validates it, generates a UUID-based object key, uploads it to MinIO storage, and creates a queued database job. The job is then published to the RabbitMQ `forensics.audio.detect` queue for the background worker to consume.

**Authentication**: Required (JWT Bearer Token).

**Request**:
*   `multipart/form-data`
*   **Field**: `file` (binary, max 20MB)
*   **Allowed MIME types**: `audio/wav`, `audio/x-wav`, `audio/flac`, `audio/ogg`, `audio/mpeg`, `audio/mp3`, and anything starting with `audio/`.

**Response (Success - 202 Accepted)**:
```json
{
  "job_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "queued",
  "message": "Audio successfully uploaded and queued for detection."
}
```

**Errors**:
*   `413 Payload Too Large`: If file exceeds 20MB.
*   `415 Unsupported Media Type`: If file is not an allowed audio format.
*   `500 Internal Server Error`: If MinIO upload or RabbitMQ publish fails.

### Internal Message Payload (RabbitMQ)
Once accepted, the API gateway sends the following JSON payload to the `forensics.audio.detect` queue:

```json
{
  "job_id": "123e4567-e89b-12d3-a456-426614174000",
  "object_key": "123e4567-e89b-12d3-a456-426614174000-filename.wav",
  "modality": "audio"
}
```
*Note: The consumer accesses the uploaded audio file from MinIO using the `object_key` value.*
