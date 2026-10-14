# Sprint 1 Database Schema

Based on the Sprint 1 setup (as defined in the `001_initial_migration.py` Alembic migration and the models in `shared/db`), the following is the database schema for the core infrastructure.

## Entity-Relationship Diagram

```mermaid
erDiagram
    users {
        UUID id PK
        String name
        String email UK
        String password_hash
        String role "Enum: user, admin, analyst"
        Boolean is_active
        DateTime created_at
        DateTime updated_at
    }
    
    jobs {
        UUID id PK
        UUID user_id FK
        job_status status
        modality_type modality
        String input_type
        Text input_text
        String input_object_key
        String file_name
        BigInteger file_size_bytes
        String file_mime_type
        String file_sha256
        SmallInteger priority "1 to 10"
        SmallInteger retry_count
        SmallInteger max_retries
        String queue_name
        String worker_id
        Text error_message
        String error_code
        DateTime queued_at
        DateTime started_at
        DateTime completed_at
        DateTime created_at
        DateTime deleted_at
    }

    reports {
        UUID id PK
        UUID job_id FK
        report_type report_type
        report_status status
        String minio_object_key
        BigInteger file_size_bytes
        String file_sha256
        Text signed_url
        DateTime signed_url_expires
        Integer generation_time_ms
        Text error_message
        DateTime generated_at
        DateTime created_at
    }

    users ||--o{ jobs : "submits"
    jobs ||--o{ reports : "generates"
```

## Tables Overview

### 1. `users`
- Handles the core authentication schema, storing the user's `email` and Argon2-encrypted `password_hash`. 
- A `role` column uses a check constraint allowing only `'user'`, `'admin'`, or `'analyst'`.

### 2. `jobs`
- Tracks inference requests submitted by users, linked via `user_id`.
- Supports different input configurations (validated by constraints), storing either raw `input_text` for text detection or an `input_object_key` representing a file in MinIO (for images/audio/video).
- Core ENUMs defined in PostgreSQL:
    - `job_status`: `'queued'`, `'processing'`, `'completed'`, `'failed'`, `'cancelled'`
    - `modality_type`: `'text'`, `'image'`, `'audio'`, `'video'`, `'multimodal'`
- Tracks telemetry, metadata, error details, and timeline information (queued, started, completed).

### 3. `reports`
- Represents the resulting forensics and analysis output tied to a completed `job`.
- `report_type` ENUM: `'pdf'`, `'json'`, `'html'`
- `report_status` ENUM: `'pending'`, `'generating'`, `'completed'`, `'failed'`
- Contains a unique combination constraint on `job_id` and `report_type` so each job only generates one report of a specific type.
- Points to the finalized report payload living in object storage (`minio_object_key` / `signed_url`).
