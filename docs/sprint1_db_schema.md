# Sprint 1 — Database Schema

Source of truth: [`shared/db/alembic/versions/001_initial_migration.py`](../../shared/db/alembic/versions/001_initial_migration.py)

Three tables are created in this migration. All use UUID primary keys (PostgreSQL `uuid` type, default `gen_random_uuid()`).

---

## `users`

| Column | Type | Constraints |
|---|---|---|
| `id` | UUID | PK, default `gen_random_uuid()` |
| `name` | VARCHAR | NOT NULL |
| `email` | VARCHAR | NOT NULL, UNIQUE |
| `password_hash` | VARCHAR | NOT NULL — Argon2 hash |
| `role` | VARCHAR | NOT NULL, CHECK IN (`'user'`, `'admin'`, `'analyst'`), default `'user'` |
| `is_active` | BOOLEAN | NOT NULL, default `true` |
| `created_at` | TIMESTAMP | NOT NULL, default `now()` |
| `updated_at` | TIMESTAMP | NOT NULL, default `now()` |

---

## `jobs`

| Column | Type | Constraints |
|---|---|---|
| `id` | UUID | PK |
| `user_id` | UUID | FK → `users.id`, NOT NULL |
| `status` | `job_status` ENUM | NOT NULL, default `'queued'` |
| `modality` | `modality_type` ENUM | NOT NULL |
| `input_type` | VARCHAR | NOT NULL |
| `input_text` | TEXT | nullable (text modality only) |
| `input_object_key` | VARCHAR | nullable (file modalities) |
| `file_name` | VARCHAR | nullable |
| `file_size_bytes` | BIGINT | nullable |
| `file_mime_type` | VARCHAR | nullable |
| `file_sha256` | VARCHAR | nullable |
| `priority` | SMALLINT | NOT NULL, default `5`, CHECK 1–10 |
| `retry_count` | SMALLINT | NOT NULL, default `0` |
| `max_retries` | SMALLINT | NOT NULL, default `3` |
| `queue_name` | VARCHAR | nullable |
| `worker_id` | VARCHAR | nullable |
| `error_message` | TEXT | nullable |
| `error_code` | VARCHAR | nullable |
| `queued_at` | TIMESTAMP | nullable |
| `started_at` | TIMESTAMP | nullable |
| `completed_at` | TIMESTAMP | nullable |
| `created_at` | TIMESTAMP | NOT NULL, default `now()` |
| `deleted_at` | TIMESTAMP | nullable (soft-delete) |

**ENUMs (created in migration):**
- `job_status`: `queued`, `processing`, `completed`, `failed`, `cancelled`
- `modality_type`: `text`, `image`, `audio`, `video`, `multimodal`

---

## `reports`

| Column | Type | Constraints |
|---|---|---|
| `id` | UUID | PK |
| `job_id` | UUID | FK → `jobs.id`, NOT NULL |
| `report_type` | `report_type` ENUM | NOT NULL |
| `status` | `report_status` ENUM | NOT NULL, default `'pending'` |
| `minio_object_key` | VARCHAR | nullable |
| `file_size_bytes` | BIGINT | nullable |
| `file_sha256` | VARCHAR | nullable |
| `signed_url` | TEXT | nullable |
| `signed_url_expires` | TIMESTAMP | nullable |
| `generation_time_ms` | INTEGER | nullable |
| `error_message` | TEXT | nullable |
| `generated_at` | TIMESTAMP | nullable |
| `created_at` | TIMESTAMP | NOT NULL, default `now()` |

**ENUMs:**
- `report_type`: `pdf`, `json`, `html`
- `report_status`: `pending`, `generating`, `completed`, `failed`

**Unique constraint:** `(job_id, report_type)` — one report per type per job.

---

## Relationships

```
users (1) ──< jobs (many)
jobs  (1) ──< reports (many)
```
