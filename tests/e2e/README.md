# End-to-End Integration Tests

These are **live integration tests** — they hit real running services. They are NOT unit tests and do NOT run in CI. You need the full Docker Compose stack running before executing any of them.

## Prerequisites

```bash
# From repo root — start all services
docker-compose up -d

# Wait for all services to be healthy (usually ~30s)
docker-compose ps
```

All services must show `Up` or `healthy` before running any phase script.

## Test Phases

| Script | Tests | Requires |
|---|---|---|
| `verify_phase_a.py` | MinIO upload + RabbitMQ publish | MinIO, RabbitMQ |
| `verify_phase_b.py` | Auth register/login + image job submit + job status GET | All services |
| `verify_phase_c.py` | Full job state machine (queued → processing → completed) | All services + stub-consumer |
| `verify_phase_d.py` | Browser-based UI test via Playwright | All services + frontend on port 3000 |

## Running

```bash
# From repo root
python tests/e2e/verify_phase_a.py
python tests/e2e/verify_phase_b.py
python tests/e2e/verify_phase_c.py

# Phase D also needs Playwright browsers installed
pip install playwright
playwright install chromium
python tests/e2e/verify_phase_d.py
```

## Notes

- Test users (`phaseb_user@test.com`, `phasec_user@test.com`, `phased_user@test.com`) are created on demand by the scripts. If they already exist in the DB, the register call returns 400 (expected — the script ignores this).
- Phase C polls for up to 20 seconds waiting for `completed`. If `stub-consumer` is not running, the job will stay in `queued` and the test will fail.
- Phase D requires the React frontend running on `localhost:3000`. It uses headless Chromium (Playwright).
- These tests use real service endpoints. There is no mock or dry-run mode.
