#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Starting E2E Test Suite from $ROOT_DIR..."

cd "$ROOT_DIR"

# 1. Start Services using Docker Compose
echo "Starting docker-compose full stack..."
docker compose down -v || true
docker compose build
docker compose up -d

echo "Waiting for services to become healthy..."
sleep 15

# 2. Run Verification Scripts
echo "Running E2E Phase D Verification..."
export PYTHONPATH="$ROOT_DIR"
python3 "$SCRIPT_DIR/verify_phase_d.py"

echo "All E2E tests completed successfully!"

# 3. Teardown
echo "Tearing down docker-compose stack..."
docker compose down
