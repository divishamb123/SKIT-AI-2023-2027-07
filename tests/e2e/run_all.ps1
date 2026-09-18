$ErrorActionPreference = "Stop"

$RootDir = (Resolve-Path "$PSScriptRoot\..\..\").Path
Set-Location $RootDir

Write-Host "Starting E2E Test Suite from $RootDir..." -ForegroundColor Cyan

# Load .env variables if present
if (Test-Path "$RootDir\.env") {
    Get-Content "$RootDir\.env" | Where-Object { $_ -match "^[^#]" -and $_ -match "=" } | ForEach-Object {
        $name, $value = $_ -split '=', 2
        [System.Environment]::SetEnvironmentVariable($name, $value.Trim())
    }
}

# 1. Start Services using Docker Compose
Write-Host "Starting docker-compose full stack..."
docker compose down -v
docker compose build
docker compose up -d
Start-Sleep -Seconds 15

# 2. Wait for APIs
Write-Host "Waiting for APIs to be healthy..."
Start-Sleep -Seconds 10

# 3. Run Verification Scripts
Write-Host "Running Verification Scripts..."
Set-Location "$RootDir\tests\e2e"
$env:PYTHONPATH = $RootDir
& "..\..\venv\Scripts\python.exe" verify_phase_d.py

Write-Host "All tests completed!" -ForegroundColor Green

# Teardown
Write-Host "Tearing down docker-compose stack..."
Set-Location $RootDir
docker compose down
