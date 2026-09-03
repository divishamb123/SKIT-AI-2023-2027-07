$ErrorActionPreference = "Stop"

$RootDir = (Resolve-Path "$PSScriptRoot\..\..\").Path
Set-Location $RootDir

Write-Host "Starting E2E Test Suite from $RootDir..." -ForegroundColor Cyan

# Load .env variables
Get-Content "$RootDir\.env" | Where-Object { $_ -match "^[^#]" -and $_ -match "=" } | ForEach-Object {
    $name, $value = $_ -split '=', 2
    [System.Environment]::SetEnvironmentVariable($name, $value.Trim())
}

# 1. Start Services
Write-Host "Starting docker-compose services..."
docker-compose down -v
docker-compose up -d postgres redis rabbitmq minio
Start-Sleep -Seconds 10

# 2. Start Backend APIs
Write-Host "Starting API Gateway and Auth Service..."
$env:PYTHONPATH = $RootDir
Start-Process -NoNewWindow -WorkingDirectory "$RootDir\services\auth-service" -FilePath "$RootDir\.venv\Scripts\python.exe" -ArgumentList "-m uvicorn app.main:app --host 0.0.0.0 --port 8001"
Start-Process -NoNewWindow -WorkingDirectory "$RootDir\services\api-gateway" -FilePath "$RootDir\.venv\Scripts\python.exe" -ArgumentList "-m uvicorn app.main:app --host 0.0.0.0 --port 8000"

# 3. Start Stub Consumer
Write-Host "Starting Stub Consumer..."
Start-Process -NoNewWindow -WorkingDirectory "$RootDir" -FilePath "$RootDir\.venv\Scripts\python.exe" -ArgumentList "services\stub-consumer\main.py"

# Wait for APIs
Start-Sleep -Seconds 5

# 4. Start Next.js
Write-Host "Starting Frontend..."
Set-Location "$RootDir\frontend"
Start-Process -NoNewWindow -FilePath "npm.cmd" -ArgumentList "run dev"
Set-Location $RootDir
Start-Sleep -Seconds 10

# 5. Run Verification Scripts
Write-Host "Running Verification Scripts..."
Set-Location "$RootDir\tests\e2e"
$env:PYTHONPATH = $RootDir
& "..\..\.venv\Scripts\python.exe" verify_phase_d.py

Write-Host "All tests completed!" -ForegroundColor Green

