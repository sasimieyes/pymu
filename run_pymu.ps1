# PowerShell Script for Running PyMu (FastAPI + PaddleOCR) - PRODUCTION
# Binds HTTPS on 9989. Production normally runs via the Windows service
# (LOCAL SYSTEM); this script is for manual / dev-on-prod-machine launches.

$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

# Locate venv interpreter
$venvPython = Join-Path $scriptPath ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "X Virtualenv not found at $venvPython." -ForegroundColor Red
    Write-Host "  Create with: python -m venv .venv ; .\.venv\Scripts\Activate.ps1 ; pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}

# Verify SSL certs exist. Certs live OUTSIDE the project at ..\ssl so they can
# be shared with sibling services and never accidentally committed.
$sslDir  = Join-Path (Split-Path -Parent $scriptPath) "ssl"
$sslKey  = Get-ChildItem -Path $sslDir -Filter "*-key.pem" -ErrorAction SilentlyContinue | Select-Object -First 1
$sslCert = Get-ChildItem -Path $sslDir -Filter "*-chain.pem" -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $sslKey -or -not $sslCert) {
    Write-Host "X SSL cert/key not found in $sslDir" -ForegroundColor Red
    Write-Host "  Place a Let's Encrypt (or other) cert pair as:" -ForegroundColor Yellow
    Write-Host "    $sslDir\<your-domain>-key.pem"   -ForegroundColor Yellow
    Write-Host "    $sslDir\<your-domain>-chain.pem" -ForegroundColor Yellow
    exit 1
}

# Free port 9989 if occupied
$port = 9989
$process = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -First 1
if ($process) {
    Write-Host "! Port $port is in use by PID $process. Terminating..." -ForegroundColor Cyan
    try {
        Stop-Process -Id $process -Force -ErrorAction Stop
        Start-Sleep -Seconds 1
    }
    catch {
        Write-Host "X Failed to terminate process $process. Re-run as Administrator if needed." -ForegroundColor Red
        # Continue; uvicorn will surface a clear bind error if still locked.
    }
}

Write-Host ">> Starting PyMu PDF service [PROD MODE] on HTTPS port $port..." -ForegroundColor Green
& $venvPython -m uvicorn backend.main:app `
    --host 0.0.0.0 `
    --port $port `
    --ssl-keyfile  $sslKey.FullName `
    --ssl-certfile $sslCert.FullName
