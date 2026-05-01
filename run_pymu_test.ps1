# PowerShell Script for Running PyMu (FastAPI + PaddleOCR) - DEV / TEST
# HTTP on 8000, --reload enabled. No admin / no cert required.

$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

$venvPython = Join-Path $scriptPath ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "X Virtualenv not found at $venvPython." -ForegroundColor Red
    Write-Host "  Create with: python -m venv .venv ; .\.venv\Scripts\Activate.ps1 ; pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}

$port = 8000
$process = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -First 1
if ($process) {
    Write-Host "! Port $port is in use by PID $process. Terminating..." -ForegroundColor Cyan
    try {
        Stop-Process -Id $process -Force -ErrorAction Stop
        Start-Sleep -Seconds 1
    }
    catch { }
}

Write-Host ">> Starting PyMu PDF service [DEV MODE] on HTTP port $port..." -ForegroundColor Green
& $venvPython -m uvicorn backend.main:app --host 127.0.0.1 --port $port --reload
