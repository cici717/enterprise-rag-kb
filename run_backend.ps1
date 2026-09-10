$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Virtual environment not found. Run .\setup.ps1 first."
}

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
}

& ".venv\Scripts\python.exe" -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000