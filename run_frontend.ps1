$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Virtual environment not found. Run .\setup.ps1 first."
}

$env:API_BASE = "http://localhost:8000/api"
& ".venv\Scripts\python.exe" -m streamlit run frontend/app.py --server.port 8501