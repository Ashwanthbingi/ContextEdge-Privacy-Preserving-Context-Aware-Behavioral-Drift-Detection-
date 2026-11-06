param([int]$Port = 8501)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Create the Python 3.11/3.12 virtual environment and install requirements first; see README.md.'
}
& $pythonPath -m streamlit run app.py --server.address 127.0.0.1 --server.port $Port --browser.gatherUsageStats false
