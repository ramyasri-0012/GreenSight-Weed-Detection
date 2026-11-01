$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Error "Python environment not found. Run: python -m venv .venv; .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
}

$model = if ($env:GREENSIGHT_MODEL) { $env:GREENSIGHT_MODEL } else { "runs\weed_detector\weights\best.pt" }
if (-not (Test-Path $model)) {
    Write-Error "Trained model not found at '$model'. Train the model first with: .\.venv\Scripts\python.exe train.py --data data.yaml"
}

& ".\.venv\Scripts\python.exe" -m uvicorn api:app --host 127.0.0.1 --port 8000
