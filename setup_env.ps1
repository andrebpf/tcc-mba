# PowerShell script to setup the environment
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    python -m venv .venv
}

Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& .\.venv\Scripts\Activate.ps1

Write-Host "Updating pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip

Write-Host "Installing requirements..." -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host "Registering Jupyter Kernel..." -ForegroundColor Cyan
python -m ipykernel install --user --name=mba-tcc --display-name "Python (mba-tcc)"

Write-Host "Environment setup complete!" -ForegroundColor Green
