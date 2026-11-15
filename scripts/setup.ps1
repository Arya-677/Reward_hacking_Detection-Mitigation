# setup.ps1 - Environment setup and dependency installation
$ErrorActionPreference = "Stop"

Write-Host "Setting up Python virtual environment..."
if (!(Test-Path ".venv")) {
    python -m venv .venv
}

Write-Host "Activating virtual environment and installing dependencies..."
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\pip.exe" install -r requirements.txt
& ".\.venv\Scripts\pip.exe" install -e .

Write-Host "Verifying setup..."
& ".\.venv\Scripts\python.exe" -m reward_hacking.check_setup
Write-Host "Setup complete."
