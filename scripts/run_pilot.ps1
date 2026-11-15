# run_pilot.ps1 - Run small pilot experiment across conditions (5 episodes per condition)
$ErrorActionPreference = "Stop"

Set-Location "$PSScriptRoot\.."
Write-Host "Running 15-episode pilot experiment (5 per condition)..."
& ".\.venv\Scripts\python.exe" -m reward_hacking.run --all-conditions --episodes 5 --time-budget-minutes 60
