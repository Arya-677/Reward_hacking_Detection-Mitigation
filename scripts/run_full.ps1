# run_full.ps1 - Run full blocked experiment under time-budget guard
$ErrorActionPreference = "Stop"

Set-Location "$PSScriptRoot\.."
Write-Host "Running full experiment with auto-sizing and 150-minute budget guard..."
& ".\.venv\Scripts\python.exe" -m reward_hacking.run --all-conditions --auto-size --time-budget-minutes 150
