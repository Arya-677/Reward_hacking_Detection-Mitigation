# run_smoke_test.ps1 - Run 1 episode per condition to verify pipeline and calibrate latency
$ErrorActionPreference = "Stop"

Set-Location "$PSScriptRoot\.."
Write-Host "Running 3-episode smoke test (Target Only, Static Hacker, Adaptive Hacker)..."
& ".\.venv\Scripts\python.exe" -m reward_hacking.run --all-conditions --auto-size --time-budget-minutes 30
