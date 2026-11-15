# generate_report.ps1 - Generate latest experiment report, CSVs, and charts
$ErrorActionPreference = "Stop"

Set-Location "$PSScriptRoot\.."
Write-Host "Compiling report from latest experiment run..."
& ".\.venv\Scripts\python.exe" -m reward_hacking.report --latest

Write-Host "Generating publication-grade PDF report..."
& ".\.venv\Scripts\python.exe" generate_pdf.py
Write-Host "PDF report generated: C:\demo_cap\REPORT.pdf"
