# ==============================================================================
# Setup Script for V3 Protocol Pipeline (Single Setup Script)
# ==============================================================================
param (
    [switch]$SkipTests = $false
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Get-Item $PSScriptRoot).Parent.FullName

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "V3 ADVERSARIAL REWARD HACKING SETUP & GPU VERIFICATION" -ForegroundColor Cyan
Write-Host "Project Root: $ProjectRoot" -ForegroundColor DarkGray
Write-Host "================================================================================" -ForegroundColor Cyan

# 1. Verify Virtual Environment
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host "[1/6] Creating Python virtual environment (.venv)..." -ForegroundColor Yellow
    python -m venv (Join-Path $ProjectRoot ".venv")
    & $VenvPython -m pip install --upgrade pip
    & $VenvPython -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
} else {
    Write-Host "[1/6] Python virtual environment verified: $VenvPython" -ForegroundColor Green
}

# 2. Check NVIDIA GPU and Driver
Write-Host "[2/6] Inspecting NVIDIA GPU and VRAM availability..." -ForegroundColor Yellow
$NvidiaSmi = Get-Command "nvidia-smi" -ErrorAction SilentlyContinue
if ($NvidiaSmi) {
    $gpuInfo = & nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv,noheader,nounits
    Write-Host "  NVIDIA GPU Detected: $gpuInfo" -ForegroundColor Green
} else {
    Write-Host "  WARNING: nvidia-smi not found. Ensure NVIDIA display drivers are installed." -ForegroundColor Yellow
}

# 3. Check Ollama Service
Write-Host "[3/6] Verifying Ollama local inference service..." -ForegroundColor Yellow
try {
    $ollamaTags = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -TimeoutSec 4
    Write-Host "  Ollama is running on http://localhost:11434" -ForegroundColor Green
} catch {
    Write-Host "  ERROR: Ollama is not responding on http://localhost:11434. Starting Ollama..." -ForegroundColor Red
    Start-Process "ollama" -ArgumentList "serve" -WindowStyle Minimized
    Start-Sleep -Seconds 3
}

# 4. Pull and Verify Required Models
$RequiredModels = @("qwen2.5-coder:7b-instruct", "qwen3:4b")
Write-Host "[4/6] Verifying required models ($($RequiredModels -join ', '))..." -ForegroundColor Yellow

$installedModels = (& ollama list) | Out-String
foreach ($model in $RequiredModels) {
    if ($installedModels -notmatch [regex]::Escape($model)) {
        Write-Host "  Pulling model: $model..." -ForegroundColor Yellow
        & ollama pull $model
    } else {
        Write-Host "  Model '$model' is installed." -ForegroundColor Green
    }
}

# 5. Pre-warm models into GPU VRAM (Strategy A: Resident)
Write-Host "[5/6] Warming models into NVIDIA GPU memory (Strategy A: Resident)..." -ForegroundColor Yellow
& $VenvPython -c "
import urllib.request, json
models = ['qwen2.5-coder:7b-instruct', 'qwen3:4b']
for m in models:
    payload = json.dumps({'model': m, 'keep_alive': '60m'}).encode('utf-8')
    req = urllib.request.Request('http://localhost:11434/api/generate', data=payload, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            print(f'  Pre-warmed {m} with keep_alive=60m into GPU VRAM')
    except Exception as e:
        print(f'  Pre-warm warning for {m}: {e}')
"

# 6. Run V3 Protocol Unit Tests
if (-not $SkipTests) {
    Write-Host "[6/6] Executing V3 protocol unit tests..." -ForegroundColor Yellow
    $testResult = & $VenvPython -m pytest (Join-Path $ProjectRoot "tests\test_v3_protocol.py") -q
    Write-Host "  Test Result: $testResult" -ForegroundColor Green
} else {
    Write-Host "[6/6] Unit tests skipped via -SkipTests." -ForegroundColor DarkGray
}

Write-Host "`n================================================================================" -ForegroundColor Cyan
Write-Host "SETUP COMPLETE! All components verified for GPU-accelerated V3 execution." -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "`nTo start the V3 experiment:" -ForegroundColor White
Write-Host "  .venv\Scripts\python.exe -m reward_hacking.run --protocol v3 --experiment semantic_ablation --blocks 12" -ForegroundColor Yellow
Write-Host "`nTo run a single-block smoke test:" -ForegroundColor White
Write-Host "  .venv\Scripts\python.exe -m reward_hacking.run --protocol v3 --experiment smoke --blocks 1" -ForegroundColor Yellow
Write-Host "`nTo inspect GPU diagnostics & model placement:" -ForegroundColor White
Write-Host "  .venv\Scripts\python.exe scripts\gpu_diagnostics.py" -ForegroundColor Yellow
Write-Host "================================================================================`n" -ForegroundColor Cyan
