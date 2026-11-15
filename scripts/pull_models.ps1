# pull_models.ps1 - Pull required local models into Ollama
$ErrorActionPreference = "Continue"

Write-Host "Verifying Ollama service..."
try {
    Invoke-RestMethod http://localhost:11434/api/tags -TimeoutSec 3 | Out-Null
    Write-Host "Ollama is reachable."
} catch {
    Write-Host "Starting Ollama serve in background..."
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 3
}

Write-Host "Pulling Target Model (qwen2.5-coder:3b-instruct)..."
ollama pull qwen2.5-coder:3b-instruct

Write-Host "Pulling Hacker Model (qwen3:4b)..."
ollama pull qwen3:4b

Write-Host "Models ready:"
ollama list
