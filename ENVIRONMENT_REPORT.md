# Environment and Hardware Inspection Report

Recorded Date: 2026-11-02

## 1. System Overview
- **OS**: Windows 11 Home / Pro (Windows 64-bit, Hostname: ARYA_PC)
- **CPU**: 11th Gen Intel(R) Core(TM) i5-11400F @ 2.60GHz (6 physical cores, 12 logical processors)
- **RAM**: 17,009,442,816 bytes (~16.0 GB Total Physical Memory)
- **Disk Free Space (C:)**: ~352 GB free of 475 GB (~146 GB used)

## 2. Accelerator / GPU
- **GPU**: NVIDIA GeForce RTX 3060
- **Total VRAM**: 12,288 MiB (12.0 GB)
- **Driver Version**: 610.88
- **CUDA Version**: 13.3
- **Baseline VRAM Usage**: ~1,022 MiB used at idle / desktop compositing (~11.2 GB available)

## 3. Ollama & LLM Runtime
- **Ollama Status**: Installed & Running (Version 0.40.13, API responsive at http://localhost:11434)
- **Pre-existing Models**: `qwen2.5-coder:3b-instruct` (1.9 GB)
- **Pull in Progress**: `qwen3:4b` (~2.5 GB)
- **Target / Hacker Presets**:
  - Primary Preset: Target = `qwen2.5-coder:7b-instruct` or `qwen2.5-coder:3b-instruct`, Hacker = `qwen3:4b`
  - Fallback Preset A: Target = `qwen2.5-coder:3b-instruct`, Hacker = `qwen3:1.7b` or `qwen3:4b` depending on VRAM headroom.

## 4. Container / Virtualization Runtime
- **Docker**: Not installed / daemon not found.
- **Podman**: Not found.
- **WSL Status**: WSL not installed (`wsl --status` reports "The Windows Subsystem for Linux is not installed").
- **Sandbox Strategy**: In accordance with Master Spec Section 4.2, `SANDBOX_MODE=restricted_subprocess` is enabled as the sandboxing mechanism on Windows with AST import checking, isolated temp execution directories, environment stripping (`python -I`), and execution timeouts. Prominent disclosure banners will be displayed on every execution and documented in all reports.

## 5. Python Environment
- **Python Host**: Python 3.12.10 (`C:\Users\aryad\AppData\Local\Programs\Python\Python312\python.exe`)
- **Pip Version**: 25.0.1
- **Virtualenv**: Created at `c:\demo_cap\reward_hacking_mvp\.venv`
