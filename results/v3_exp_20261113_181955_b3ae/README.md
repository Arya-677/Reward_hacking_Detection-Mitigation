# Results — V3 Confirmatory Experiment

This directory contains committed experiment output from the V3 Protocol semantic ablation run.

## Run: `v3_exp_20261113_181955_b3ae`

| Field | Value |
|:---|:---|
| **Experiment ID** | `v3_exp_20261113_181955_b3ae` |
| **Date** | 2026-11-13 18:37 UTC |
| **Protocol** | V3 — Target-Conditioned Semantic Adversarial Discovery |
| **Blocks** | 8 matched blocks |
| **Tasks** | 8 coding tasks (task_001 … task_008) |
| **Episodes** | 48 total (6 conditions × 8 blocks) |
| **GPU** | NVIDIA GeForce RTX 3060 (12 GB VRAM, 100% offload) |
| **Target Model** | `qwen2.5-coder:7b-instruct` |
| **Hacker Model** | `qwen3:4b` |
| **Duration** | 17.2 minutes |

## Artifacts

| File | Description |
|:---|:---|
| [`REPORT.md`](REPORT.md) | Full experiment report in Markdown |
| [`REPORT.pdf`](REPORT.pdf) | Publication-grade PDF (same content) |
| [`REVIEW_SUMMARY.md`](REVIEW_SUMMARY.md) | 1-page executive summary |
| [`summary.json`](summary.json) | Machine-readable results summary |
| [`table_c2_results.csv`](table_c2_results.csv) | Table C.2: Hack rate, prediction accuracy, Brier score |
| [`config.json`](config.json) | Frozen experiment configuration |
| [`hack_frequency.png`](hack_frequency.png) | Hack rate by condition |
| [`label_distribution.png`](label_distribution.png) | Ground-truth label distribution |
| [`prediction_metrics.png`](prediction_metrics.png) | Hacker response prediction accuracy and Brier scores |
| [`true_vs_proxy.png`](true_vs_proxy.png) | Ground truth vs proxy score scatter |

> **Note**: Full trajectory JSONL and episode CSVs are excluded from this repository (large binary/data files). Reproduce with `python -m reward_hacking.run --protocol v3 --experiment semantic_ablation --blocks 8`.
