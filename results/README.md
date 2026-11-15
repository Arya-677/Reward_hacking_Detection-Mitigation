# Results Directory

This directory contains committed experimental results, charts, and reports from completed V3 protocol runs.

Each subdirectory corresponds to one experiment run and contains:
- `REPORT.md` / `REPORT.pdf` — Full publication-grade experiment report
- `REVIEW_SUMMARY.md` — 1-page executive summary
- `table_c2_results.csv` — Main results table (Table C.2)
- `summary.json` — Machine-readable results
- `config.json` — Frozen run configuration
- `*.png` — Charts (hack frequency, label distribution, prediction metrics, proxy vs true)

## Runs

| Run ID | Protocol | Blocks | Date | Key Finding |
|:---|:---|:---:|:---|:---|
| [`v3_exp_20261113_181955_b3ae`](v3_exp_20261113_181955_b3ae/) | V3 Semantic Ablation | 8 | 2026-11-13 | 0 confirmed hacks; structured profile Brier 0.75 |
