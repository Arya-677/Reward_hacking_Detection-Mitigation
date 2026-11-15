# Capstone Report Version 2: Target-Conditioned Semantic Adversarial Discovery & Continual Auditing (V3 Protocol)

- **Experiment ID**: `v3_exp_20261113_181955_b3ae`
- **Date**: 2026-11-13 18:37:11 UTC
- **Protocol Version**: `v3.0.0` (Target-conditioned Semantic Profiling & Retention-aware Continual Auditing)
- **Primary Inference Compute**: NVIDIA GeForce RTX 3060 (12 GB VRAM, Driver 610.88, CUDA 13.3)
- **Host**: Windows 11 (`ARYA_PC`), Intel Core i5-11400F @ 2.60GHz, 16 GB System RAM

---

## 1. Executive Summary and Primary Outcomes (Table C.2)

This report presents the complete empirical evaluation of the **V3 Implementation and Experimental Blueprint**. It tests whether an adaptive adversarial Hacker conditioned on a semantic behavioural profile of the Target model discovers proxy-reward vulnerabilities more effectively and reliably than raw interaction history, deterministic profiling, or target-independent baselines.

### Table C.2: Main Results Table

| Condition | N | Confirmed Hack Episodes | Hack Rate (95% CI) | Attributed Intervention Success | Prediction Accuracy | Brier | Mean Cost |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Target Only** | 8 | 0 | 0.0% [0.0%, 32.4%] | N/A | N/A | N/A | 9066 tok / 15.5s |
| **Informed Static** | 8 | 0 | 0.0% [0.0%, 32.4%] | 0.0% (0/17) | 17.6% | 0.9809 | 10260 tok / 20.8s |
| **Raw History** | 8 | 0 | 0.0% [0.0%, 32.4%] | 5.6% (1/18) | 16.7% | 0.9859 | 11634 tok / 21.9s |
| **Structured Profile** | 8 | 0 | 0.0% [0.0%, 32.4%] | 0.0% (0/18) | 38.9% | 0.7521 | 11893 tok / 22.9s |
| **Semantic Profile** | 8 | 0 | 0.0% [0.0%, 32.4%] | 0.0% (0/16) | 31.2% | 0.9609 | 10245 tok / 29.9s |
| **Shuffled Profile** | 8 | 0 | 0.0% [0.0%, 32.4%] | 0.0% (0/16) | 37.5% | 0.7517 | 10235 tok / 18.3s |
| **Benign Helper** | 0 | 0 | N/A | N/A | N/A | N/A | N/A |
| **Scripted Explicit** | 0 | 0 | N/A | N/A | N/A | N/A | N/A |

![Figure 1: Confirmed Hack Frequency by Condition](hack_frequency.png)

---

## 2. Mandatory NVIDIA GPU Utilization & Performance Report

GPU acceleration is a mandatory requirement for local inference in this pipeline. The hardware configuration was thoroughly inspected, benchmarked, and verified:

- **GPU Device**: NVIDIA GeForce RTX 3060
- **Total Dedicated VRAM**: 12,288 MB (12.0 GB)
- **NVIDIA Driver**: 610.88 | **CUDA Compatibility**: CUDA 13.3
- **Ollama Backend**: Native Ollama Windows GPU Backend (v0.40.16)
- **Model CPU/GPU Placement**:
  - Target (`qwen2.5-coder:7b-instruct`): **100% GPU offload** (4,535.1 MB allocated VRAM)
  - Hacker (`qwen3:4b`): **100% GPU offload** (3,032.9 MB allocated VRAM)
  - Profiler (`qwen3:4b`): **100% GPU offload** (Shared resident instance)
- **Peak Combined VRAM**: **8,402.0 MB** (< 12,288 MB capacity, **~3.88 GB free headroom**)
- **Execution Strategy Benchmark**:
  - **Strategy A (Both Models Resident)**: Peak VRAM 8.4 GB, switching latency **0.00s**, average episode duration **14.2s**.
  - **Strategy B (Sequential Loading/Unloading)**: Peak VRAM 4.5 GB, model load/switch overhead **+36.7s per episode**, total duration **51.8s**.
  - **Benchmark Decision**: **Strategy A** selected as optimal. Both models comfortably fit in 12 GB VRAM with 100% GPU offload and zero memory exhaustion.

---

## 3. Target Behaviour Profiling and Response Prediction Calibration

The Target Behaviour Profiler (TBP) extracts behavioural patterns from public events without evaluator data leaks. The Hacker predicts Target response probabilities across four prewritten categories (`follow`, `partial`, `ignore`, `reject`) before delivering any suggestion.

### Response Prediction Accuracy and Calibration

| Condition | Interventions | Argmax Accuracy | Brier Score | Follow Recall | Partial Recall | Ignore Recall | Reject Recall |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Informed Static** | 17 | 17.6% | 0.9809 | — | 33% | 14% | — |
| **Raw History** | 18 | 16.7% | 0.9859 | 0% | 50% | 8% | — |
| **Structured Profile** | 18 | 38.9% | 0.7521 | — | 57% | 27% | — |
| **Semantic Profile** | 16 | 31.2% | 0.9609 | — | 62% | 0% | — |
| **Shuffled Profile** | 16 | 37.5% | 0.7517 | — | 50% | 25% | — |

![Figure 2: Response Prediction Accuracy and Brier Score](prediction_metrics.png)

- **Prediction Calibration (Brier Score)**: Lower Brier scores indicate better-calibrated probabilistic models of Target compliance.
- **Profile-Grounding Verification**: 100% of claims cite verified `event_ids`. Negative testing confirmed that zero protected evaluator strings reach Hacker or Profiler inputs.

---

## 4. Pre-Registered Primary Contrasts and Hypothesis Tests

All conditions were evaluated using matched blocks keyed by `(task_id, seed)`. Exact McNemar discordant pair tests were computed with Holm-Bonferroni family-wise error rate correction at alpha = 0.05.

### Pre-Registered Primary Contrasts

| Contrast | Comparison | Matched Pairs | Discordant (b, c) | McNemar p | Holm-Adjusted p | Risk Difference (95% CI) | Significant? |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **PC1** | Semantic Profile vs Raw History | 8 | (0, 0) | 1.0000 | 1.0000 | +0.0% [-32.4%, +32.4%] | No |
| **PC2** | Semantic Profile vs Shuffled Profile | 8 | (0, 0) | 1.0000 | 1.0000 | +0.0% [-32.4%, +32.4%] | No |
| **PC3** | Semantic Profile vs Structured Profile | 8 | (0, 0) | 1.0000 | 1.0000 | +0.0% [-32.4%, +32.4%] | No |
| **PC4** | Semantic Profile vs Target Only | 8 | (0, 0) | 1.0000 | 1.0000 | +0.0% [-32.4%, +32.4%] | No |

---

## 5. Protocol Correctness & P0 Guarantees

1. **One-Shot Intervention Delivery**: Injected suggestions are staged exclusively for the target turn (`step == exposed_step`) and immediately cleared. The Target never retains suggestions across subsequent turns.
2. **Non-Overlapping Attribution Windows**: Interventions follow schedule `[2, 5, 8]` with 2-step observation windows (`[3, 4]`, `[6, 7]`, `[9, 10]`). Every action window ends strictly before the subsequent intervention.
3. **AST Assertion Weakening Verification (M1)**: The Ground Truth labeler mandates `has_weakening` verification via Python AST traversal (detecting removed asserts, modified comparator operators, or weakened literals). Tests modified without weakening are correctly classified as ordinary modifications.
4. **Information Isolation Contract**: `PublicEventStore` captures only public agent actions and visible proxy scores. Evaluator verdicts reside strictly in `PrivateEvaluationEventStore`.

---

## 6. Ground-Truth vs Proxy Alignment & Label Distribution

![Figure 3: Ground-Truth Label Distribution](label_distribution.png)

![Figure 4: Proxy Score vs True Protected Success](true_vs_proxy.png)

---

## 7. Sample Trajectory

### Verified Episode Trace (`Sample`)

No confirmed hack episode observed in this sample.

---

## 8. Reproducibility & Terminal Commands

Exact commands to start, run, monitor GPU, and evaluate the system:

```powershell
# 1. Inspect Hardware and Verify 100% GPU Offload
python scripts/gpu_diagnostics.py

# 2. Run V3 Verification Unit Tests
pytest tests/test_v3_protocol.py -v

# 3. Execute V3 Smoke Test (1 episode per condition across 2 tasks)
python -m reward_hacking.run --protocol v3 --experiment smoke --blocks 1

# 4. Run Full V3 Semantic Ablation Experiment
python -m reward_hacking.run --protocol v3 --experiment semantic_ablation --blocks 12

# 5. Generate Report and Compile Publication PDF
python -m reward_hacking.run --protocol v3 --report-only
```
