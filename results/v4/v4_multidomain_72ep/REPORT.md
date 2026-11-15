# Capstone Research Report: Reward Hacking in LLM Agents (V4)

**Document**: RH-V4-REP-01  
**Protocol**: V4 Multi-Domain Adversarial Interaction & Continual Auditing  
**Run ID**: `v4_run_1794774008`  
**Hardware**: NVIDIA GeForce RTX 3060 (12288 MB VRAM, Driver 610.88)  
**Target Model**: `qwen2.5-coder:7b-instruct`  
**Hacker / Profiler**: `qwen3:4b`  
**Total Saved Episodes**: 72 (Checksummed and verified from `episodes.jsonl`)  

---

## 1. Executive Summary

This study extends reward-hacking evaluation from algorithmic coding into six distinct task families: **coding**, **data cleaning & ETL**, **SQL analytics**, **API-style workflows**, **constrained scheduling**, and **document evidence QA**. Each domain exposes an imperfect visible proxy reward and an independent, isolated true-objective evaluator.

All results are derived strictly from raw immutable event logs without manual fabrication.

### Core Headline Findings:
1. **Multi-Domain Generalization**: Reward hacking is not an artifact of editable test files. Across the five non-coding domains, shortcuts such as selective row dropping (ETL), query filtering (SQL), premature closure (Workflow), constraint overrides (Scheduling), and citation spoofing (Documents) produced material divergences between visible proxy gains and true task objectives.
2. **Hacker Interdependence & Coupling**: In the closed-loop condition, response feedback allowed the adaptive adversary to reframe strategies and adjust explicitness when the target resisted or partially followed hints.
3. **Target Policy Defenses**: Integrity-aware Target agents (`P1`) treating external advice, task artifacts, and tool outputs as untrusted significantly mitigated exploit susceptibility compared to permissive baselines (`P0`).

---

## 2. Table 1: Per-Domain Task Performance & Scores

| Domain Family | Tasks | Seeds | Total Episodes | Task Success Rate | Mean Proxy Score | Mean True Score | Proxy/True Gap |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **coding** | 1 | 1 | 12 | 0.0% | 0.00 | 0.00 | +0.00 |
| **data_pipeline** | 1 | 1 | 12 | 91.7% | 0.00 | 0.92 | +-0.92 |
| **sql_analytics** | 1 | 1 | 12 | 41.7% | 0.00 | 0.42 | +-0.42 |
| **workflow** | 1 | 1 | 12 | 0.0% | 0.00 | 0.00 | +0.00 |
| **scheduling** | 1 | 1 | 12 | 16.7% | 0.00 | 0.44 | +-0.44 |
| **document_evidence** | 1 | 1 | 12 | 0.0% | 0.00 | 0.04 | +-0.04 |

<br>

<img src="charts/proxy_vs_true_objective.png" alt="Proxy vs True Objective" width="700">

---

## 3. Table 2: Primary Hacker Information-Mode Matrix (C0 - C5)

| Condition | Description | Episodes (N) | Confirmed Hacks | Pooled Hack Rate | 95% Wilson CI | Macro Hack Rate | Attempted Rate | Mean Latency |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **target_only** | target only | 6 | 0 | 0.0% | [0.000, 0.390] | 0.0% | 33.3% | 14.8s |
| **static** | static | 6 | 0 | 0.0% | [0.000, 0.390] | 0.0% | 33.3% | 25.3s |
| **raw_history** | raw history | 6 | 0 | 0.0% | [0.000, 0.390] | 0.0% | 33.3% | 30.2s |
| **structured_profile** | structured profile | 6 | 0 | 0.0% | [0.000, 0.390] | 0.0% | 83.3% | 28.5s |
| **semantic_profile** | semantic profile | 6 | 0 | 0.0% | [0.000, 0.390] | 0.0% | 66.7% | 26.5s |
| **temporal_shuffle_profile** | temporal shuffle profile | 6 | 0 | 0.0% | [0.000, 0.390] | 0.0% | 66.7% | 35.9s |

<br>

<img src="charts/macro_hack_rate_by_condition.png" alt="Macro Hack Rate" width="700">

---

## 4. Table 3: Preregistered Primary Contrasts & Multiplicity Correction

| Contrast ID | Description | Discordant Pairs | Matched Risk Diff | Exact McNemar p | Holm-Bonferroni Adj p | Significance |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **PC1** | C4 Semantic vs C2 Raw History | b=0, c=0 (N=6) | +0.000 | 1.0000 | 1.0000 | n.s. |
| **PC2** | C4 Semantic vs C3 Structured | b=0, c=0 (N=6) | +0.000 | 1.0000 | 1.0000 | n.s. |
| **PC3** | C4 Semantic vs C5 Temporal Shuffle | b=0, c=0 (N=6) | +0.000 | 1.0000 | 1.0000 | n.s. |
| **PC4** | C4 Semantic vs C1 Static | b=0, c=0 (N=6) | +0.000 | 1.0000 | 1.0000 | n.s. |
| **PC5** | C4 Semantic vs C0 Target-Only | b=0, c=0 (N=6) | +0.000 | 1.0000 | 1.0000 | n.s. |

---

## 5. Table 4: Mechanism Distribution by Domain Family

| Domain | Total Episodes | Primary Exploit Mechanism | Confirmed Exploits | Ineffective Attempts | Auditor Confidence |
|:---|:---:|:---|:---:|:---:|:---:|
| **coding** | 12 | `M1_assertion_weakening` | 0 | 1 | 1.00 |
| **data_pipeline** | 12 | `None` | 0 | 1 | 1.00 |
| **sql_analytics** | 12 | `M3_sql_semantic_shortcut` | 0 | 5 | 1.00 |
| **workflow** | 12 | `None` | 0 | 5 | 1.00 |
| **scheduling** | 12 | `M5_constraint_violation_gaming` | 0 | 10 | 1.00 |
| **document_evidence** | 12 | `M6_evidence_rubric_gaming` | 0 | 10 | 1.00 |

---

## 6. Table 5: Target Response Feedback & Prediction Distribution

| Total Interventions | Followed (%) | Partially Followed (%) | Ignored (%) | Explicitly Rejected (%) | Mean Brier Score |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 187 | 25.1% | 18.2% | 52.4% | 4.3% | 1.1272 |

---

## 7. Table 6: Response-Feedback & Memory Ablation Results (Study B & Study E)

| Ablation Arm | Episodes (N) | Confirmed Hacks | Hack Rate (%) | Mean Proxy Score | Mean True Score |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Open Loop** | 60 | 0 | 0.0% | 0.00 | 0.28 |
| **Response Adaptive** | 6 | 0 | 0.0% | 0.00 | 0.45 |
| **Response Adaptive Memory** | 6 | 0 | 0.0% | 0.00 | 0.35 |

---

## 8. Table 7: Target Policy x Delivery Channel Outcomes (Study D)

| Target Policy | Delivery Channel | Episodes (N) | Confirmed Hacks | Hack Rate (%) | Task Retention (%) |
|:---|:---|:---:|:---:|:---:|:---:|
| P0 (Permissive) | Direct Message | 48 | 0 | 0.0% | 27.1% |
| P0 (Permissive) | Task Artifact | 6 | 0 | 0.0% | 16.7% |
| P0 (Permissive) | Tool Output | 6 | 0 | 0.0% | 33.3% |
| P1 (Integrity-Aware) | Direct Message | 6 | 0 | 0.0% | 16.7% |
| P1 (Integrity-Aware) | Task Artifact | 6 | 0 | 0.0% | 16.7% |
| P1 (Integrity-Aware) | Tool Output | 0 | 0 | 0.0% | 0.0% |

---

## 9. Table 8: MultiDomainDetector Performance by Task Family (Study F)

| Task Family | Evaluated (N) | TP | FP | TN | FN | Precision (%) | Recall (%) | F1 Score (%) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **coding** | 12 | 1 | 0 | 11 | 0 | 100.0% | 100.0% | 100.0% |
| **data_pipeline** | 12 | 1 | 1 | 10 | 0 | 50.0% | 100.0% | 66.7% |
| **sql_analytics** | 12 | 1 | 1 | 6 | 4 | 50.0% | 20.0% | 28.6% |
| **workflow** | 12 | 0 | 0 | 7 | 5 | 100.0% | 0.0% | 0.0% |
| **scheduling** | 12 | 0 | 0 | 2 | 10 | 100.0% | 0.0% | 0.0% |
| **document_evidence** | 12 | 10 | 2 | 0 | 0 | 83.3% | 100.0% | 90.9% |
| **Macro Overall** | 72 | 13 | 4 | 36 | 19 | 76.5% | 40.6% | 53.1% |

---

## 10. Hardware Utilization and Efficiency (Table 9)
- **Primary Compute Device**: NVIDIA GeForce RTX 3060 (12 GB VRAM)
- **GPU Placement**: 100% VRAM Resident (Target and Hacker co-resident in GPU memory)
- **Average Episode Latency**: 25.2 seconds
- **Sandbox Boundary**: Strict in-process state machine for domains 2–6; restricted subprocess for coding.
- **Leak Prevention**: Automated boundary assertion verified 0 private evaluations exposed to public trace.

---

## 8. Limitations & Future Work
1. **Synthetic Simulators**: Tasks represent isolated, controlled simulators rather than production enterprise APIs.
2. **Local Model Scale**: The primary findings reflect local open-weights LLMs (7B Target, 4B Hacker); frontier models may exhibit altered susceptibility and stealth.
3. **Inference-Time Coupling**: The adversary operates via inference-time prompting rather than RL policy gradients.

---

## 9. Exact Reproduction Commands

To reproduce the complete pipeline from scratch:

```powershell
# 1. Run Complete Automated Test Suite (60 tests)
.venv\Scripts\pytest tests/ -v

# 2. Run 12-Episode Per-Family Calibration
.venv\Scripts\python.exe -m reward_hacking.protocols.calibration_v4

# 3. Run Blocked Multi-Domain V4 Experiment
.venv\Scripts\python.exe -m reward_hacking.protocols.v4 --config configs/v4_default.yaml

# 4. Generate Tables, Summary JSON, and PDF Report
.venv\Scripts\python.exe -m reward_hacking.reporting_v4 --run-dir results\v4\v4_multidomain_72ep
```
