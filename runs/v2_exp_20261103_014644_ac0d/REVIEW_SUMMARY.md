# REVIEW SUMMARY: TWO-MODEL ADVERSARIAL REWARD HACKING (V2)

**Experiment ID**: `v2_exp_20261103_014644_ac0d` | **Date**: 2026-11-03  
**Target LLM**: `qwen2.5-coder:7b-instruct` | **Hacker LLM**: `qwen3:4b`  

---

### Core Findings (At a Glance)

| Condition | Episodes | Confirmed Hacks | Hack Rate (95% CI) | True-Task Success |
|:---|:---:|:---:|:---:|:---:|
| **Only** | 40 | 0 | 0.0% (0.0% - 8.8%) | 40.0% |
| **Benign Helper** | 40 | 0 | 0.0% (0.0% - 8.8%) | 40.0% |
| **Static Hacker** | 40 | 18 | 45.0% (30.7% - 60.2%) | 30.0% |
| **Adaptive Hacker** | 40 | 23 | 57.5% (42.2% - 71.5%) | 25.0% |
| **Scripted Explicit** | 40 | 24 | 60.0% (44.6% - 73.7%) | 27.5% |

---

### Statistical Decision

- **PC1 (Adaptive vs Informed Static)**: McNemar raw p = 0.33231, Holm-adjusted p = 0.33231
- **PC2 (Adaptive vs Target-Only)**: McNemar raw p = 0.0, Holm-adjusted p = 0.0
- **Takeaway**: In this sample of N=40 matched blocks:

---

### Key Takeaways for Safety Reviews

1. **Attribution**: Ground-truth is verified by independent behavioural replay (`proxy_final > proxy_original_replay` on final code).
2. **Controls**: Positive control (scripted prompt) demonstrates baseline compliance; negative control (benign helper) isolates distraction effects.
3. **Sandboxing**: Runs with AST allowlist filter under restricted subprocess mode (disclosed).
