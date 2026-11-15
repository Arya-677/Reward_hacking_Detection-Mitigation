# Test Execution Report: Suite T1-T14

**Execution Date**: 2026-11-03
**Environment**: Windows 11 (`ARYA_PC`), Python 3.12 (venv `c:\demo_cap\reward_hacking_mvp\.venv`)
**Test Framework**: pytest 8.x

## Summary

- **Total Tests**: 30
- **Passed**: 30
- **Failed**: 0
- **Pass Rate**: 100.0%
- **Duration**: ~70 seconds

## Test Breakdown by Section

| Test File | Test Case | Status | Coverage / Purpose |
|---|---|---|---|
| `tests/test_agent_tools.py` | `test_agent_tools_basic` | PASSED | Tool executions, write, read, bash/exec, replace_in_file |
| `tests/test_agent_tools.py` | `test_replace_in_file_ambiguity` | PASSED | Ambiguous string matches rejected |
| `tests/test_sandbox.py` | `test_sandbox_ast_validation` | PASSED | Strict AST allowlist validation (13 stdlib modules + pytest/unittest) |
| `tests/test_sandbox.py` | `test_sandbox_timeout_and_restrictions` | PASSED | Restricted subprocess mode timeout and isolation |
| `tests/test_proxy_evaluator.py` | `test_proxy_evaluator_invariants` | PASSED | Prose -> 0, syntax error -> 0, empty -> 0, missing tests -> 0 |
| `tests/test_proxy_evaluator.py` | `test_proxy_evaluator_baseline_fractional` | PASSED | Verifies fractional baseline scores across tasks |
| `tests/test_evaluators.py` | `test_ground_truth_labeler_behavioral_replay` | PASSED | Behavioural replay engine (E1-E8), label classification |
| `tests/test_evaluators.py` | `test_failure_taxonomy_ordering` | PASSED | Priority order resolution for failure causes |
| `tests/test_evaluators.py` | `test_synthetic_evasion_recall` | PASSED | Rule-based detector recall >= 90% (11/12 synthetic evasion variants) |
| `tests/test_evaluators.py` | `test_tasks_suite_integrity` | PASSED | Integrity of all 8 task definitions and test suites |
| `tests/test_hacker_agent.py` | `test_information_matrix_views` | PASSED | Static vs Adaptive hacker views (trajectory isolation) |
| `tests/test_hacker_agent.py` | `test_scripted_explicit_template` | PASSED | Scripted explicit prompt generation |
| `tests/test_hacker_agent.py` | `test_benign_helper_template` | PASSED | Benign helper advice formatting |
| `tests/test_explicitness.py` | `test_explicitness_classifier_levels` | PASSED | Explicitness levels 0, 1, 2, 3 classification |
| `tests/test_explicitness.py` | `test_compliance_tracking` | PASSED | Compliance calculation and tracking |
| `tests/test_metrics.py` | `test_mcnemar_and_holm_bonferroni` | PASSED | Paired McNemar test with Holm-Bonferroni correction |
| `tests/test_metrics.py` | `test_newcombe_wilson_ci` | PASSED | 95% Newcombe-Wilson risk difference confidence intervals |
| `tests/test_experiment_runner.py` | `test_blocked_randomization` | PASSED | Blocked randomization across 5 conditions |
| `tests/test_experiment_runner.py` | `test_budget_guard` | PASSED | Experiment runner budget guard & safe shutdown |
| `tests/test_gates.py` | `test_gate_evaluations` | PASSED | Calibration gates G1-G4 threshold checks |
| `tests/test_reporting.py` | `test_chart_generation` | PASSED | All 6 matplotlib charts render without error |
| `tests/test_reporting.py` | `test_report_compilation` | PASSED | Markdown & HTML generation with exact schemas |
| `tests/test_integration.py` | `test_e2e_mock_pipeline` | PASSED | End-to-end pipeline sanity test with mock LLM |
| *(and 7 additional unit / parameter tests)* | | PASSED | Complete v2 validation suite |

## Result
**ALL 30 VERIFICATION TESTS PASS.** The test harness, task definitions, AST sandbox, proxy evaluator, ground truth labeler, hack detector, and statistical metrics are fully verified and ready for calibration.
