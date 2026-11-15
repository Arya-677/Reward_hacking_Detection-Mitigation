import math
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from scipy import stats


def wilson_score_interval(
    k: int, n: int, confidence: float = 0.95
) -> Tuple[float, float, float]:
    """Compute 95% Wilson score confidence interval for a binomial proportion.
    
    Returns: (point_estimate, ci_lower, ci_upper) in range [0.0, 1.0].
    """
    if n <= 0:
        return 0.0, 0.0, 0.0

    p = k / n
    # For 95% confidence, z = 1.95996
    z = stats.norm.ppf((1 + confidence) / 2.0)
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    spread = (z * math.sqrt((p * (1 - p) / n) + (z2 / (4 * n * n)))) / denom

    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)
    return round(p, 4), round(lower, 4), round(upper, 4)


def compute_contingency_and_fisher(
    k1: int, n1: int, k2: int, n2: int
) -> Tuple[List[List[int]], float]:
    """Compute 2x2 contingency table and Fisher's exact test p-value."""
    # Table: [[k1, n1 - k1], [k2, n2 - k2]]
    table = [[k1, max(0, n1 - k1)], [k2, max(0, n2 - k2)]]
    if n1 == 0 or n2 == 0:
        return table, 1.0
    _, p_value = stats.fisher_exact(table)
    return table, float(p_value)


def compute_mcnemar_test(
    matched_pairs: List[Tuple[bool, bool]]
) -> Tuple[int, int, float]:
    """Compute exact McNemar test for discordant pairs in matched blocks.
    
    matched_pairs: list of (condition_A_hacked, condition_B_hacked)
    Returns: (b, c, p_value) where b = (0, 1) and c = (1, 0) discordant counts.
    """
    b = 0  # Cond 1 No, Cond 2 Yes
    c = 0  # Cond 1 Yes, Cond 2 No

    for x1, x2 in matched_pairs:
        if not x1 and x2:
            b += 1
        elif x1 and not x2:
            c += 1

    total_discordant = b + c
    if total_discordant == 0:
        return b, c, 1.0

    # Exact binomial test for two-sided p-value
    # Null hypothesis: p = 0.5
    binom_res = stats.binomtest(b, total_discordant, 0.5, alternative="two-sided")
    return b, c, float(binom_res.pvalue)


def calculate_detector_metrics(
    episodes: List[Dict[str, Any]],
    strict_confirmed_only: bool = False,
) -> Dict[str, Any]:
    """Calculate detector performance metrics (precision, recall, F1, FPR, confusion matrix)."""
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for ep in episodes:
        pred = bool(ep.get("detector_predicted_hack"))
        label = ep.get("ground_truth_label")

        if strict_confirmed_only:
            is_positive = (label == "confirmed_hack")
        else:
            is_positive = label in ("confirmed_hack", "attempted_hack")

        if pred and is_positive:
            tp += 1
        elif pred and not is_positive:
            fp += 1
        elif not pred and not is_positive:
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    fpr = fp / (fp + tn) if (fp + tn) > 0 else None

    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = None

    return {
        "strict_mode": strict_confirmed_only,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "f1": round(f1, 4) if f1 is not None else None,
        "false_positive_rate": round(fpr, 4) if fpr is not None else None,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def analyze_experiment_results(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metrics, Wilson intervals, hypothesis tests, and system benchmarks."""
    conditions = ["target_only", "target_static_hacker", "target_adaptive_hacker"]
    by_condition: Dict[str, List[Dict[str, Any]]] = {c: [] for c in conditions}

    for ep in episodes:
        c = ep.get("condition")
        if c in by_condition:
            by_condition[c].append(ep)

    condition_metrics: Dict[str, Any] = {}

    for c in conditions:
        eps = by_condition[c]
        n = len(eps)

        confirmed_count = sum(1 for e in eps if e.get("ground_truth_label") == "confirmed_hack")
        attempted_count = sum(1 for e in eps if e.get("ground_truth_label") == "attempted_hack")
        benign_count = sum(1 for e in eps if e.get("ground_truth_label") == "benign_success")
        failure_count = sum(1 for e in eps if e.get("ground_truth_label") == "ordinary_failure")
        ambiguous_count = sum(1 for e in eps if e.get("ground_truth_label") == "ambiguous")

        total_hack_events = sum(e.get("exploit_event_count", 0) for e in eps)
        events_per_100 = (total_hack_events / n * 100.0) if n > 0 else 0.0

        p_hack, ci_low, ci_high = wilson_score_interval(confirmed_count, n)
        proxy_succ_count = sum(1 for e in eps if e.get("final_proxy_score", 0.0) == 1.0)
        true_succ_count = sum(1 for e in eps if e.get("true_success"))

        # Proxy vs True Mismatch: proxy == 1.0 but true_success is False
        mismatch_count = sum(
            1 for e in eps if (e.get("final_proxy_score", 0.0) == 1.0 and not e.get("true_success"))
        )

        mean_proxy = sum(e.get("final_proxy_score", 0.0) for e in eps) / n if n > 0 else 0.0
        avg_tokens = sum(e.get("total_tokens", 0) for e in eps) / n if n > 0 else 0.0
        avg_duration = sum(e.get("duration_seconds", 0.0) for e in eps) / n if n > 0 else 0.0
        total_tokens = sum(e.get("total_tokens", 0) for e in eps)

        condition_metrics[c] = {
            "n": n,
            "confirmed_hack_episodes": confirmed_count,
            "confirmed_hack_rate": p_hack,
            "confirmed_hack_ci_lower": ci_low,
            "confirmed_hack_ci_upper": ci_high,
            "confirmed_hack_events": total_hack_events,
            "events_per_100_episodes": round(events_per_100, 2),
            "attempted_hack_episodes": attempted_count,
            "attempted_hack_rate": round(attempted_count / n, 4) if n > 0 else 0.0,
            "benign_success_episodes": benign_count,
            "ordinary_failure_episodes": failure_count,
            "ambiguous_episodes": ambiguous_count,
            "proxy_success_count": proxy_succ_count,
            "proxy_success_rate": round(proxy_succ_count / n, 4) if n > 0 else 0.0,
            "true_success_count": true_succ_count,
            "true_success_rate": round(true_succ_count / n, 4) if n > 0 else 0.0,
            "proxy_true_mismatch_count": mismatch_count,
            "proxy_true_mismatch_rate": round(mismatch_count / n, 4) if n > 0 else 0.0,
            "mean_proxy_score": round(mean_proxy, 4),
            "avg_tokens_per_episode": round(avg_tokens, 1),
            "total_tokens": total_tokens,
            "avg_duration_seconds": round(avg_duration, 2),
        }

    # Hypothesis Tests
    n_a = condition_metrics["target_only"]["n"]
    k_a = condition_metrics["target_only"]["confirmed_hack_episodes"]

    n_b = condition_metrics["target_static_hacker"]["n"]
    k_b = condition_metrics["target_static_hacker"]["confirmed_hack_episodes"]

    n_c = condition_metrics["target_adaptive_hacker"]["n"]
    k_c = condition_metrics["target_adaptive_hacker"]["confirmed_hack_episodes"]

    _, p_a_vs_b = compute_contingency_and_fisher(k_a, n_a, k_b, n_b)
    _, p_a_vs_c = compute_contingency_and_fisher(k_a, n_a, k_c, n_c)
    _, p_b_vs_c = compute_contingency_and_fisher(k_b, n_b, k_c, n_c)

    # Paired McNemar Test on Matched Blocks
    blocks: Dict[Tuple[str, int], Dict[str, bool]] = {}
    for ep in episodes:
        key = (ep.get("task_id"), ep.get("seed"))
        if key not in blocks:
            blocks[key] = {}
        cond = ep.get("condition")
        is_hacked = (ep.get("ground_truth_label") == "confirmed_hack")
        blocks[key][cond] = is_hacked

    pairs_c_vs_a: List[Tuple[bool, bool]] = []
    pairs_c_vs_b: List[Tuple[bool, bool]] = []

    for key, c_map in blocks.items():
        if "target_only" in c_map and "target_adaptive_hacker" in c_map:
            pairs_c_vs_a.append((c_map["target_only"], c_map["target_adaptive_hacker"]))
        if "target_static_hacker" in c_map and "target_adaptive_hacker" in c_map:
            pairs_c_vs_b.append((c_map["target_static_hacker"], c_map["target_adaptive_hacker"]))

    b_ca, c_ca, mcnemar_p_c_vs_a = compute_mcnemar_test(pairs_c_vs_a)
    b_cb, c_cb, mcnemar_p_c_vs_b = compute_mcnemar_test(pairs_c_vs_b)

    # Detector evaluations
    detector_standard = calculate_detector_metrics(episodes, strict_confirmed_only=False)
    detector_strict = calculate_detector_metrics(episodes, strict_confirmed_only=True)

    return {
        "conditions": condition_metrics,
        "statistical_tests": {
            "fisher_exact": {
                "a_vs_b": {"p_value": round(p_a_vs_b, 5)},
                "a_vs_c": {"p_value": round(p_a_vs_c, 5)},
                "b_vs_c": {"p_value": round(p_b_vs_c, 5)},
            },
            "mcnemar_paired": {
                "c_vs_a": {"b": b_ca, "c": c_ca, "p_value": round(mcnemar_p_c_vs_a, 5)},
                "c_vs_b": {"b": b_cb, "c": c_cb, "p_value": round(mcnemar_p_c_vs_b, 5)},
            },
        },
        "detector": {
            "standard_positive_class": detector_standard,
            "strict_positive_class": detector_strict,
        },
    }
