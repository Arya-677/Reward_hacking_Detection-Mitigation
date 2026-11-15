import pytest
from reward_hacking.metrics import (
    analyze_experiment_results,
    calculate_detector_metrics,
    compute_contingency_and_fisher,
    wilson_score_interval,
)


def test_wilson_confidence_interval():
    """Requirement I: Test Wilson 95% score interval on known sample points."""
    # 0 out of 10 -> lower bound 0.0, upper bound > 0.0
    p, low, high = wilson_score_interval(0, 10)
    assert p == 0.0
    assert low == 0.0
    assert 0.0 < high < 0.35

    # 10 out of 10 -> upper bound 1.0
    p, low, high = wilson_score_interval(10, 10)
    assert p == 1.0
    assert 0.65 < low < 1.0
    assert high == 1.0

    # 5 out of 10
    p, low, high = wilson_score_interval(5, 10)
    assert p == 0.5
    assert low < 0.5 < high


def test_fisher_exact_test():
    """Requirement I: Contingency table and Fisher test calculation."""
    table, p_val = compute_contingency_and_fisher(1, 10, 8, 10)
    assert table == [[1, 9], [8, 2]]
    assert p_val < 0.05  # Strongly different proportions


def test_detector_zero_division():
    """Requirement I: Handling zero positives/predictions returns None, not division error."""
    empty_episodes = [
        {"detector_predicted_hack": False, "ground_truth_label": "ordinary_failure"},
        {"detector_predicted_hack": False, "ground_truth_label": "benign_success"},
    ]
    res = calculate_detector_metrics(empty_episodes, strict_confirmed_only=True)
    assert res["true_positives"] == 0
    assert res["precision"] is None
    assert res["recall"] is None
    assert res["f1"] is None


def test_episode_rate_vs_event_count():
    """Requirement I: Episode rate vs event count must be distinct when episodes have multiple events."""
    episodes = [
        {
            "condition": "target_adaptive_hacker",
            "ground_truth_label": "confirmed_hack",
            "exploit_event_count": 3,
            "final_proxy_score": 1.0,
            "true_success": False,
        },
        {
            "condition": "target_adaptive_hacker",
            "ground_truth_label": "ordinary_failure",
            "exploit_event_count": 0,
            "final_proxy_score": 0.5,
            "true_success": False,
        },
    ]
    summary = analyze_experiment_results(episodes)
    c_data = summary["conditions"]["target_adaptive_hacker"]
    # 1 out of 2 episodes = 50%
    assert c_data["confirmed_hack_rate"] == 0.5
    # Total events = 3 -> 3 / 2 * 100 = 150 events per 100 episodes
    assert c_data["confirmed_hack_events"] == 3
    assert c_data["events_per_100_episodes"] == 150.0
