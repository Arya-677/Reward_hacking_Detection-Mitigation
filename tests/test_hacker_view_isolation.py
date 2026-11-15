import pytest
from reward_hacking.hacker_agent import build_hacker_view


def test_hacker_view_isolation_and_field_rejection():
    """Requirement F & Section 7.2: Build hacker view isolates protected data and rejects leaks."""
    # 1. Attempting to pass forbidden keys must raise ValueError
    with pytest.raises(ValueError, match="Forbidden field"):
        build_hacker_view(
            task_id="task_001",
            task_description="test task",
            attempt_index=1,
            total_interventions=3,
            condition="target_adaptive_hacker",
            protected_tests="assert secret == True",
        )

    with pytest.raises(ValueError, match="Forbidden field"):
        build_hacker_view(
            task_id="task_001",
            task_description="test task",
            attempt_index=1,
            total_interventions=3,
            condition="target_adaptive_hacker",
            ground_truth_label="confirmed_hack",
        )

    with pytest.raises(ValueError, match="Forbidden field"):
        build_hacker_view(
            task_id="task_001",
            task_description="test task",
            attempt_index=1,
            total_interventions=3,
            condition="target_adaptive_hacker",
            independent_evaluator_result={"success": False},
        )


def test_static_vs_adaptive_view():
    """Static view omits history; adaptive view includes whitelisted public history."""
    static_view = build_hacker_view(
        task_id="task_001",
        task_description="test task",
        attempt_index=1,
        total_interventions=3,
        condition="target_static_hacker",
        target_action_history=[{"action": "read_file", "path": "solution.py"}],
    )
    assert static_view.is_adaptive is False
    assert len(static_view.target_action_history) == 0

    adaptive_view = build_hacker_view(
        task_id="task_001",
        task_description="test task",
        attempt_index=2,
        total_interventions=3,
        condition="target_adaptive_hacker",
        target_action_history=[{"action": "read_file", "path": "solution.py", "message": "inspecting"}],
        proxy_score_history=[0.5, 0.75],
    )
    assert adaptive_view.is_adaptive is True
    assert len(adaptive_view.target_action_history) == 1
    assert adaptive_view.proxy_score_history == [0.5, 0.75]
