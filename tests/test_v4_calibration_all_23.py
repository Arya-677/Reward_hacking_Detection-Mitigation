"""Unit test suite verifying complete 23-task calibration across all 5 fixture classes (115 cases)."""

import pytest
from pathlib import Path
from reward_hacking.protocols.calibration_all_23 import (
    run_all_23_tasks_calibration,
    run_single_calibration,
    FixtureClass,
)
from reward_hacking.registry import TASKS_BY_FAMILY


@pytest.mark.parametrize("family,tasks", TASKS_BY_FAMILY.items())
def test_family_calibration_fixtures(family: str, tasks: list):
    """Test all tasks in family across all 5 fixture classes."""
    classes = ["benign_success", "ordinary_failure", "attempted_hack", "confirmed_hack", "ambiguous"]
    for task_id in tasks:
        for fclass in classes:
            res = run_single_calibration(family, task_id, fclass)
            assert res["passed"] is True, f"Failed calibration for {task_id} with class {fclass}: {res}"


def test_full_calibration_bundle_output(tmp_path: Path):
    """Test that full calibration runs and generates compliant bundle JSON."""
    out_file = tmp_path / "task_calibration_bundle.json"
    summary = run_all_23_tasks_calibration(output_path=out_file)
    assert summary["all_passed"] is True
    assert summary["total_tasks"] == 23
    assert summary["total_cases"] == 115
    assert summary["total_passed"] == 115
    assert out_file.exists()
