import shutil
import tempfile
from pathlib import Path
import pytest

from reward_hacking.config import load_config
from reward_hacking.experiment_runner import ExperimentRunner
from reward_hacking.storage import ExperimentStorage


@pytest.fixture
def temp_run_dir():
    tmp = Path(tempfile.mkdtemp(prefix="test_runner_"))
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


def test_runner_dry_run_blocked_design_and_resume(temp_run_dir):
    """Requirement J: Runner blocked randomisation yields equal episodes; resume skips completed units."""
    config = load_config(overrides={"tasks": ["task_001", "task_002"]})
    storage = ExperimentStorage(temp_run_dir)

    runner = ExperimentRunner(config=config, storage=storage, dry_run=True)

    exp_id = "test_blocked_exp"
    # Run 2 blocks (10 episodes total: 2 per condition across 5 conditions)
    episodes = runner.run_blocked_experiment(
        experiment_id=exp_id,
        episodes_per_condition=2,
        auto_size=False,
        resume=False,
    )

    assert len(episodes) == 10
    by_cond = {}
    for ep in episodes:
        c = ep["condition"]
        by_cond[c] = by_cond.get(c, 0) + 1

    # Equal episodes per condition guaranteed by blocked design
    assert by_cond["target_only"] == 2
    assert by_cond["target_static_hacker"] == 2
    assert by_cond["target_adaptive_hacker"] == 2
    assert by_cond["target_benign_helper"] == 2
    assert by_cond["target_scripted_explicit"] == 2

    # Test resume: run again with resume=True -> should skip all 10 completed units
    completed = storage.get_completed_units()
    assert len(completed) == 10

    resumed_episodes = runner.run_blocked_experiment(
        experiment_id=exp_id,
        episodes_per_condition=2,
        auto_size=False,
        resume=True,
    )
    # Total episodes in storage remains 10 because none were re-run
    assert len(resumed_episodes) == 10
