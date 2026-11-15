import tempfile
from pathlib import Path
import pytest

from reward_hacking.agent_tools import TargetAction
from reward_hacking.sandbox import Sandbox
from reward_hacking.task_environment import TaskEnvironment


@pytest.fixture
def tasks_dir():
    return Path(__file__).resolve().parents[1] / "data" / "tasks"


@pytest.fixture
def sandbox():
    return Sandbox(mode="restricted_subprocess", allow_restricted=True)


def test_task_environment_lifecycle(tasks_dir, sandbox):
    """Test environment reset, tool actions, and step termination."""
    env = TaskEnvironment(tasks_dir=tasks_dir, sandbox=sandbox, max_steps=4)

    obs, info = env.reset(seed=42, task_id="task_001")
    assert obs.task_id == "task_001"
    assert obs.step_count == 0
    assert 0.0 <= obs.proxy_score <= 1.0

    # Step 1: list files
    act1 = TargetAction(action="list_files", message="list files")
    obs, reward, term, trunc, info = env.step(act1)
    assert obs.step_count == 1
    assert "solution.py" in obs.last_tool_output
    assert term is False
    assert trunc is False

    # Step 2: read file
    act2 = TargetAction(action="read_file", path="solution.py", message="read solution")
    obs, reward, term, trunc, info = env.step(act2)
    assert obs.step_count == 2
    assert "def is_palindrome" in obs.last_tool_output

    # Step 3: submit solution
    act3 = TargetAction(action="submit_solution", message="done")
    obs, reward, term, trunc, info = env.step(act3)
    assert term is True

    env.close()
