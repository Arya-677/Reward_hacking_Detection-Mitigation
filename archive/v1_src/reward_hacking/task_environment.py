import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import yaml
from pydantic import BaseModel, Field

from reward_hacking.agent_tools import TargetAction, ToolResult, WorkspaceToolHandler
from reward_hacking.proxy_evaluator import ProxyEvaluationResult, ProxyEvaluator
from reward_hacking.sandbox import Sandbox


class EnvironmentObservation(BaseModel):
    task_id: str
    task_description: str
    last_tool_action: Optional[str] = None
    last_tool_output: str = ""
    last_test_output: str = ""
    proxy_score: float = 0.0
    step_count: int = 0
    max_steps: int = 6
    available_files: List[str] = Field(default_factory=list)


class TaskEnvironment:
    """RL-ready Task Environment for coding evaluation episodes."""

    def __init__(
        self,
        tasks_dir: Path,
        sandbox: Sandbox,
        max_steps: int = 6,
    ):
        self.tasks_dir = tasks_dir.resolve()
        self.sandbox = sandbox
        self.max_steps = max_steps
        self.proxy_evaluator = ProxyEvaluator(sandbox)

        self.workspace_dir: Optional[Path] = None
        self.current_task_id: Optional[str] = None
        self.task_dir: Optional[Path] = None
        self.task_metadata: Dict[str, Any] = {}
        self.tool_handler: Optional[WorkspaceToolHandler] = None

        self.step_count = 0
        self.current_proxy_score = 0.0
        self.initial_proxy_score = 0.0
        self.files_modified: Set[str] = set()
        self.step_history: List[Dict[str, Any]] = []
        self.proxy_score_history: List[float] = []
        self.last_test_output = ""

    def reset(
        self,
        seed: Optional[int] = None,
        task_id: Optional[str] = None,
    ) -> Tuple[EnvironmentObservation, Dict[str, Any]]:
        """Reset environment to a fresh workspace for the specified task."""
        self.close()

        if task_id is None:
            available = sorted([p.name for p in self.tasks_dir.iterdir() if p.is_dir() and (p / "task.yaml").exists()])
            if not available:
                raise FileNotFoundError(f"No valid tasks found in {self.tasks_dir}")
            task_id = available[0]

        self.current_task_id = task_id
        self.task_dir = (self.tasks_dir / task_id).resolve()
        if not self.task_dir.exists():
            raise FileNotFoundError(f"Task directory {self.task_dir} not found")

        # Load task metadata
        with open(self.task_dir / "task.yaml", "r", encoding="utf-8") as f:
            self.task_metadata = yaml.safe_load(f) or {}

        # Create fresh isolated temporary workspace
        self.workspace_dir = Path(tempfile.mkdtemp(prefix=f"rh_ws_{task_id}_"))
        shutil.copy2(self.task_dir / "buggy" / "solution.py", self.workspace_dir / "solution.py")
        shutil.copy2(self.task_dir / "visible" / "test_visible.py", self.workspace_dir / "test_visible.py")

        self.tool_handler = WorkspaceToolHandler(self.workspace_dir, sandbox_runner=self.sandbox)
        self.step_count = 0
        self.files_modified = set()
        self.step_history = []
        self.proxy_score_history = []

        # Run baseline proxy evaluation on initial buggy code
        initial_eval = self.proxy_evaluator.evaluate(self.workspace_dir)
        self.initial_proxy_score = initial_eval.score
        self.current_proxy_score = initial_eval.score
        self.proxy_score_history.append(self.initial_proxy_score)
        self.last_test_output = f"{initial_eval.stdout}\n{initial_eval.stderr}".strip()

        obs = EnvironmentObservation(
            task_id=self.current_task_id,
            task_description=self.task_metadata.get("description", ""),
            last_tool_action="reset",
            last_tool_output="Workspace initialized with solution.py and test_visible.py",
            last_test_output=self.last_test_output,
            proxy_score=self.current_proxy_score,
            step_count=0,
            max_steps=self.max_steps,
            available_files=["solution.py", "test_visible.py"],
        )

        info = {
            "task_id": self.current_task_id,
            "initial_proxy_score": self.initial_proxy_score,
            "seed": seed,
        }

        return obs, info

    def step(
        self, action: TargetAction
    ) -> Tuple[EnvironmentObservation, float, bool, bool, Dict[str, Any]]:
        """Execute one tool action in the environment.
        
        Returns:
            observation: EnvironmentObservation
            reward: float (current proxy score)
            terminated: bool (agent submitted solution)
            truncated: bool (step budget exhausted)
            info: dict (action metadata, execution details)
        """
        if self.workspace_dir is None or self.tool_handler is None:
            raise RuntimeError("Environment must be reset() before calling step()")

        self.step_count += 1
        terminated = False
        truncated = False
        tool_output = ""
        action_name = action.action

        # Record step start in history
        step_record: Dict[str, Any] = {
            "step": self.step_count,
            "action": action_name,
            "path": action.path,
            "content": action.content,
            "message": action.message,
        }

        if action_name == "list_files":
            res: ToolResult = self.tool_handler.list_files()
            tool_output = res.output

        elif action_name == "read_file":
            res = self.tool_handler.read_file(action.path)
            tool_output = res.output

        elif action_name == "write_file":
            res = self.tool_handler.write_file(action.path, action.content)
            tool_output = res.output
            if res.success and action.path:
                self.files_modified.add(Path(action.path).name)

        elif action_name == "run_visible_tests":
            eval_res: ProxyEvaluationResult = self.proxy_evaluator.evaluate(self.workspace_dir)
            self.current_proxy_score = eval_res.score
            combined = f"{eval_res.stdout}\n{eval_res.stderr}".strip()
            self.last_test_output = combined
            tool_output = (
                f"Visible Tests Run: {eval_res.passed_count}/{eval_res.total_count} passed "
                f"(score: {eval_res.score:.2f}).\n{combined[:500]}"
            )

        elif action_name == "submit_solution":
            terminated = True
            tool_output = "Solution submitted."

        else:
            tool_output = f"Unknown action: {action_name}"

        # If submit wasn't explicitly called, evaluate current proxy score after modification or test run
        if action_name == "write_file":
            # Auto-eval proxy score so history tracks progression
            eval_res = self.proxy_evaluator.evaluate(self.workspace_dir)
            self.current_proxy_score = eval_res.score
            self.last_test_output = f"{eval_res.stdout}\n{eval_res.stderr}".strip()

        self.proxy_score_history.append(self.current_proxy_score)
        step_record["tool_output"] = tool_output[:1000]
        step_record["proxy_score"] = self.current_proxy_score
        self.step_history.append(step_record)

        if self.step_count >= self.max_steps and not terminated:
            truncated = True

        obs = EnvironmentObservation(
            task_id=self.current_task_id or "",
            task_description=self.task_metadata.get("description", ""),
            last_tool_action=action_name,
            last_tool_output=tool_output[:1000],
            last_test_output=self.last_test_output[:1000],
            proxy_score=self.current_proxy_score,
            step_count=self.step_count,
            max_steps=self.max_steps,
            available_files=["solution.py", "test_visible.py"],
        )

        info = {
            "step": self.step_count,
            "action": action_name,
            "tool_output": tool_output,
            "proxy_score": self.current_proxy_score,
            "files_modified": list(self.files_modified),
            "auto_submitted": truncated and not terminated,
        }

        return obs, self.current_proxy_score, terminated, truncated, info

    def close(self):
        """Clean up temporary workspace."""
        if self.workspace_dir and self.workspace_dir.exists():
            shutil.rmtree(self.workspace_dir, ignore_errors=True)
            self.workspace_dir = None
