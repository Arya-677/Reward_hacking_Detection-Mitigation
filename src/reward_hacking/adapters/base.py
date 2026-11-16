"""V4 Generic Environment and TaskAdapter Contract (Section 4.2).

Defines the universal request/response objects, tool specification,
and TaskAdapter protocol shared across all 6 task domains:
- coding
- data_pipeline
- sql_analytics
- workflow
- scheduling
- document_evidence
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, runtime_checkable
from pydantic import BaseModel, Field


def read_proxy_score(observation: TaskObservation) -> float:
    """Extract canonical proxy score from TaskObservation with strict contract validation (Section 3).
    
    A missing or out-of-range score is a task/evaluator contract error and must fail fast,
    never silently default to zero.
    """
    if not hasattr(observation, "visible_proxy_feedback"):
        raise TypeError(f"Expected TaskObservation with visible_proxy_feedback, got {type(observation).__name__}")
    raw = observation.visible_proxy_feedback.get("score")
    if raw is None:
        raise ValueError(
            f"Missing canonical proxy score for {observation.task_family}/"
            f"{observation.task_id}: {observation.visible_proxy_feedback}"
        )
    try:
        score = float(raw)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Non-numeric proxy score '{raw}': {e}") from e

    if not math.isfinite(score) or not (0.0 <= score <= 1.0):
        raise ValueError(f"Invalid normalized proxy score {score} for {observation.task_family}/{observation.task_id}")
    return score


class AgentAction(BaseModel):
    """Universal action emitted by Target Agent across all domains."""
    action: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""


class ToolSpec(BaseModel):
    """Specification of an allowed tool in the domain catalogue."""
    name: str
    description: str
    argument_schema: Dict[str, Any]
    risk_tags: List[str] = Field(default_factory=list)


class TaskObservation(BaseModel):
    """Universal observation exposed to Target Agent across all domains."""
    task_id: str
    task_family: str
    task_description: str
    public_state: Dict[str, Any]
    available_tools: List[ToolSpec]
    last_action_summary: str = ""
    visible_proxy_feedback: Dict[str, Any] = Field(default_factory=dict)
    step_count: int = 0
    max_steps: int = 10

    @property
    def proxy_score(self) -> float:
        """Convenience property extracting normalized proxy score (0.0 to 1.0)."""
        return read_proxy_score(self)


class StepResult(BaseModel):
    """Result returned by TaskAdapter.execute(action)."""
    observation: TaskObservation
    terminated: bool = False
    truncated: bool = False
    public_events: List[Dict[str, Any]] = Field(default_factory=list)


@runtime_checkable
class TaskAdapter(Protocol):
    """Universal TaskAdapter protocol (Blueprint Section 4.2).

    Boundary Rule:
    `private_audit_snapshot()` is callable ONLY by evaluator/auditor code post-episode.
    It must NEVER be serialized into TaskObservation, PublicEventStore, HackerView,
    Profiler input, or VulnerabilityMemory in primary conditions.
    """
    family: str

    def reset(self, task_id: str, seed: int) -> TaskObservation:
        """Reset task environment to initial state using deterministic seed."""
        ...

    def tools(self) -> Sequence[ToolSpec]:
        """Return allowed domain tool specifications."""
        ...

    def execute(self, action: AgentAction) -> StepResult:
        """Execute one validated action in the sandbox and return StepResult."""
        ...

    def public_evidence(self) -> Mapping[str, Any]:
        """Export public execution evidence (action history, public outputs, proxy trace)."""
        ...

    def private_audit_snapshot(self) -> Mapping[str, Any]:
        """Export private evaluation snapshot (hidden gold data, true state, evaluator assertions)."""
        ...

    def close(self) -> None:
        """Clean up ephemeral files, sandbox containers, and memory structures."""
        ...
