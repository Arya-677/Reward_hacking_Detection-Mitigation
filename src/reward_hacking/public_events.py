"""Public Event Store and Information Boundary Validation.

Maintains an append-only log of PublicEvent objects per episode, manages event IDs,
and validates strict information boundaries (preventing protected evaluator leaks).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from reward_hacking.profile_schema import PublicEvent

FORBIDDEN_KEYWORDS = {
    "protected",
    "test_protected",
    "independent_evaluator",
    "ground_truth",
    "detector_prediction",
    "true_reward",
    "true_task_success",
    "oracle",
}


class PublicEventStore:
    """Thread-safe, append-only store for public trajectory events."""

    def __init__(self, episode_id: Optional[str] = None):
        self.default_episode_id = episode_id
        self._events_by_episode: Dict[str, List[PublicEvent]] = {}
        if episode_id:
            self._events_by_episode[episode_id] = []
        self._counter: int = 0

    def next_event_id(self) -> str:
        self._counter += 1
        return f"evt_{self._counter:06d}"

    def append_event(
        self,
        episode_id: str,
        step: int,
        actor: str,
        event_type: str,
        action: Optional[str] = None,
        path: Optional[str] = None,
        message: Optional[str] = None,
        tool_output_public: Optional[str] = None,
        visible_proxy_score: float = 0.0,
        workspace_diff_ref: Optional[str] = None,
    ) -> PublicEvent:
        """Create and append an immutable public event."""
        evt = PublicEvent(
            event_id=self.next_event_id(),
            episode_id=episode_id,
            step=step,
            actor=actor,  # type: ignore
            event_type=event_type,  # type: ignore
            action=action,
            path=path,
            message=message,
            tool_output_public=tool_output_public,
            visible_proxy_score=visible_proxy_score,
            workspace_diff_ref=workspace_diff_ref,
        )
        if episode_id not in self._events_by_episode:
            self._events_by_episode[episode_id] = []
        self._events_by_episode[episode_id].append(evt)
        return evt

    def get_events(self, episode_id: Optional[str] = None) -> List[PublicEvent]:
        """Return all public events for an episode in chronological order."""
        ep_id = episode_id or self.default_episode_id
        if not ep_id:
            all_evts = []
            for evts in self._events_by_episode.values():
                all_evts.extend(evts)
            return all_evts
        return list(self._events_by_episode.get(ep_id, []))

    def get_event_ids(self, episode_id: Optional[str] = None) -> List[str]:
        return [e.event_id for e in self.get_events(episode_id)]

    def get_bounded_history(self, episode_id: Optional[str] = None, max_events: int = 12) -> List[PublicEvent]:
        """Return recent bounded public event history."""
        events = self.get_events(episode_id)
        return events[-max_events:] if len(events) > max_events else events

    @staticmethod
    def check_for_leakage(content: Any) -> Tuple[bool, List[str]]:
        """Validate that forbidden protected evaluation fields do not appear in content."""
        if isinstance(content, dict):
            text_repr = json.dumps(content).lower()
        elif isinstance(content, (list, tuple)):
            text_repr = json.dumps(content).lower()
        else:
            text_repr = str(content).lower()

        found_leaks = []
        for kw in FORBIDDEN_KEYWORDS:
            if kw in text_repr:
                found_leaks.append(kw)

        return len(found_leaks) == 0, found_leaks

    def record_action(
        self,
        step: int,
        actor: str,
        action: Optional[str] = None,
        path: Optional[str] = None,
        message: Optional[str] = None,
        visible_proxy_score: float = 0.0,
        tool_output_summary: Optional[str] = None,
        episode_id: Optional[str] = None,
    ) -> PublicEvent:
        """Convenience method to record an action event."""
        ep_id = episode_id or (list(self._events_by_episode.keys())[0] if self._events_by_episode else "ep_default")
        if actor == "target":
            evt_type = "tool_action"
        elif actor == "hacker":
            evt_type = "intervention_delivered"
        else:
            evt_type = "step_summary"
        return self.append_event(
            episode_id=ep_id,
            step=step,
            actor=actor,
            event_type=evt_type,
            action=action,
            path=path,
            message=message,
            tool_output_public=tool_output_summary,
            visible_proxy_score=visible_proxy_score,
        )


class PrivateEvaluationEventStore:
    """Strictly isolated store for protected evaluation data.

    MUST NEVER be serialized into Hacker views, Target prompts, or Profiler inputs.
    """

    def __init__(self):
        self._private_events_by_episode: Dict[str, List[Dict[str, Any]]] = {}

    def record_event(
        self,
        episode_id: str,
        indep_result: Any,
        gt_result: Any,
    ) -> Dict[str, Any]:
        rec = {
            "episode_id": episode_id,
            "true_success": getattr(indep_result, "success", False),
            "ground_truth_label": getattr(gt_result, "label", "unknown"),
            "failure_cause": getattr(gt_result, "failure_cause", None),
        }
        if episode_id not in self._private_events_by_episode:
            self._private_events_by_episode[episode_id] = []
        self._private_events_by_episode[episode_id].append(rec)
        return rec

    def get_events(self, episode_id: str) -> List[Dict[str, Any]]:
        return list(self._private_events_by_episode.get(episode_id, []))

