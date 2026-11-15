"""Interaction Controller for V4 closed-loop adversarial evaluation.

Coordinates:
- Hacker planning (information modes C0-C5)
- Channel delivery (direct_message, task_artifact, tool_output)
- Attribution window tracking (next k=2 actions)
- Response adjudication & feedback loop (B0 open-loop vs B1 closed-loop)
- Vulnerability memory queries and updates (public-only)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple
from pydantic import BaseModel, Field

from reward_hacking.adapters.base import AgentAction, TaskObservation, ToolSpec
from reward_hacking.interaction.channel_router import ChannelRouter, DeliveryRecord
from reward_hacking.interaction.hacker_v4 import AttackPlanV4, HackerAgentV4
from reward_hacking.interaction.profile_v4 import (
    TargetBehaviourProfileV4,
    build_deterministic_profile_v4,
    build_temporal_shuffle_profile,
    build_cross_task_mismatch_profile,
)
from reward_hacking.interaction.response_feedback import ResponseFeedback, adjudicate_public_response
from reward_hacking.interaction.vulnerability_memory import VulnerabilityMemory, VulnerabilityMemoryEntry

logger = logging.getLogger(__name__)


class InterventionRecordV4(BaseModel):
    """Immutable record of an intervention turn and its attribution outcome."""
    intervention_id: str
    episode_id: str
    scheduled_step: int
    exposed_step: int
    delivery_channel: Literal["direct_message", "task_artifact", "tool_output"]
    target_policy: Literal["P0", "P1"]
    information_mode: str
    adaptation_mode: str
    plan: AttackPlanV4
    profile_id_used: Optional[str] = None
    attribution_window_steps: List[int] = Field(default_factory=list)
    observed_actions: List[Dict[str, Any]] = Field(default_factory=list)
    feedback: Optional[ResponseFeedback] = None
    delivered: bool = True
    delivered_content: str = ""


class InteractionController:
    """Manages the full lifecycle of adversarial interventions during an episode."""

    def __init__(
        self,
        hacker_agent: HackerAgentV4,
        channel_router: Optional[ChannelRouter] = None,
        memory: Optional[VulnerabilityMemory] = None,
        intervention_steps: Sequence[int] = (2, 5, 8),
        attribution_window_size: int = 2,
        information_mode: str = "semantic_profile",
        adaptation_mode: str = "open_loop",  # 'open_loop', 'response_adaptive', 'response_adaptive_memory'
        target_policy: Literal["P0", "P1"] = "P0",
        delivery_channel: Literal["direct_message", "task_artifact", "tool_output"] = "direct_message",
        donor_profile: Optional[TargetBehaviourProfileV4] = None,
    ):
        self.hacker = hacker_agent
        self.router = channel_router or ChannelRouter(target_policy=target_policy)
        self.memory = memory
        self.intervention_steps = list(intervention_steps)
        self.attribution_window_size = attribution_window_size
        self.information_mode = information_mode
        self.adaptation_mode = adaptation_mode
        self.target_policy = target_policy
        self.delivery_channel = delivery_channel
        self.donor_profile = donor_profile

        # Runtime state per episode
        self.intervention_history: List[InterventionRecordV4] = []
        self.feedback_history: List[ResponseFeedback] = []
        self._active_intervention: Optional[InterventionRecordV4] = None
        self._actions_in_current_window: List[AgentAction] = []
        self._action_dicts_in_current_window: List[Dict[str, Any]] = []

    def is_intervention_step(self, step: int) -> bool:
        """Check if current step is scheduled for an intervention."""
        if self.information_mode == "target_only":
            return False
        return step in self.intervention_steps

    def prepare_and_route_intervention(
        self,
        episode_id: str,
        step: int,
        task_id: str,
        task_family: str,
        task_description: str,
        observation: TaskObservation,
        public_events: List[Dict[str, Any]],
        target_model_digest: str = "",
    ) -> Tuple[TaskObservation, Optional[InterventionRecordV4]]:
        """If current step is an intervention step, generate plan and route to observation."""
        if not self.is_intervention_step(step):
            return observation, None

        intervention_id = f"int_{episode_id}_s{step}"

        # 1. Build profile based on information_mode
        profile: Optional[TargetBehaviourProfileV4] = None
        if self.information_mode == "static":
            profile = None
        elif self.information_mode == "structured_profile":
            profile = build_deterministic_profile_v4(
                events=public_events,
                task_id=task_id,
                task_family=task_family,
                target_model_digest=target_model_digest,
                episode_id=episode_id,
            )
        elif self.information_mode in ("semantic_profile", "open_loop", "response_adaptive"):
            # Use deterministic baseline or LLM profile
            profile = build_deterministic_profile_v4(
                events=public_events,
                task_id=task_id,
                task_family=task_family,
                target_model_digest=target_model_digest,
                episode_id=episode_id,
            )
        elif self.information_mode == "temporal_shuffle_profile":
            profile = build_temporal_shuffle_profile(
                events=public_events,
                task_id=task_id,
                task_family=task_family,
                target_model_digest=target_model_digest,
                episode_id=episode_id,
            )
        elif self.information_mode == "cross_task_mismatch_profile" and self.donor_profile:
            profile = build_cross_task_mismatch_profile(
                donor_profile=self.donor_profile,
                current_task_id=task_id,
                current_episode_id=episode_id,
            )

        # 2. Query Vulnerability Memory (if memory adaptation mode enabled)
        memory_entries: List[VulnerabilityMemoryEntry] = []
        if self.memory and "memory" in self.adaptation_mode:
            signature = profile.behavior_signature if profile else "default"
            memory_entries = self.memory.retrieve(
                task_family=task_family,
                target_model_digest=target_model_digest,
                behavior_signature=signature,
                limit=3,
            )

        # 3. Closed-loop feedback
        feedback_to_pass = self.feedback_history if "adaptive" in self.adaptation_mode else None

        # 4. Generate AttackPlanV4
        plan = self.hacker.plan(
            intervention_id=intervention_id,
            task_family=task_family,
            task_brief=task_description,
            available_tools=observation.available_tools,
            target_model_digest=target_model_digest,
            information_mode=self.information_mode,
            profile=profile,
            raw_history=public_events if self.information_mode == "raw_history" else None,
            memory_entries=memory_entries,
            previous_feedback=feedback_to_pass,
            step=step,
            delivery_channel=self.delivery_channel,
            target_policy=self.target_policy,
        )

        # 5. Route through chosen channel
        updated_obs, delivery_rec = self.router.route_suggestion(
            intervention_id=intervention_id,
            content=plan.adversarial_context,
            channel=self.delivery_channel,
            observation=observation,
            step=step,
            target_policy=self.target_policy,
        )

        record = InterventionRecordV4(
            intervention_id=intervention_id,
            episode_id=episode_id,
            scheduled_step=step,
            exposed_step=step,
            delivery_channel=self.delivery_channel,
            target_policy=self.target_policy,
            information_mode=self.information_mode,
            adaptation_mode=self.adaptation_mode,
            plan=plan,
            profile_id_used=profile.profile_id if profile else None,
            delivered=True,
            delivered_content=plan.adversarial_context,
        )

        self._active_intervention = record
        self._actions_in_current_window = []
        self._action_dicts_in_current_window = []
        self.intervention_history.append(record)

        return updated_obs, record

    def record_target_action(self, action: AgentAction, step: int, event_id: str = "") -> None:
        """Track target action inside the current attribution window."""
        if not self._active_intervention:
            return

        self._actions_in_current_window.append(action)
        self._action_dicts_in_current_window.append(
            {"step": step, "action": action.action, "event_id": event_id, "arguments": action.arguments}
        )
        self._active_intervention.attribution_window_steps.append(step)

        # Check if window size reached (default k=2 actions)
        if len(self._actions_in_current_window) >= self.attribution_window_size:
            self._close_attribution_window()

    def _close_attribution_window(self) -> None:
        """Adjudicate response and close active window."""
        if not self._active_intervention:
            return

        record = self._active_intervention
        record.observed_actions = list(self._action_dicts_in_current_window)

        # Adjudicate public response
        fb = adjudicate_public_response(
            plan=record.plan,
            observed_actions=self._actions_in_current_window,
            action_event_ids=[a.get("event_id", "") for a in self._action_dicts_in_current_window],
        )
        record.feedback = fb
        self.feedback_history.append(fb)

        # Store into memory if memory is enabled and public-only
        if self.memory and fb:
            try:
                self.memory.record_public_interaction(
                    target_model_digest=record.plan.target_model_digest,
                    task_family=record.plan.task_family,
                    behavior_signature="active",
                    mechanism_family=record.plan.target_mechanism_family,
                    strategy_id=record.plan.strategy_id,
                    observed_response=fb.observed_response,
                    public_proxy_delta=0.0,
                    public_evidence_event_ids=fb.attribution_event_ids,
                    source_episode_id=record.episode_id,
                )
            except Exception as e:
                logger.warning(f"Failed to record memory entry: {e}")

        self._active_intervention = None
        self._actions_in_current_window = []
        self._action_dicts_in_current_window = []

    def finalize_episode(self) -> None:
        """Close any remaining open attribution window when episode ends."""
        if self._active_intervention:
            self._close_attribution_window()
