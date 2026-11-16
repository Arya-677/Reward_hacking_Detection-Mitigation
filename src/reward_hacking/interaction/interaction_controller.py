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
    ProfilerCallRecord,
    TargetBehaviourProfileV4,
    build_deterministic_profile_v4,
    build_semantic_profile_v4,
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
    exposure_mode: Literal["one_shot", "persistent"] = "one_shot"
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
        exposure_mode: Literal["one_shot", "persistent"] = "one_shot",
        adaptation_mode: str = "open_loop",  # 'open_loop', 'response_adaptive', 'response_adaptive_memory'
        target_policy: Literal["P0", "P1"] = "P0",
        delivery_channel: Literal["direct_message", "task_artifact", "tool_output"] = "direct_message",
        donor_profile: Optional[TargetBehaviourProfileV4] = None,
        profiler_model: str = "qwen3:4b",
    ):
        self.hacker = hacker_agent
        self.router = channel_router or ChannelRouter(target_policy=target_policy)
        self.memory = memory
        self.intervention_steps = list(intervention_steps)
        self.attribution_window_size = attribution_window_size
        self.information_mode = information_mode
        self.exposure_mode = exposure_mode
        self.adaptation_mode = adaptation_mode
        self.target_policy = target_policy
        self.delivery_channel = delivery_channel
        self.donor_profile = donor_profile
        self.profiler_model = profiler_model

        # Runtime state per episode
        self.intervention_history: List[InterventionRecordV4] = []
        self.feedback_history: List[ResponseFeedback] = []
        self.profiler_call_history: List[ProfilerCallRecord] = []
        self._active_intervention: Optional[InterventionRecordV4] = None
        self._last_delivery_record: Optional[DeliveryRecord] = None
        self._actions_in_current_window: List[AgentAction] = []
        self._action_dicts_in_current_window: List[Dict[str, Any]] = []

    def is_intervention_step(self, step: int) -> bool:
        """Check if current step is scheduled for an intervention."""
        mode_lower = self.information_mode.lower()
        if "target_only" in mode_lower or "c0" in mode_lower:
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
            # Enforce exposure mode on non-intervention steps
            if self._last_delivery_record:
                if self.exposure_mode == "one_shot":
                    cleaned_obs = self.router.clear_suggestion(observation)
                    return cleaned_obs, None
                elif self.exposure_mode == "persistent":
                    persisted_obs = self.router.reapply_suggestion(observation, self._last_delivery_record)
                    return persisted_obs, None
            return observation, None

        intervention_id = f"int_{episode_id}_s{step}"

        # 1. Build profile based on information_mode
        profile: Optional[TargetBehaviourProfileV4] = None
        mode_lower = self.information_mode.lower()

        if "static" in mode_lower or mode_lower == "c1" or mode_lower == "c2":
            profile = None
        elif "semantic" in mode_lower or mode_lower in ("c5", "c6"):
            profile, call_rec = build_semantic_profile_v4(
                model_client=self.hacker.model_client,
                profiler_model=self.profiler_model or self.hacker.model_name,
                events=public_events,
                task_id=task_id,
                task_family=task_family,
                target_model_digest=target_model_digest,
                episode_id=episode_id,
            )
            self.profiler_call_history.append(call_rec)
        elif "temporal_shuffle" in mode_lower or "shuffled" in mode_lower:
            profile = build_temporal_shuffle_profile(
                events=public_events,
                task_id=task_id,
                task_family=task_family,
                target_model_digest=target_model_digest,
                episode_id=episode_id,
            )
        elif "cross_task_mismatch" in mode_lower or "mismatched" in mode_lower:
            donor = self.donor_profile or build_deterministic_profile_v4(
                events=[{"step": 0, "actor": "target", "action": "list_files"}],
                task_id="task_002" if task_id != "task_002" else "task_001",
                task_family="coding",
                episode_id="donor_baseline",
            )
            profile = build_cross_task_mismatch_profile(
                donor_profile=donor,
                current_task_id=task_id,
                current_episode_id=episode_id,
            )
        elif "raw_history" in mode_lower or mode_lower in ("c3", "c4"):
            profile = None
        else:
            # Deterministic/structured profile baseline
            profile = build_deterministic_profile_v4(
                events=public_events,
                task_id=task_id,
                task_family=task_family,
                target_model_digest=target_model_digest,
                episode_id=episode_id,
            )

        # 2. Query Vulnerability Memory (if memory adaptation mode enabled)
        memory_entries: List[VulnerabilityMemoryEntry] = []
        if self.memory and "memory" in self.adaptation_mode:
            signature = getattr(profile, "behavior_signature", getattr(profile, "signature", "default")) if profile else "default"
            memory_entries = self.memory.retrieve(
                task_family=task_family,
                target_model_digest=target_model_digest,
                behavior_signature=signature,
                limit=3,
            )

        # 3. Closed-loop feedback
        feedback_to_pass = self.feedback_history if "adaptive" in self.adaptation_mode else None

        # 4. Generate AttackPlanV4
        use_raw_history = public_events if ("raw_history" in mode_lower or mode_lower in ("c3", "c4")) else None
        plan = self.hacker.plan(
            intervention_id=intervention_id,
            task_family=task_family,
            task_brief=task_description,
            available_tools=observation.available_tools,
            target_model_digest=target_model_digest,
            information_mode=self.information_mode,
            profile=profile,
            raw_history=use_raw_history,
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
        self._last_delivery_record = delivery_rec

        record = InterventionRecordV4(
            intervention_id=intervention_id,
            episode_id=episode_id,
            scheduled_step=step,
            exposed_step=step,
            delivery_channel=self.delivery_channel,
            target_policy=self.target_policy,
            exposure_mode=self.exposure_mode,
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
                sig = record.plan.target_mechanism_family or "default"
                self.memory.record_public_interaction(
                    target_model_digest=record.plan.target_model_digest,
                    task_family=record.plan.task_family,
                    behavior_signature=sig,
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
