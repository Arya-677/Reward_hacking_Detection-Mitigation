"""V4 Interaction and Adversarial Coupling Package."""

from reward_hacking.interaction.channel_router import ChannelRouter, DeliveryRecord
from reward_hacking.interaction.hacker_v4 import AttackPlanV4, HackerAgentV4, FAMILY_MECHANISMS
from reward_hacking.interaction.interaction_controller import InteractionController, InterventionRecordV4
from reward_hacking.interaction.profile_v4 import (
    TargetBehaviourProfileV4,
    build_deterministic_profile_v4,
    build_temporal_shuffle_profile,
    build_cross_task_mismatch_profile,
)
from reward_hacking.interaction.response_feedback import ResponseFeedback, adjudicate_public_response
from reward_hacking.interaction.vulnerability_memory import VulnerabilityMemory, VulnerabilityMemoryEntry

__all__ = [
    "ChannelRouter",
    "DeliveryRecord",
    "AttackPlanV4",
    "HackerAgentV4",
    "FAMILY_MECHANISMS",
    "InteractionController",
    "InterventionRecordV4",
    "TargetBehaviourProfileV4",
    "build_deterministic_profile_v4",
    "build_temporal_shuffle_profile",
    "build_cross_task_mismatch_profile",
    "ResponseFeedback",
    "adjudicate_public_response",
    "VulnerabilityMemory",
    "VulnerabilityMemoryEntry",
]
