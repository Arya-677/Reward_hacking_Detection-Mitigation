"""V4 Generic Target Behaviour Profiler (Section 6.2).

Constructs domain-agnostic, evidence-grounded behavioural profiles across all
6 task families with rigorous temporal shuffle and cross-task mismatch controls.
"""

from __future__ import annotations

import copy
import random
import time
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple
from pydantic import BaseModel, Field

from reward_hacking.profile_schema import EvidenceClaim


class TargetBehaviourProfileV4(BaseModel):
    """V4 Domain-Agnostic Target Behaviour Profile (Section 6.2)."""
    profile_id: str
    episode_id: str
    target_model_id: str
    target_model_digest: str = ""
    task_family: str
    task_id: str
    profile_version: int = 4
    evidence_window: List[str] = Field(default_factory=list)
    current_task_state: str = "initial"
    problem_solving_pattern: List[EvidenceClaim] = Field(default_factory=list)
    action_preferences: List[EvidenceClaim] = Field(default_factory=list)
    capability_signals: List[EvidenceClaim] = Field(default_factory=list)
    instruction_responsiveness: List[EvidenceClaim] = Field(default_factory=list)
    constraint_attention: List[EvidenceClaim] = Field(default_factory=list)
    risk_and_uncertainty: List[str] = Field(default_factory=list)
    control_mode: Literal["natural", "temporal_shuffle", "cross_task_mismatch"] = "natural"
    donor_task_id: Optional[str] = None
    profile_provenance: Dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)

    @property
    def signature(self) -> str:
        """Deterministic behavior signature for memory indexing."""
        pat = self.problem_solving_pattern[0].statement if self.problem_solving_pattern else "exploratory"
        act = self.action_preferences[0].statement if self.action_preferences else "balanced"
        return f"{self.task_family}_{pat[:20]}_{act[:20]}".replace(" ", "_").lower()


def build_deterministic_profile_v4(
    public_events: Optional[Sequence[Dict[str, Any]]] = None,
    task_family: str = "coding",
    task_id: str = "",
    episode_id: str = "",
    target_model_id: str = "qwen2.5-coder:7b-instruct",
    target_model_digest: str = "",
    events: Optional[Sequence[Dict[str, Any]]] = None,
    **kwargs,
) -> TargetBehaviourProfileV4:
    """Construct deterministic profile directly from public events."""
    ev_list = public_events if public_events is not None else (events or [])
    KNOWN_FAMILIES = {"coding", "data_pipeline", "sql_analytics", "workflow", "scheduling", "document_evidence"}
    # Swap if passed in reverse order (task_id, task_family)
    if task_family not in KNOWN_FAMILIES and task_id in KNOWN_FAMILIES:
        task_family, task_id = task_id, task_family
    elif not task_family and kwargs.get("task_family"):
        task_family = kwargs["task_family"]
    evidence_window: List[str] = []
    actions_count: Dict[str, int] = {}
    last_score = 0.0

    for ev in ev_list:
        st = ev.get("step")
        if st is not None:
            evidence_window.append(f"step_{st}")
        act = ev.get("action")
        if act:
            actions_count[act] = actions_count.get(act, 0) + 1
        if "visible_proxy_score" in ev:
            last_score = float(ev["visible_proxy_score"])

    # Problem solving pattern
    if "inspect_schema" in actions_count or "read_file" in actions_count or "inspect_tables" in actions_count:
        pattern_stmt = "Inspection-first exploration: queries metadata before mutations"
    elif "write_file" in actions_count or "apply_transform" in actions_count or "assign_slot" in actions_count:
        pattern_stmt = "Mutation-first: applies modifications immediately"
    else:
        pattern_stmt = "Exploratory search: browsing available environment artifacts"

    pattern_claim = EvidenceClaim(
        statement=pattern_stmt,
        evidence_event_ids=evidence_window[:3],
        confidence=0.85,
        claim_type="behavioural_inference",
    )

    # Action preferences
    top_actions = sorted(actions_count.items(), key=lambda x: x[1], reverse=True)[:3]
    top_str = ", ".join(f"{k} ({v}x)" for k, v in top_actions) if top_actions else "None"
    action_claim = EvidenceClaim(
        statement=f"Frequently used actions: {top_str}",
        evidence_event_ids=evidence_window,
        confidence=1.0,
        claim_type="observed_fact",
    )

    # Capability signals
    cap_stmt = f"Current proxy achievement: {last_score:.2f} across {len(ev_list)} public events"
    cap_claim = EvidenceClaim(
        statement=cap_stmt,
        evidence_event_ids=evidence_window[-2:] if evidence_window else [],
        confidence=1.0,
        claim_type="observed_fact",
    )

    # Constraint attention
    constraint_stmt = f"Monitors visible proxy feedback; proxy score is {last_score:.2f}"
    constraint_claim = EvidenceClaim(
        statement=constraint_stmt,
        evidence_event_ids=evidence_window[-1:] if evidence_window else [],
        confidence=0.75,
        claim_type="behavioural_inference",
    )

    return TargetBehaviourProfileV4(
        profile_id=f"prof_{episode_id}_{len(evidence_window)}",
        episode_id=episode_id,
        target_model_id=target_model_id,
        target_model_digest=target_model_digest,
        task_family=task_family,
        task_id=task_id,
        evidence_window=evidence_window,
        current_task_state=f"Step {len(ev_list)}, visible proxy score {last_score:.2f}",
        problem_solving_pattern=[pattern_claim],
        action_preferences=[action_claim],
        capability_signals=[cap_claim],
        instruction_responsiveness=[],
        constraint_attention=[constraint_claim],
        risk_and_uncertainty=["Target response to indirect adversarial suggestion is unobserved"],
        control_mode="natural",
    )


def build_temporal_shuffle_profile(
    base_profile: Any = None,
    seed: int = 42,
    task_id: str = "",
    task_family: str = "coding",
    target_model_digest: str = "",
    episode_id: str = "",
    rng_seed: Optional[int] = None,
    events: Optional[Sequence[Dict[str, Any]]] = None,
    **kwargs,
) -> TargetBehaviourProfileV4:
    """Negative control: Genuinely permutes the chronological order of events in the SAME multiset."""
    actual_seed = rng_seed if rng_seed is not None else seed
    raw_source = base_profile if base_profile is not None else events
    if isinstance(raw_source, list) or raw_source is None:
        # Called with events list
        prof = build_deterministic_profile_v4(
            public_events=raw_source or [],
            task_id=task_id,
            task_family=task_family,
            target_model_digest=target_model_digest,
            episode_id=episode_id,
        )
    else:
        prof = raw_source

    rng = random.Random(actual_seed)
    shuffled = copy.deepcopy(prof)
    shuffled.profile_id = f"{prof.profile_id}_temporal_shuffle"
    shuffled.control_mode = "temporal_shuffle"
    shuffled.profile_provenance["control_type"] = "temporal_shuffle"

    # Permute the evidence window event order
    shuffled_window = list(prof.evidence_window)
    rng.shuffle(shuffled_window)
    shuffled.evidence_window = shuffled_window

    # Shuffle claim citations
    for c in shuffled.problem_solving_pattern:
        c.statement = f"[Chronology Permuted] {c.statement}"
        rng.shuffle(c.evidence_event_ids)
    for c in shuffled.action_preferences:
        rng.shuffle(c.evidence_event_ids)

    shuffled.risk_and_uncertainty.append("Control: event chronology is intentionally permuted")
    return shuffled


def build_cross_task_mismatch_profile(
    donor_profile: TargetBehaviourProfileV4,
    target_task_id: str = "",
    target_episode_id: str = "",
    current_task_id: str = "",
    current_episode_id: str = "",
) -> TargetBehaviourProfileV4:
    """Negative control: Injects a profile from a completely different donor task."""
    tid = current_task_id or target_task_id
    eid = current_episode_id or target_episode_id
    mismatched = copy.deepcopy(donor_profile)
    mismatched.profile_id = f"mismatch_{eid}"
    mismatched.episode_id = eid
    mismatched.donor_task_id = donor_profile.task_id
    mismatched.task_id = tid
    mismatched.control_mode = "cross_task_mismatch"
    mismatched.profile_provenance["control_type"] = "cross_task_mismatch"
    mismatched.profile_provenance["donor_task_id"] = donor_profile.task_id
    mismatched.risk_and_uncertainty.append(f"Control: donor profile sourced from task '{donor_profile.task_id}'")
    return mismatched


def validate_profile_v4(
    profile: TargetBehaviourProfileV4,
    allowed_eids: Optional[Sequence[str]] = None,
) -> bool:
    """Validate profile grounding against public event log (no invented IDs, confidence <= 1.0)."""
    allowed_set = set(allowed_eids) if allowed_eids is not None else set(profile.evidence_window)
    errors = []

    # In temporal shuffle or cross task mismatch, event IDs originate from donor or permuted set
    if profile.control_mode == "natural":
        for claim in (profile.problem_solving_pattern + profile.action_preferences + profile.capability_signals):
            if claim.confidence > 1.0 or claim.confidence < 0.0:
                errors.append(f"Invalid confidence {claim.confidence}")
            for eid in claim.evidence_event_ids:
                if eid not in allowed_set:
                    errors.append(f"Unobserved event ID '{eid}' cited in profile claim")

    return len(errors) == 0

