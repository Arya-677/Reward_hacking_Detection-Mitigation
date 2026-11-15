"""V4 MechanismAuditor Protocol and AuditEvidenceBundle (Section 7.1 & 7.2).

Formalizes the evidence contract for domain-specific auditors (M1-M6).
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Protocol, runtime_checkable
from pydantic import BaseModel, Field


class AuditEvidenceBundle(BaseModel):
    """Section 7.2 multi-axis evidence record returned by a MechanismAuditor."""
    mechanism_id: str
    attempted: bool = False
    verified: bool = False
    attributable_proxy_gain: float = 0.0
    independent_objective_violation: float = 0.0  # 0.0 = no violation, 1.0 = complete failure
    evidence_event_ids: List[str] = Field(default_factory=list)
    artifact_evidence: Dict[str, Any] = Field(default_factory=dict)
    audit_confidence: float = 1.0
    evidence_summary: str = ""


@runtime_checkable
class MechanismAuditor(Protocol):
    """Protocol for mechanism-specific exploit auditing across all domains."""
    mechanism_id: str
    family: str

    def inspect(
        self,
        private_snapshot: Mapping[str, Any],
        public_evidence: Mapping[str, Any],
    ) -> AuditEvidenceBundle:
        """Inspect isolated private snapshot and public trace to verify exploit mechanism."""
        ...
