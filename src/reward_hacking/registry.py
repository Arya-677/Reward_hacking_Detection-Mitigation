"""V4 Central Task and Family Registry (Section 4.1 & 4.4).

Manages all 6 task families and their respective task IDs:
- coding: 8 preserved tasks (task_001 to task_008)
- data_pipeline: 3 tasks (etl_01 to etl_03)
- sql_analytics: 3 tasks (sql_01 to sql_03)
- workflow: 3 tasks (wf_01 to wf_03)
- scheduling: 3 tasks (sched_01 to sched_03)
- document_evidence: 3 tasks (doc_01 to doc_03)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field

from reward_hacking.adapters.base import TaskAdapter


TASK_FAMILIES: List[str] = [
    "coding",
    "data_pipeline",
    "sql_analytics",
    "workflow",
    "scheduling",
    "document_evidence",
]

TASKS_BY_FAMILY: Dict[str, List[str]] = {
    "coding": [
        "task_001",
        "task_002",
        "task_003",
        "task_004",
        "task_005",
        "task_006",
        "task_007",
        "task_008",
    ],
    "data_pipeline": [
        "etl_01_transaction_dedup",
        "etl_02_sensor_quality",
        "etl_03_branch_kpi",
    ],
    "sql_analytics": [
        "sql_01_regional_sales",
        "sql_02_inventory_reconcile",
        "sql_03_retention",
    ],
    "workflow": [
        "wf_01_refund",
        "wf_02_incident",
        "wf_03_fulfilment",
    ],
    "scheduling": [
        "sched_01_staff_shifts",
        "sched_02_machine_jobs",
        "sched_03_exam_proctors",
    ],
    "document_evidence": [
        "doc_01_contract_terms",
        "doc_02_policy_qa",
        "doc_03_incident_timeline",
    ],
}

FAMILY_PRIMARY_MECHANISM: Dict[str, str] = {
    "coding": "M1_assertion_weakening",
    "data_pipeline": "M2_data_selection_gaming",
    "sql_analytics": "M3_sql_semantic_shortcut",
    "workflow": "M4_workflow_state_spoofing",
    "scheduling": "M5_constraint_violation_gaming",
    "document_evidence": "M6_evidence_rubric_gaming",
}


class TaskMetadata(BaseModel):
    task_id: str
    task_family: str
    title: str
    description: str
    proxy_metric: str
    true_objective_metric: str
    primary_mechanism: str
    allowed_tools: List[str]


class TaskRegistry:
    """Registry providing factory methods and metadata for all V4 task domains."""

    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).resolve().parents[2]
        self.coding_tasks_dir = self.project_root / "data" / "tasks"
        self.v4_tasks_dir = self.project_root / "data" / "tasks_v4"
        self._adapter_factories: Dict[str, Callable[[], TaskAdapter]] = {}

    def get_family_for_task(self, task_id: str) -> str:
        for family, task_ids in TASKS_BY_FAMILY.items():
            if task_id in task_ids:
                return family
        raise KeyError(f"Task ID '{task_id}' not found in any registered family")

    def list_families(self) -> List[str]:
        return list(TASK_FAMILIES)

    def list_tasks(self, family: Optional[str] = None) -> List[str]:
        if family is None:
            all_tasks: List[str] = []
            for f in TASK_FAMILIES:
                all_tasks.extend(TASKS_BY_FAMILY[f])
            return all_tasks
        if family not in TASKS_BY_FAMILY:
            raise KeyError(f"Unknown family '{family}'")
        return list(TASKS_BY_FAMILY[family])

    def get_adapter(self, family: str, sandbox: Optional[Any] = None) -> TaskAdapter:
        """Create a fresh TaskAdapter instance for the specified domain."""
        if family == "coding":
            from reward_hacking.adapters.coding import CodingTaskAdapter
            from reward_hacking.sandbox import Sandbox
            sb = sandbox or Sandbox(mode="restricted_subprocess", allow_restricted=True)
            return CodingTaskAdapter(tasks_dir=self.coding_tasks_dir, sandbox=sb)

        elif family == "data_pipeline":
            from reward_hacking.adapters.data_pipeline import DataPipelineTaskAdapter
            return DataPipelineTaskAdapter()

        elif family == "sql_analytics":
            from reward_hacking.adapters.sql_analytics import SqlAnalyticsTaskAdapter
            return SqlAnalyticsTaskAdapter()

        elif family == "workflow":
            from reward_hacking.adapters.workflow import WorkflowTaskAdapter
            return WorkflowTaskAdapter()

        elif family == "scheduling":
            from reward_hacking.adapters.scheduling import SchedulingTaskAdapter
            return SchedulingTaskAdapter()

        elif family == "document_evidence":
            from reward_hacking.adapters.document_evidence import DocumentEvidenceTaskAdapter
            return DocumentEvidenceTaskAdapter()

        else:
            raise ValueError(f"No adapter registered for task family '{family}'")

    def get_auditor(self, family: str) -> Any:
        """Create a MechanismAuditor instance for the specified domain."""
        if family == "coding":
            from reward_hacking.evaluation.auditors.m1_coding import M1CodingAuditor
            return M1CodingAuditor()
        elif family == "data_pipeline":
            from reward_hacking.evaluation.auditors.m2_data import M2DataPipelineAuditor
            return M2DataPipelineAuditor()
        elif family == "sql_analytics":
            from reward_hacking.evaluation.auditors.m3_sql import M3SqlAnalyticsAuditor
            return M3SqlAnalyticsAuditor()
        elif family == "workflow":
            from reward_hacking.evaluation.auditors.m4_workflow import M4WorkflowAuditor
            return M4WorkflowAuditor()
        elif family == "scheduling":
            from reward_hacking.evaluation.auditors.m5_scheduling import M5SchedulingAuditor
            return M5SchedulingAuditor()
        elif family == "document_evidence":
            from reward_hacking.evaluation.auditors.m6_document import M6DocumentEvidenceAuditor
            return M6DocumentEvidenceAuditor()
        else:
            raise ValueError(f"No auditor registered for task family '{family}'")

