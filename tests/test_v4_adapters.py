"""Unit and contract tests for all V4 domain task adapters.

Verifies:
- TaskAdapter protocol compliance across all 6 domains
- Deterministic reset with seeds
- Tool validation and rejection of out-of-bounds actions
- Separation of visible proxy feedback and isolated private audit snapshots
"""

import pytest
from reward_hacking.adapters.base import AgentAction
from reward_hacking.registry import TaskRegistry, TASKS_BY_FAMILY


@pytest.fixture
def registry():
    return TaskRegistry()


def test_registry_families_and_tasks(registry):
    families = registry.list_families()
    assert len(families) == 6
    assert set(families) == {"coding", "data_pipeline", "sql_analytics", "workflow", "scheduling", "document_evidence"}

    all_tasks = registry.list_tasks()
    assert len(all_tasks) == 23
    assert len(registry.list_tasks("coding")) == 8
    assert len(registry.list_tasks("data_pipeline")) == 3
    assert len(registry.list_tasks("sql_analytics")) == 3
    assert len(registry.list_tasks("workflow")) == 3
    assert len(registry.list_tasks("scheduling")) == 3
    assert len(registry.list_tasks("document_evidence")) == 3


def test_data_pipeline_adapter(registry):
    adapter = registry.get_adapter("data_pipeline")
    obs = adapter.reset("etl_01_transaction_dedup", seed=42)
    assert obs.task_id == "etl_01_transaction_dedup"
    assert obs.task_family == "data_pipeline"
    assert len(obs.available_tools) >= 5

    # 1. Invalid action test
    res = adapter.execute(AgentAction(action="non_existent_tool", arguments={}))
    assert "unknown" in res.observation.last_action_summary.lower()

    # 2. Inspect schema
    res = adapter.execute(AgentAction(action="inspect_schema", arguments={}))
    assert "schema" in res.observation.last_action_summary.lower()

    # 3. Apply transform (valid)
    res = adapter.execute(AgentAction(action="apply_transform", arguments={"operation": "deduplicate_by_key", "params": {"keys": ["transaction_id"]}}))
    assert res.observation.visible_proxy_feedback.get("score", 0) > 0

    # 4. Submit result
    res = adapter.execute(AgentAction(action="submit_result", arguments={}))
    assert res.terminated is True

    # 5. Private audit snapshot
    snapshot = adapter.private_audit_snapshot()
    assert "true_objective_score" in snapshot
    adapter.close()


def test_sql_analytics_adapter(registry):
    adapter = registry.get_adapter("sql_analytics")
    obs = adapter.reset("sql_01_regional_sales", seed=42)
    assert obs.task_id == "sql_01_regional_sales"
    assert obs.task_family == "sql_analytics"

    # 1. Reject mutating SQL
    res = adapter.execute(AgentAction(action="execute_query", arguments={"query": "DROP TABLE customers;"}))
    assert "not authorized" in res.observation.last_action_summary.lower() or "error" in res.observation.last_action_summary.lower()

    # 2. Valid read query
    res = adapter.execute(AgentAction(action="execute_query", arguments={"query": "SELECT * FROM customers LIMIT 5;"}))
    assert "sample rows" in res.observation.last_action_summary.lower() or "result" in res.observation.last_action_summary.lower()

    # 3. Submit analytical query
    res = adapter.execute(AgentAction(action="submit_query", arguments={"query": "SELECT region, SUM(amount) FROM customers JOIN orders ON customers.id = orders.cust_id GROUP BY region;"}))
    assert res.terminated is True

    snapshot = adapter.private_audit_snapshot()
    assert "true_objective_score" in snapshot
    adapter.close()


def test_workflow_adapter(registry):
    adapter = registry.get_adapter("workflow")
    obs = adapter.reset("wf_01_refund", seed=42)
    assert obs.task_id == "wf_01_refund"
    assert obs.task_family == "workflow"

    # Happy path execution
    res1 = adapter.execute(AgentAction(action="get_case", arguments={}))
    assert "case" in res1.observation.last_action_summary.lower()

    valid_token = adapter.case_data.get("valid_auth_token", "")
    res2 = adapter.execute(AgentAction(action="verify_identity", arguments={"auth_token": valid_token}))
    assert "verified" in res2.observation.last_action_summary.lower()

    res3 = adapter.execute(AgentAction(action="issue_refund", arguments={"amount": 50.0, "reason": "damaged_goods"}))
    assert "issued" in res3.observation.last_action_summary.lower()

    res4 = adapter.execute(AgentAction(action="update_status", arguments={"status": "CLOSED", "note": "Issue resolved and refunded."}))
    res5 = adapter.execute(AgentAction(action="submit_case", arguments={}))
    assert res5.terminated is True

    snapshot = adapter.private_audit_snapshot()
    assert snapshot["true_objective_score"] == 1.0
    adapter.close()



def test_scheduling_adapter(registry):
    adapter = registry.get_adapter("scheduling")
    obs = adapter.reset("sched_01_staff_shifts", seed=42)
    assert obs.task_id == "sched_01_staff_shifts"
    assert obs.task_family == "scheduling"

    # Assign valid slot
    res1 = adapter.execute(AgentAction(action="assign_slot", arguments={"slot_id": "D1_MORN", "entity_id": "STF-01"}))
    assert res1.observation.visible_proxy_feedback.get("score", 0) > 0

    # Submit schedule
    res2 = adapter.execute(AgentAction(action="submit_schedule", arguments={}))
    assert res2.terminated is True

    snapshot = adapter.private_audit_snapshot()
    assert "true_objective_score" in snapshot
    adapter.close()


def test_document_evidence_adapter(registry):
    adapter = registry.get_adapter("document_evidence")
    obs = adapter.reset("doc_01_contract_terms", seed=42)
    assert obs.task_id == "doc_01_contract_terms"
    assert obs.task_family == "document_evidence"

    # Read section
    res1 = adapter.execute(AgentAction(action="read_section", arguments={"doc_id": "DOC_MSA", "section_id": "SEC_4.2"}))
    assert "SEC_4.2" in res1.observation.last_action_summary

    # Submit valid answers
    res2 = adapter.execute(AgentAction(action="submit_answers", arguments={
        "answers": {
            "contract_id": "CTR-01",
            "liability_cap": "$500,000",
            "governing_law": "Delaware",
            "citations": ["DOC-01:P2"],
        }
    }))
    assert res2.terminated is True


    snapshot = adapter.private_audit_snapshot()
    assert "true_objective_score" in snapshot
    adapter.close()


def test_adapter_seed_determinism(registry):
    adapter1 = registry.get_adapter("data_pipeline")
    adapter2 = registry.get_adapter("data_pipeline")

    obs1 = adapter1.reset("etl_01_transaction_dedup", seed=123)
    obs2 = adapter2.reset("etl_01_transaction_dedup", seed=123)

    assert obs1.public_state == obs2.public_state
    assert obs1.task_description == obs2.task_description
    adapter1.close()
    adapter2.close()
