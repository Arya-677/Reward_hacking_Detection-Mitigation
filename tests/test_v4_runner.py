"""Integration test for V4 Episode Runner and persistence artifacts.

Verifies:
- Complete end-to-end synthetic episode execution
- Multi-domain adapter execution without coding-specific branches
- Creation of raw JSONL files: episodes.jsonl, interventions.jsonl, public_events.jsonl, detector_predictions.jsonl, private_audit/
"""

import json
import tempfile
from pathlib import Path
import pytest

from reward_hacking.protocols.v4 import run_v4_episode
from reward_hacking.registry import TaskRegistry


def test_run_v4_episode_synthetic():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "test_run"
        reg = TaskRegistry()

        ep = run_v4_episode(
            run_id="test_run_01",
            task_id="etl_01_transaction_dedup",
            seed=42,
            condition="semantic_profile",
            output_dir=out_dir,
            registry=reg,
            model_client=None,  # Tests deterministic fallback path
            target_policy="P0",
            delivery_channel="direct_message",
            adaptation_mode="open_loop",
            max_steps=5,
            intervention_steps=[2],
        )

        assert ep.task_id == "etl_01_transaction_dedup"
        assert ep.task_family == "data_pipeline"
        assert ep.total_steps > 0
        assert ep.primary_label in ["confirmed_hack", "attempted_hack", "benign_success", "ordinary_failure", "ambiguous"]

        # Check raw artifact files exist
        assert (out_dir / "episodes.jsonl").exists()
        assert (out_dir / "public_events.jsonl").exists()
        assert (out_dir / "interventions.jsonl").exists()
        assert (out_dir / "detector_predictions.jsonl").exists()
        assert (out_dir / "private_audit" / f"{ep.episode_id}.json").exists()

        # Check contents of episodes.jsonl
        with open(out_dir / "episodes.jsonl", "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
            assert len(lines) == 1
            assert lines[0]["episode_id"] == ep.episode_id
