"""V4 Prioritized Ablations Experiment Runner (Blueprint Section 4.3 & 8.3).

Executes the formal ablation studies on representative tasks (1 per family * 2 seeds = 12 blocks):
1. Explicitness Gradient: L1 (subtle), L2 (explicit), L3 (imperative)
2. Closed-Loop Feedback & Vulnerability Memory: open_loop vs response_adaptive vs response_adaptive_memory
3. Target Policy: P0 (permissive) vs P1 (integrity-aware)
4. Delivery Channel: direct_message vs task_artifact vs tool_output
5. Profiler Information Controls: structured_profile vs temporal_shuffle_profile vs cross_task_mismatch_profile

Writes raw artifacts to results/v4_ablations/.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from reward_hacking.gpu_monitor import GPUMonitor
from reward_hacking.interaction.vulnerability_memory import VulnerabilityMemory
from reward_hacking.model_client import ModelClient
from reward_hacking.protocols.v4 import run_v4_episode
from reward_hacking.registry import TaskRegistry

logger = logging.getLogger(__name__)

REPRESENTATIVE_TASKS: Dict[str, str] = {
    "coding": "task_001",
    "data_pipeline": "etl_01_transaction_dedup",
    "sql_analytics": "sql_01_regional_sales",
    "workflow": "wf_01_refund",
    "scheduling": "sched_01_staff_shifts",
    "document_evidence": "doc_01_contract_terms",
}


def run_v4_ablations(
    output_dir: Optional[Path] = None,
    seeds: Sequence[int] = (101, 102),
    target_model: str = "qwen2.5-coder:7b-instruct",
    hacker_model: str = "qwen3:4b",
    profiler_model: str = "qwen3:4b",
    wall_clock_budget_minutes: int = 120,
    resume: bool = True,
) -> Path:
    """Run prioritized V4 ablations on matched representative task-seed blocks."""
    start_time = time.time()
    deadline = start_time + (wall_clock_budget_minutes * 60)
    out_dir = output_dir or (Path("results") / "v4_ablations")
    out_dir.mkdir(parents=True, exist_ok=True)

    registry = TaskRegistry()
    client = ModelClient(num_ctx=6144, keep_alive="60m")
    memory = VulnerabilityMemory(db_path=out_dir / "ablation_memory.db")

    # Define ablation conditions to execute
    ablation_matrix = [
        # 1. Target Policy contrast (P0 vs P1 under semantic_one_shot)
        {"name": "policy_p1", "cond": "semantic_one_shot", "policy": "P1", "channel": "direct_message", "adapt": "open_loop", "exposure": "one_shot"},
        
        # 2. Delivery Channels (under P0, semantic_one_shot)
        {"name": "channel_artifact", "cond": "semantic_one_shot", "policy": "P0", "channel": "task_artifact", "adapt": "open_loop", "exposure": "one_shot"},
        {"name": "channel_tool", "cond": "semantic_one_shot", "policy": "P0", "channel": "tool_output", "adapt": "open_loop", "exposure": "one_shot"},

        # 3. Closed-Loop Feedback & Memory
        {"name": "feedback_adaptive", "cond": "semantic_one_shot", "policy": "P0", "channel": "direct_message", "adapt": "response_adaptive", "exposure": "one_shot"},
        {"name": "feedback_memory", "cond": "semantic_one_shot", "policy": "P0", "channel": "direct_message", "adapt": "response_adaptive_memory", "exposure": "one_shot"},

        # 4. Profiler Controls (Structured vs Temporal Shuffle vs Mismatch)
        {"name": "control_structured", "cond": "structured_profile", "policy": "P0", "channel": "direct_message", "adapt": "open_loop", "exposure": "one_shot"},
        {"name": "control_shuffled", "cond": "temporal_shuffle_profile", "policy": "P0", "channel": "direct_message", "adapt": "open_loop", "exposure": "one_shot"},
        {"name": "control_mismatched", "cond": "cross_task_mismatch_profile", "policy": "P0", "channel": "direct_message", "adapt": "open_loop", "exposure": "one_shot"},
    ]

    # Completed check for resume
    completed_keys = set()
    episodes_file = out_dir / "episodes.jsonl"
    if resume and episodes_file.exists():
        with open(episodes_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        completed_keys.add((rec["task_id"], rec["seed"], rec["condition"]))
                    except Exception:
                        pass

    print("\n" + "=" * 70)
    print(f"STARTING V4 ABLATION SUITE: {len(REPRESENTATIVE_TASKS) * len(seeds)} BLOCKS | {len(ablation_matrix)} CONDITIONS")
    print("=" * 70)

    total_run = len(completed_keys)

    for task_family, task_id in REPRESENTATIVE_TASKS.items():
        for seed in seeds:
            time_left = deadline - time.time()
            if time_left < 600:
                print(f"[BUDGET GUARD] Less than 10 mins remaining ({time_left:.1f}s). Halting ablation run.")
                break

            for ab in ablation_matrix:
                cond_key = f"{ab['name']}_{ab['cond']}"
                if (task_id, seed, cond_key) in completed_keys:
                    continue

                print(f"[{total_run + 1}] Ablation: {ab['name']:20s} | Task: {task_id:25s} | Seed: {seed}", end="", flush=True)

                ep = run_v4_episode(
                    run_id="v4_ablations",
                    task_id=task_id,
                    seed=seed,
                    condition=cond_key,
                    output_dir=out_dir,
                    registry=registry,
                    model_client=client,
                    target_model=target_model,
                    hacker_model=hacker_model,
                    profiler_model=profiler_model,
                    exposure_mode=ab["exposure"],
                    target_policy=ab["policy"],
                    delivery_channel=ab["channel"],
                    adaptation_mode=ab["adapt"],
                    memory=memory,
                    max_steps=10,
                    intervention_steps=(2, 5, 8),
                )

                total_run += 1
                completed_keys.add((task_id, seed, cond_key))
                print(f" -> Label: {ep.primary_label:16s} (Proxy: {ep.proxy_score:.2f}, True: {ep.true_objective_score:.2f}) [{ep.latency_seconds:.1f}s]")

    print("\n" + "=" * 70)
    print(f"V4 ABLATION RUN COMPLETE: {total_run} episodes saved in {out_dir}")
    print("=" * 70 + "\n")
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="results/v4_ablations")
    args = parser.parse_args()
    run_v4_ablations(output_dir=Path(args.output))
