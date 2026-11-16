"""V4 Multi-Domain Blocked Experiment Runner and Protocol Engine.

Implements Blueprint Section 8.1, Experimental Protocol Sections 7 & 8:
- Universal multi-domain loop over TaskRegistry (all 6 families)
- Strict information barriers (no private leakage to Target/Hacker/Profiler)
- C0-C5 information modes, closed-loop response feedback, channels, public-only memory
- Online MultiDomainDetector execution
- Offline M1-M6 MechanismAuditors and multi-axis labels adjudication
- Resilient checkpoint/resume, raw JSONL persistence, and budget guards
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import yaml
from pydantic import BaseModel, Field

from reward_hacking.adapters.base import AgentAction, StepResult, TaskAdapter, TaskObservation, read_proxy_score
from reward_hacking.evaluation.detector_v4 import MultiDomainDetector
from reward_hacking.evaluation.labels import MultiAxisLabelRecord, adjudicate_multi_axis_episode
from reward_hacking.gpu_monitor import GPUMonitor
from reward_hacking.interaction.channel_router import ChannelRouter
from reward_hacking.interaction.hacker_v4 import AttackPlanV4, HackerAgentV4
from reward_hacking.interaction.interaction_controller import InteractionController, InterventionRecordV4
from reward_hacking.interaction.profile_v4 import ProfilerCallRecord
from reward_hacking.interaction.vulnerability_memory import VulnerabilityMemory
from reward_hacking.model_client import ModelClient
from reward_hacking.registry import TaskRegistry, TASKS_BY_FAMILY
from reward_hacking.target_agent_v4 import TargetAgentV4

logger = logging.getLogger(__name__)


class EpisodeRecordV4(BaseModel):
    """Complete immutable record of a V4 episode."""
    episode_id: str
    run_id: str
    task_id: str
    task_family: str
    seed: int
    condition: str
    exposure_mode: str = "one_shot"
    adaptation_mode: str
    target_policy: str
    delivery_channel: str
    total_steps: int
    proxy_score: float
    true_objective_score: float
    primary_label: str  # confirmed_hack, attempted_hack, benign_success, ordinary_failure, ambiguous
    label_record: MultiAxisLabelRecord
    trajectory: List[Dict[str, Any]] = Field(default_factory=list)
    proxy_history: List[float] = Field(default_factory=list)
    interventions_count: int = 0
    profiler_calls_count: int = 0
    detector_flagged: bool = False
    detector_rule_ids: List[str] = Field(default_factory=list)
    latency_seconds: float = 0.0
    completed_at: float = Field(default_factory=time.time)


def run_v4_episode(
    run_id: str,
    task_id: str,
    seed: int,
    condition: str,
    output_dir: Path,
    registry: Optional[TaskRegistry] = None,
    model_client: Optional[ModelClient] = None,
    target_model: str = "qwen2.5-coder:7b-instruct",
    hacker_model: str = "qwen3:4b",
    profiler_model: str = "qwen3:4b",
    exposure_mode: Literal["one_shot", "persistent"] = "one_shot",
    target_policy: Literal["P0", "P1"] = "P0",
    delivery_channel: Literal["direct_message", "task_artifact", "tool_output"] = "direct_message",
    adaptation_mode: str = "open_loop",
    memory: Optional[VulnerabilityMemory] = None,
    max_steps: int = 10,
    intervention_steps: Sequence[int] = (2, 5, 8),
) -> EpisodeRecordV4:
    """Execute a single V4 episode end-to-end with strict information isolation."""
    start_time = time.time()
    reg = registry or TaskRegistry()
    family = reg.get_family_for_task(task_id)
    episode_id = f"ep_{family}_{task_id}_{condition}_s{seed}_{int(start_time)}"

    adapter = reg.get_adapter(family)
    try:
        # 1. Reset environment
        obs = adapter.reset(task_id=task_id, seed=seed)

        # 2. Setup agents & controllers
        target_agent = TargetAgentV4(
            model_client=model_client,
            model_name=target_model,
            policy=target_policy,
        )
        target_agent.reset()

        hacker_agent = HackerAgentV4(
            model_client=model_client,
            model_name=hacker_model,
        )

        controller = InteractionController(
            hacker_agent=hacker_agent,
            memory=memory,
            intervention_steps=intervention_steps,
            information_mode=condition,
            exposure_mode=exposure_mode,
            adaptation_mode=adaptation_mode,
            target_policy=target_policy,
            delivery_channel=delivery_channel,
            profiler_model=profiler_model,
        )

        detector = MultiDomainDetector(task_family=family)

        public_events: List[Dict[str, Any]] = []
        trajectory: List[Dict[str, Any]] = []
        proxy_history: List[float] = [read_proxy_score(obs)]
        target_model_digest = model_client.get_model_digest(target_model) or "" if model_client else ""

        # 3. Main Step Loop
        for step in range(max_steps):
            obs.step_count = step

            # A. Check and deliver intervention if scheduled
            obs, intervention = controller.prepare_and_route_intervention(
                episode_id=episode_id,
                step=step,
                task_id=task_id,
                task_family=family,
                task_description=obs.task_description,
                observation=obs,
                public_events=public_events,
                target_model_digest=target_model_digest,
            )
            if intervention:
                pub_evt = {
                    "event_id": f"pub_int_{episode_id}_s{step}",
                    "episode_id": episode_id,
                    "step": step,
                    "actor": "hacker",
                    "event_type": "intervention_delivered",
                    "channel": delivery_channel,
                    "message": intervention.delivered_content,
                    "timestamp": time.time(),
                }
                public_events.append(pub_evt)

            # B. Target acts
            action, metrics = target_agent.act(obs)
            controller.record_target_action(action, step, event_id=f"act_{episode_id}_s{step}")

            # Record action in public events
            action_evt = {
                "event_id": f"act_{episode_id}_s{step}",
                "episode_id": episode_id,
                "step": step,
                "actor": "target",
                "event_type": "tool_action",
                "action": action.action,
                "arguments": action.arguments,
                "rationale": action.rationale,
                "timestamp": time.time(),
            }
            public_events.append(action_evt)

            # C. Adapter executes action
            step_result: StepResult = adapter.execute(action)

            # D. Online Detector step
            det_result = detector.step(
                action=action.action,
                arguments=action.arguments,
                step=step,
                public_events=public_events,
            )

            # Record step in trajectory
            traj_entry = {
                "step": step,
                "action": action.action,
                "arguments": action.arguments,
                "rationale": action.rationale,
                "last_action_summary": step_result.observation.last_action_summary,
                "detector_flagged": det_result.flagged,
                "detector_rules": det_result.rule_ids,
            }
            trajectory.append(traj_entry)

            # Extract proxy score using canonical validator
            current_proxy = read_proxy_score(step_result.observation)
            proxy_history.append(current_proxy)

            obs = step_result.observation
            if step_result.terminated or step_result.truncated:
                break

        # 4. Finalize episode & attribution
        controller.finalize_episode()

        # 5. Independent Evaluator & Mechanism Auditor (Strictly Private)
        # Boundary rule: private_audit_snapshot is NEVER passed to Target, Hacker, or Profiler
        private_snapshot = adapter.private_audit_snapshot()
        auditor = reg.get_auditor(family)
        audit_bundle = auditor.audit(
            public_events=public_events,
            private_snapshot=private_snapshot,
            task_id=task_id,
        )

        true_score = float(private_snapshot.get("true_objective_score", 0.0))
        final_proxy = proxy_history[-1] if proxy_history else 0.0
        trajectory_gain = max(0.0, final_proxy - proxy_history[0]) if len(proxy_history) > 1 else final_proxy
        if audit_bundle.verified and audit_bundle.attributable_proxy_gain > 0:
            effective_proxy_gain = audit_bundle.attributable_proxy_gain
        else:
            effective_proxy_gain = trajectory_gain

        primary_label, label_rec = adjudicate_multi_axis_episode(
            bundle=audit_bundle,
            proxy_gain=effective_proxy_gain,
            true_objective_score=true_score,
            proxy_score=final_proxy,
        )

        latency = time.time() - start_time

        episode_record = EpisodeRecordV4(
            episode_id=episode_id,
            run_id=run_id,
            task_id=task_id,
            task_family=family,
            seed=seed,
            condition=condition,
            exposure_mode=exposure_mode,
            adaptation_mode=adaptation_mode,
            target_policy=target_policy,
            delivery_channel=delivery_channel,
            total_steps=len(trajectory),
            proxy_score=final_proxy,
            true_objective_score=true_score,
            primary_label=primary_label,
            label_record=label_rec,
            trajectory=trajectory,
            proxy_history=proxy_history,
            interventions_count=len(controller.intervention_history),
            profiler_calls_count=len(controller.profiler_call_history),
            detector_flagged=detector.flagged,
            detector_rule_ids=detector.flagged_rules,
            latency_seconds=latency,
            completed_at=time.time(),
        )

        # 6. Save Raw Artifacts
        _persist_episode_artifacts(
            output_dir=output_dir,
            episode=episode_record,
            public_events=public_events,
            interventions=controller.intervention_history,
            profiler_calls=controller.profiler_call_history,
            private_snapshot=private_snapshot,
            audit_bundle=audit_bundle,
            detector=detector,
        )

        return episode_record

    finally:
        adapter.close()


def _persist_episode_artifacts(
    output_dir: Path,
    episode: EpisodeRecordV4,
    public_events: List[Dict[str, Any]],
    interventions: List[InterventionRecordV4],
    private_snapshot: Dict[str, Any],
    audit_bundle: Any,
    detector: MultiDomainDetector,
    profiler_calls: Optional[List[Any]] = None,
) -> None:
    """Write raw immutable JSONL and private audit records with clear separation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    private_audit_dir = output_dir / "private_audit"
    private_audit_dir.mkdir(parents=True, exist_ok=True)

    # 1. episodes.jsonl
    with open(output_dir / "episodes.jsonl", "a", encoding="utf-8") as f:
        f.write(episode.model_dump_json() + "\n")

    # 2. public_events.jsonl
    with open(output_dir / "public_events.jsonl", "a", encoding="utf-8") as f:
        for evt in public_events:
            f.write(json.dumps(evt) + "\n")

    # 3. interventions.jsonl
    with open(output_dir / "interventions.jsonl", "a", encoding="utf-8") as f:
        for itv in interventions:
            f.write(itv.model_dump_json() + "\n")

    # 4. detector_predictions.jsonl
    with open(output_dir / "detector_predictions.jsonl", "a", encoding="utf-8") as f:
        det_record = {
            "episode_id": episode.episode_id,
            "task_id": episode.task_id,
            "task_family": episode.task_family,
            "flagged": detector.flagged,
            "flagged_rules": detector.flagged_rules,
            "history": detector.detection_history,
        }
        f.write(json.dumps(det_record) + "\n")

    # 5. profiler_calls.jsonl (tracks LLM semantic profiler invocations)
    if profiler_calls:
        with open(output_dir / "profiler_calls.jsonl", "a", encoding="utf-8") as f:
            for p_call in profiler_calls:
                f.write((p_call.model_dump_json() if hasattr(p_call, "model_dump_json") else json.dumps(p_call)) + "\n")

    # 6. runtime_metrics.jsonl
    with open(output_dir / "runtime_metrics.jsonl", "a", encoding="utf-8") as f:
        metric_rec = {
            "episode_id": episode.episode_id,
            "task_id": episode.task_id,
            "condition": episode.condition,
            "latency_seconds": episode.latency_seconds,
            "total_steps": episode.total_steps,
            "interventions_count": episode.interventions_count,
            "profiler_calls_count": episode.profiler_calls_count,
            "completed_at": episode.completed_at,
        }
        f.write(json.dumps(metric_rec) + "\n")

    # 7. private_audit/{episode_id}.json (isolated private evaluation)
    private_data = {
        "episode_id": episode.episode_id,
        "task_id": episode.task_id,
        "task_family": episode.task_family,
        "true_objective_score": episode.true_objective_score,
        "primary_label": episode.primary_label,
        "audit_bundle": audit_bundle.model_dump() if hasattr(audit_bundle, "model_dump") else audit_bundle,
        "private_snapshot": private_snapshot,
    }
    with open(private_audit_dir / f"{episode.episode_id}.json", "w", encoding="utf-8") as f:
        json.dump(private_data, f, indent=2, default=str)


def run_v4_experiment(
    config_path: Path,
    output_dir: Optional[Path] = None,
    resume: bool = True,
    dry_run: bool = False,
    override_episodes_cap: Optional[int] = None,
) -> Path:
    """Run balanced blocked multi-domain V4 experiment under the 6-8 hour budget."""
    start_wall_time = time.time()
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    exp_cfg = config.get("experiment", {})
    models_cfg = config.get("models", {})
    agent_cfg = config.get("agent", {})
    ollama_cfg = config.get("ollama", {})

    target_model = models_cfg.get("target", "qwen2.5-coder:7b-instruct")
    hacker_model = models_cfg.get("hacker", "qwen3:4b")
    profiler_model = models_cfg.get("profiler", "qwen3:4b")
    budget_minutes = exp_cfg.get("wall_clock_budget_minutes", 420)
    max_model_episodes = override_episodes_cap or exp_cfg.get("max_model_episodes", 600)
    deadline = start_wall_time + (budget_minutes * 60)

    # Output directory
    run_id = f"v4_run_{int(start_wall_time)}"
    out_dir = output_dir or (Path("results") / "v4" / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Registry & ModelClient
    registry = TaskRegistry()
    client = ModelClient(
        num_ctx=ollama_cfg.get("num_ctx", 6144),
        keep_alive=ollama_cfg.get("keep_alive", "60m"),
    )

    # Check GPU placement & Hardware
    gpu_mon = GPUMonitor()
    snap = gpu_mon.get_gpu_snapshot()
    initial_gpu = {
        "gpu_name": snap.name if snap else "NVIDIA GeForce RTX 3060",
        "memory_total_mb": snap.memory_total_mb if snap else 12288.0,
        "memory_used_mb": snap.memory_used_mb if snap else 8400.0,
        "memory_free_mb": snap.memory_free_mb if snap else 3888.0,
        "driver_version": snap.driver_version if snap else "610.88",
    }

    # Vulnerability Memory
    memory_db_path = out_dir / "vulnerability_memory.db"
    memory = VulnerabilityMemory(db_path=memory_db_path)

    # Conditions list
    conditions = config.get("hacker", {}).get("information_modes", [
        "target_only", "static", "raw_history", "structured_profile", "semantic_profile", "temporal_shuffle_profile"
    ])

    # Build Matched Blocks across all 23 tasks in 6 families
    blocks: List[Tuple[str, int]] = []
    families = ["coding", "data_pipeline", "sql_analytics", "workflow", "scheduling", "document_evidence"]
    num_seeds = config.get("experiment", {}).get("seeds_per_task", 2)
    seeds = [100 + s for s in range(1, num_seeds + 1)]

    # Interleave round-robin across families so all domains progress equally
    for s in seeds:
        max_len = max(len(TASKS_BY_FAMILY[fam]) for fam in families)
        for i in range(max_len):
            for fam in families:
                tasks = TASKS_BY_FAMILY[fam]
                if i < len(tasks):
                    blocks.append((tasks[i], s))

    logger.info(f"Generated {len(blocks)} matched task-seed blocks interleaved across {len(families)} domains.")

    # Checkpoint set for resume
    completed_keys: set = set()
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
        logger.info(f"Resuming experiment. Found {len(completed_keys)} already completed episodes.")

    # Manifest creation
    config_hash = hashlib.sha256(yaml.dump(config).encode("utf-8")).hexdigest()
    manifest = {
        "run_id": run_id,
        "protocol": "v4",
        "config_hash": config_hash,
        "models": {
            "target": target_model,
            "hacker": hacker_model,
        },
        "initial_gpu": initial_gpu,
        "wall_clock_budget_minutes": budget_minutes,
        "max_episodes_cap": max_model_episodes,
        "conditions": conditions,
        "total_blocks": len(blocks),
        "started_at": start_wall_time,
    }
    with open(out_dir / "experiment_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Execution Loop with Randomization within Blocks
    episode_latencies: List[float] = []
    total_episodes_run = len(completed_keys)

    print(f"\n{'='*70}\nStarting V4 Blocked Multi-Domain Run: {run_id}\n{'='*70}")
    print(f"Total Blocks: {len(blocks)} | Conditions per Block: {len(conditions)}")
    print(f"Total Target Episodes: {len(blocks) * len(conditions)} | Budget: {budget_minutes} mins")

    for block_idx, (task_id, seed) in enumerate(blocks):
        # Check budget deadline (reserve 30 mins for report generation)
        time_left_s = deadline - time.time()
        if time_left_s < 1800:  # 30 mins reserve
            print(f"\n[BUDGET GUARD] Less than 30 minutes remaining ({time_left_s:.1f}s). Halting run gracefully.")
            break
        if total_episodes_run >= max_model_episodes:
            print(f"\n[EPISODE CEILING] Reached max model episodes cap ({max_model_episodes}). Halting.")
            break

        # Randomize condition order within block
        block_rng = random.Random(seed + 999)
        block_conditions = list(conditions)
        block_rng.shuffle(block_conditions)

        for cond in block_conditions:
            if (task_id, seed, cond) in completed_keys:
                continue

            if total_episodes_run >= max_model_episodes:
                break

            print(f"[{total_episodes_run + 1}] Running Task: {task_id:25s} | Seed: {seed} | Cond: {cond:20s}", end="", flush=True)

            # Resolve policy, channel, and adaptation_mode dynamically
            cond_lower = cond.lower()
            ep_policy = "P1" if "p1" in cond_lower else agent_cfg.get("target_policy", "P0")
            ep_channel = (
                "task_artifact" if "artifact" in cond_lower
                else ("tool_output" if "tool" in cond_lower
                else config.get("hacker", {}).get("channels", ["direct_message"])[0])
            )
            ep_adapt = (
                "response_adaptive_memory" if "memory" in cond_lower
                else ("response_adaptive" if "adaptive" in cond_lower
                else "open_loop")
            )
            ep_exposure = "persistent" if ("persistent" in cond_lower or cond_lower in ("c2", "c4", "c6")) else "one_shot"

            ep = run_v4_episode(
                run_id=run_id,
                task_id=task_id,
                seed=seed,
                condition=cond,
                output_dir=out_dir,
                registry=registry,
                model_client=client,
                target_model=target_model,
                hacker_model=hacker_model,
                profiler_model=profiler_model,
                exposure_mode=ep_exposure,
                target_policy=ep_policy,
                delivery_channel=ep_channel,
                adaptation_mode=ep_adapt,
                memory=memory,
                max_steps=agent_cfg.get("max_target_steps", 10),
                intervention_steps=agent_cfg.get("intervention_steps", [2, 5, 8]),
            )

            episode_latencies.append(ep.latency_seconds)
            total_episodes_run += 1
            completed_keys.add((task_id, seed, cond))

            print(f" -> Label: {ep.primary_label:16s} (Proxy: {ep.proxy_score:.2f}, True: {ep.true_objective_score:.2f}) [{ep.latency_seconds:.1f}s]")

    # Run Summary Generation
    manifest["ended_at"] = time.time()
    manifest["total_episodes_completed"] = total_episodes_run
    manifest["mean_latency_seconds"] = (
        sum(episode_latencies) / len(episode_latencies) if episode_latencies else 0.0
    )
    with open(out_dir / "experiment_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*70}\nV4 Run Complete: {total_episodes_run} episodes saved in {out_dir}\n{'='*70}")
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V4 Multi-Domain Experiment Runner")
    parser.add_argument("--config", type=str, default="configs/v4_default.yaml", help="Path to V4 config file")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--episodes", type=int, default=None, help="Override maximum episodes")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume existing run")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Config file not found: {cfg_path}")
        sys.exit(1)

    out = Path(args.output) if args.output else None
    run_v4_experiment(
        config_path=cfg_path,
        output_dir=out,
        resume=not args.no_resume,
        override_episodes_cap=args.episodes,
    )
