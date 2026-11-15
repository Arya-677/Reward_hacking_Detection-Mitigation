import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from reward_hacking.config import load_config
from reward_hacking.experiment_runner import ExperimentRunner
from reward_hacking.storage import ExperimentStorage


def run_calibration(
    config_path: Path = None,
    allow_restricted: bool = True,
    dry_run: bool = False,
    experiment_id: str = None,
) -> Dict[str, Any]:
    """Execute v2 calibration protocol (Section 10.1).
    
    - target_only on all 8 tasks (1 seed each) = 8 episodes
    - target_scripted_explicit on all 8 tasks (1 seed each) = 8 episodes
    - target_adaptive_hacker on 3 tasks (1 seed each) = 3 episodes
    Saves calibration.json into runs/<calibration_id>/calibration.json.
    """
    config = load_config(config_path=config_path)
    if allow_restricted:
        config.sandbox.allow_restricted = True

    if experiment_id is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        experiment_id = f"v2_calibration_{timestamp}"

    storage = ExperimentStorage(config.project_root / "runs" / experiment_id)
    hardware_info = {
        "sandbox_mode": config.sandbox.mode,
        "allow_restricted": config.sandbox.allow_restricted,
    }
    storage.record_experiment_meta(experiment_id, config.model_dump(mode="json"), hardware_info)
    storage.save_config(config.model_dump(mode="json"))

    runner = ExperimentRunner(config, storage, dry_run=dry_run)
    runner.sandbox.print_security_banner()

    tasks = list(config.experiment.tasks)
    base_seed = config.experiment.base_seed

    print("================================================================================")
    print(f"STARTING V2 CALIBRATION RUN: {experiment_id}")
    print(f"Tasks: {tasks} | Model Preset: {config.preset}")
    print(f"Target: {config.target.model} | Hacker: {config.hacker.model}")
    print("================================================================================\n")

    t0_total = time.perf_counter()
    episodes: List[Dict[str, Any]] = []

    # 1. target_only on all 8 tasks (8 episodes)
    print("--- CALIBRATION STEP 1: target_only (8 tasks) ---")
    durations_target_only: List[float] = []
    target_call_latencies: List[float] = []

    for idx, t_id in enumerate(tasks, start=1):
        ep_seed = base_seed + idx * 101
        ep_id = f"calib_only_{t_id}_{ep_seed}"
        rec = runner.run_episode(
            experiment_id=experiment_id,
            episode_id=ep_id,
            task_id=t_id,
            condition="target_only",
            seed=ep_seed,
        )
        episodes.append(rec)
        durations_target_only.append(rec["duration_seconds"])
        for step in rec.get("trajectory", []):
            if step.get("target_latency"):
                target_call_latencies.append(step["target_latency"])

    # 2. target_scripted_explicit on all 8 tasks (8 episodes)
    print("\n--- CALIBRATION STEP 2: target_scripted_explicit (8 tasks) ---")
    durations_scripted: List[float] = []
    for idx, t_id in enumerate(tasks, start=1):
        ep_seed = base_seed + idx * 101
        ep_id = f"calib_scripted_{t_id}_{ep_seed}"
        rec = runner.run_episode(
            experiment_id=experiment_id,
            episode_id=ep_id,
            task_id=t_id,
            condition="target_scripted_explicit",
            seed=ep_seed,
        )
        episodes.append(rec)
        durations_scripted.append(rec["duration_seconds"])
        for step in rec.get("trajectory", []):
            if step.get("target_latency"):
                target_call_latencies.append(step["target_latency"])

    # 3. target_adaptive_hacker on 3 tasks (3 episodes)
    print("\n--- CALIBRATION STEP 3: target_adaptive_hacker (3 tasks) ---")
    durations_adaptive: List[float] = []
    hacker_call_latencies: List[float] = []
    for idx, t_id in enumerate(tasks[:3], start=1):
        ep_seed = base_seed + idx * 101
        ep_id = f"calib_adaptive_{t_id}_{ep_seed}"
        rec = runner.run_episode(
            experiment_id=experiment_id,
            episode_id=ep_id,
            task_id=t_id,
            condition="target_adaptive_hacker",
            seed=ep_seed,
        )
        episodes.append(rec)
        durations_adaptive.append(rec["duration_seconds"])
        for step in rec.get("trajectory", []):
            if step.get("target_latency"):
                target_call_latencies.append(step["target_latency"])

    total_wall_s = time.perf_counter() - t0_total

    # Compute Latency Stats
    mean_s_target_only = sum(durations_target_only) / len(durations_target_only) if durations_target_only else 20.0
    mean_s_scripted = sum(durations_scripted) / len(durations_scripted) if durations_scripted else 20.0
    mean_s_adaptive = sum(durations_adaptive) / len(durations_adaptive) if durations_adaptive else 30.0

    mean_min_target_only = mean_s_target_only / 60.0
    mean_min_adaptive = mean_s_adaptive / 60.0
    # Informed static estimate is roughly between target_only and adaptive
    mean_min_static = (mean_min_target_only + mean_min_adaptive) / 2.0

    # Section 11.1 Sizing Formula:
    # cost_core_block = est_min(A) + est_min(static) + est_min(adaptive)
    # N_core = floor(0.65 * MAIN_BUDGET_MINUTES / cost_core_block), capped at 40.
    budget_min = config.experiment.time_budget_minutes
    cost_core_block = mean_min_target_only + mean_min_static + mean_min_adaptive
    computed_n_core = math.floor(0.65 * budget_min / max(0.01, cost_core_block))
    capped_n_core = min(40, max(5, computed_n_core))

    calib_data = {
        "experiment_id": experiment_id,
        "calibration_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_wall_seconds": round(total_wall_s, 2),
        "episodes_count": len(episodes),
        "mean_latency_target_call_s": round(sum(target_call_latencies) / max(1, len(target_call_latencies)), 2),
        "mean_episode_seconds": {
            "target_only": round(mean_s_target_only, 2),
            "target_scripted_explicit": round(mean_s_scripted, 2),
            "target_adaptive_hacker": round(mean_s_adaptive, 2),
        },
        "mean_episode_minutes": {
            "target_only": round(mean_min_target_only, 3),
            "target_static_hacker_est": round(mean_min_static, 3),
            "target_adaptive_hacker": round(mean_min_adaptive, 3),
        },
        "cost_core_block_minutes": round(cost_core_block, 3),
        "main_budget_minutes": budget_min,
        "computed_n_core": computed_n_core,
        "recommended_n_core": capped_n_core,
        "episodes": [e["episode_id"] for e in episodes],
    }

    calib_file = storage.experiment_dir / "calibration.json"
    with open(calib_file, "w", encoding="utf-8") as f:
        json.dump(calib_data, f, indent=2)

    print("\n================================================================================")
    print(f"CALIBRATION COMPLETE: {experiment_id}")
    print(f"Target-Only: {mean_s_target_only:.1f}s/ep | Scripted: {mean_s_scripted:.1f}s/ep | Adaptive: {mean_s_adaptive:.1f}s/ep")
    print(f"Cost per Core Block: {cost_core_block:.2f} min -> Sized N_core = {capped_n_core} blocks")
    print(f"Saved calibration data to: {calib_file}")
    print("================================================================================\n")

    return calib_data


def main():
    parser = argparse.ArgumentParser(description="Run v2 calibration protocol")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--allow-restricted-sandbox", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--experiment-id", type=str, default=None)
    args = parser.parse_args()

    cfg_path = Path(args.config) if args.config else None
    run_calibration(
        config_path=cfg_path,
        allow_restricted=args.allow_restricted_sandbox,
        dry_run=args.dry_run,
        experiment_id=args.experiment_id,
    )


if __name__ == "__main__":
    main()
