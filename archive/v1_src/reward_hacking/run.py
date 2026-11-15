import argparse
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from reward_hacking.config import load_config
from reward_hacking.experiment_runner import ExperimentRunner
from reward_hacking.reporting import generate_experiment_report
from reward_hacking.storage import ExperimentStorage


def main():
    parser = argparse.ArgumentParser(
        description="Run Two-Model Adversarial Reward Hacking Experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--condition",
        choices=["target_only", "target_static_hacker", "target_adaptive_hacker"],
        help="Run a specific condition only.",
    )
    parser.add_argument(
        "--all-conditions",
        action="store_true",
        help="Run all three conditions in randomized, matched blocks.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes per condition.",
    )
    parser.add_argument(
        "--auto-size",
        action="store_true",
        help="Calibrate latency on 3 episodes and auto-size episodes per condition based on time budget.",
    )
    parser.add_argument(
        "--time-budget-minutes",
        type=int,
        default=150,
        help="Maximum wall-clock execution time limit in minutes.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume an interrupted experiment run by experiment ID.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Base random seed.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        help="Comma-separated list of task IDs to use (e.g. 'task_001,task_002').",
    )
    parser.add_argument(
        "--preset",
        choices=["primary", "fallback_a", "fallback_b"],
        help="Model preset selection.",
    )
    parser.add_argument(
        "--allow-restricted-sandbox",
        action="store_true",
        help="Explicitly permit restricted_subprocess sandboxing mode.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in synthetic fixture mode without calling real LLMs (for testing).",
    )

    args = parser.parse_args()

    # Apply CLI overrides to configuration
    overrides = {}
    if args.preset:
        overrides["preset"] = args.preset
    if args.seed is not None:
        overrides["base_seed"] = args.seed
    if args.time_budget_minutes:
        overrides["time_budget_minutes"] = args.time_budget_minutes
    if args.allow_restricted_sandbox:
        overrides["allow_restricted"] = True
    if args.tasks:
        overrides["tasks"] = [t.strip() for t in args.tasks.split(",") if t.strip()]

    config = load_config(overrides=overrides)

    # Determine Experiment ID
    if args.resume:
        experiment_id = args.resume
        runs_dir = config.project_root / "runs"
        exp_dir = runs_dir / experiment_id
        if not exp_dir.exists():
            print(f"Error: Experiment ID '{experiment_id}' not found at {exp_dir}")
            sys.exit(1)
        resuming = True
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        experiment_id = f"exp_{timestamp}_{uuid.uuid4().hex[:4]}"
        exp_dir = config.project_root / "runs" / experiment_id
        resuming = False

    storage = ExperimentStorage(exp_dir)

    # Save experiment metadata
    hardware_info = {
        "sandbox_mode": config.sandbox.mode,
        "allow_restricted": config.sandbox.allow_restricted,
    }
    storage.record_experiment_meta(experiment_id, config.model_dump(mode="json"), hardware_info)

    runner = ExperimentRunner(config=config, storage=storage, dry_run=args.dry_run)

    # Execute
    if args.condition:
        # Run specific condition
        print(f"\nRunning Condition: {args.condition} ({args.episodes} episodes)")
        runner.sandbox.print_security_banner()
        tasks = config.experiment.tasks
        base_seed = config.experiment.base_seed
        for i in range(args.episodes):
            t_id = tasks[i % len(tasks)]
            s_val = base_seed + i * 17
            ep_id = f"ep_{i+1:03d}_{args.condition}_{uuid.uuid4().hex[:6]}"
            runner.run_episode(
                experiment_id=experiment_id,
                episode_id=ep_id,
                task_id=t_id,
                condition=args.condition,
                seed=s_val,
            )
        all_episodes = storage.get_all_episodes()
    else:
        # Default: all conditions (blocked design)
        all_episodes = runner.run_blocked_experiment(
            experiment_id=experiment_id,
            episodes_per_condition=args.episodes,
            auto_size=args.auto_size,
            resume=resuming,
        )

    # Generate final reporting and summary table
    report_file = generate_experiment_report(
        experiment_id=experiment_id,
        episodes=all_episodes,
        experiment_dir=exp_dir,
    )

    print("\n================================================================================")
    print("FINAL EXPERIMENT SUMMARY TABLE")
    print("================================================================================")
    summary_path = exp_dir / "summary.json"
    if summary_path.exists():
        import json
        with open(summary_path, "r", encoding="utf-8") as f:
            sum_data = json.load(f)
        conds = sum_data.get("conditions", {})
        print(f"{'Condition':<28} | {'Episodes':<8} | {'Confirmed Hacks':<15} | {'Hack Rate':<10} | {'True Success':<12}")
        print("-" * 84)
        for c_k, c_v in conds.items():
            print(
                f"{c_k:<28} | {c_v['n']:<8} | {c_v['confirmed_hack_episodes']:<15} | "
                f"{c_v['confirmed_hack_rate']*100:.1f}%{'':<5} | {c_v['true_success_rate']*100:.1f}%"
            )

    print(f"\nReport generated at: {report_file}")
    print(f"Artifacts saved in: {exp_dir}")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
