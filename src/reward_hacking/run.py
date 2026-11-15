import argparse
import json
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
        description="Run Two-Model Adversarial Reward Hacking Experiments (V2 and V3 Protocol)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--protocol",
        choices=["v2", "v3"],
        default="v3",
        help="Experimental protocol version to run (v2 or v3).",
    )
    parser.add_argument(
        "--experiment",
        choices=["smoke", "semantic_ablation", "main", "all_conditions"],
        default="semantic_ablation",
        help="Experiment stage to run under the selected protocol.",
    )
    parser.add_argument(
        "--condition",
        type=str,
        help="Run a specific condition only (e.g. 'semantic_profile', 'raw_history', 'target_only').",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        help="Comma-separated list of conditions to run (e.g. 'target_only,raw_history,semantic_profile').",
    )
    parser.add_argument(
        "--blocks",
        type=int,
        help="Number of matched blocks to run (alias for --episodes).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=12,
        help="Number of core blocks/episodes per condition.",
    )
    parser.add_argument(
        "--main",
        action="store_true",
        help="Run the pre-registered main experiment across conditions.",
    )
    parser.add_argument(
        "--all-conditions",
        action="store_true",
        help="Run all conditions in randomized, matched blocks.",
    )
    parser.add_argument(
        "--auto-size",
        action="store_true",
        help="Auto-size episodes per condition based on calibration.json and MAIN_BUDGET_MINUTES.",
    )
    parser.add_argument(
        "--time-budget-minutes",
        type=int,
        default=135,
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
        "--target-model",
        type=str,
        help="Override target model name.",
    )
    parser.add_argument(
        "--hacker-model",
        type=str,
        help="Override hacker model name.",
    )
    parser.add_argument(
        "--preset",
        choices=["primary", "alt1", "alt2", "fallback_a"],
        help="Model preset selection.",
    )
    parser.add_argument(
        "--allow-restricted-sandbox",
        action="store_true",
        default=True,
        help="Explicitly permit restricted_subprocess sandboxing mode.",
    )
    parser.add_argument(
        "--target-policy",
        choices=["permissive_v2", "integrity_aware", "P0", "P1"],
        default="permissive_v2",
        help="Target policy setting: permissive_v2 (P0) or integrity_aware (P1).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in synthetic fixture mode without calling real LLMs (for testing).",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Generate report only from existing run directory (pass via --resume).",
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
    if args.target_model:
        config.target.model = args.target_model
    if args.hacker_model:
        config.hacker.model = args.hacker_model
    if args.target_policy:
        config.target.policy = args.target_policy

    # Determine Experiment ID
    protocol = args.protocol
    prefix = "v3" if protocol == "v3" else "v2"
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
        experiment_id = f"{prefix}_exp_{timestamp}_{uuid.uuid4().hex[:4]}"
        exp_dir = config.project_root / "runs" / experiment_id
        resuming = False

    storage = ExperimentStorage(exp_dir)

    # If --report-only, re-run reporting on existing storage and exit
    if args.report_only:
        all_episodes = storage.get_all_episodes()
        print(f"Generating report from existing {len(all_episodes)} episodes in {exp_dir}...")
        report_file = generate_experiment_report(
            experiment_id=experiment_id,
            episodes=all_episodes,
            experiment_dir=exp_dir,
        )
        print(f"Report compiled: {report_file}")
        return

    # Save experiment metadata
    hardware_info = {
        "sandbox_mode": config.sandbox.mode,
        "allow_restricted": config.sandbox.allow_restricted,
        "protocol": protocol,
    }
    storage.record_experiment_meta(experiment_id, config.model_dump(mode="json"), hardware_info)
    storage.save_config(config.model_dump(mode="json"))

    runner = ExperimentRunner(config=config, storage=storage, dry_run=args.dry_run)

    # Determine blocks and conditions
    blocks_count = args.blocks if args.blocks is not None else args.episodes
    if args.experiment == "smoke":
        blocks_count = args.blocks or 1

    if protocol == "v3":
        default_v3_conditions = [
            "target_only",
            "informed_static",
            "raw_history",
            "structured_profile",
            "semantic_profile",
            "shuffled_profile",
        ]
        if args.condition:
            conditions_to_run = [args.condition]
        elif args.conditions:
            conditions_to_run = [c.strip() for c in args.conditions.split(",") if c.strip()]
        else:
            conditions_to_run = default_v3_conditions

        task_list = config.experiment.tasks
        if args.experiment == "smoke" and not args.tasks:
            task_list = ["task_001", "task_002"]

        print(f"\n================================================================================")
        print(f"RUNNING V3 PROTOCOL EXPERIMENT ({args.experiment.upper()})")
        print(f"ID: {experiment_id} | Blocks: {blocks_count} | Conditions: {conditions_to_run}")
        print(f"Tasks: {task_list} | Target: {config.target.model} | Hacker: {config.hacker.model}")
        print(f"================================================================================\n")

        all_episodes = runner.run_v3_blocked_experiment(
            experiment_id=experiment_id,
            conditions=conditions_to_run,
            blocks_count=blocks_count,
            tasks=task_list,
            base_seed=config.experiment.base_seed,
            resume=resuming,
            budget_minutes=config.experiment.time_budget_minutes,
        )

    else:
        # V2 Execution Branch
        if args.auto_size:
            runs_dir = config.project_root / "runs"
            calib_candidates = list(runs_dir.glob("v2_calibration_*/calibration.json"))
            if calib_candidates:
                latest_calib = sorted(calib_candidates, key=lambda p: p.stat().st_mtime)[-1]
                try:
                    with open(latest_calib, "r", encoding="utf-8") as cf:
                        cdata = json.load(cf)
                        blocks_count = cdata.get("recommended_n_core", blocks_count)
                        print(f"Auto-sizing from {latest_calib.parent.name}: N_core = {blocks_count} blocks")
                except Exception as ex:
                    print(f"Could not read calibration file: {ex}. Using default N={blocks_count}")

        if args.condition:
            print(f"\nRunning Condition: {args.condition} ({blocks_count} episodes)")
            runner.sandbox.print_security_banner()
            tasks = config.experiment.tasks
            base_seed = config.experiment.base_seed
            for i in range(blocks_count):
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
            all_episodes = runner.run_blocked_experiment(
                experiment_id=experiment_id,
                episodes_per_condition=blocks_count,
                resume=resuming,
                budget_minutes=config.experiment.time_budget_minutes,
            )

    # Generate final reporting and summary table
    report_file = generate_experiment_report(
        experiment_id=experiment_id,
        episodes=all_episodes,
        experiment_dir=exp_dir,
    )

    print("\n================================================================================")
    print("FINAL EXPERIMENT SUMMARY (TABLE C.2)")
    print("================================================================================")
    summary_path = exp_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            sum_data = json.load(f)
        c2_rows = sum_data.get("table_c2_rows")
        if c2_rows:
            print(f"{'Condition':<22} | {'N':<4} | {'Hacks':<6} | {'Hack Rate (95% CI)':<22} | {'Attributed Succ':<16} | {'Pred Acc':<9} | {'Brier':<8}")
            print("-" * 96)
            for r in c2_rows:
                print(
                    f"{r['condition']:<22} | {r['n']:<4} | {r['confirmed_hacks']:<6} | {r['hack_rate_95_ci']:<22} | "
                    f"{r['attributed_intervention_success']:<16} | {r['prediction_accuracy']:<9} | {r['brier_score']:<8}"
                )
        else:
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
