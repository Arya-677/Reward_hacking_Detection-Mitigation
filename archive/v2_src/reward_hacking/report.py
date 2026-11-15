import argparse
import sys
from pathlib import Path
from reward_hacking.config import load_config
from reward_hacking.reporting import generate_experiment_report
from reward_hacking.storage import ExperimentStorage


def main():
    parser = argparse.ArgumentParser(description="Generate experiment report from recorded runs")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--latest", action="store_true", help="Report on the most recent experiment run")
    group.add_argument("--experiment", type=str, help="Report on a specific experiment ID")

    args = parser.parse_args()
    config = load_config()
    runs_dir = config.project_root / "runs"

    if not runs_dir.exists():
        print(f"Error: No runs directory found at {runs_dir}")
        sys.exit(1)

    if args.latest:
        run_dirs = [p for p in runs_dir.iterdir() if p.is_dir() and (p / "experiment.db").exists()]
        if not run_dirs:
            print("Error: No experiment runs with experiment.db found.")
            sys.exit(1)
        # Sort by modification time
        target_dir = max(run_dirs, key=lambda p: p.stat().st_mtime)
        experiment_id = target_dir.name
    else:
        experiment_id = args.experiment
        target_dir = runs_dir / experiment_id
        if not target_dir.exists():
            print(f"Error: Experiment '{experiment_id}' does not exist at {target_dir}")
            sys.exit(1)

    print(f"Generating report for experiment: {experiment_id} ({target_dir})")
    storage = ExperimentStorage(target_dir)
    episodes = storage.get_all_episodes()

    if not episodes:
        print(f"Error: No completed episodes recorded in {target_dir / 'experiment.db'}")
        sys.exit(1)

    report_path = generate_experiment_report(
        experiment_id=experiment_id,
        episodes=episodes,
        experiment_dir=target_dir,
    )

    print(f"\nReport successfully generated at: {report_path}")
    print(f"Summary JSON and charts updated in: {target_dir}")


if __name__ == "__main__":
    main()
