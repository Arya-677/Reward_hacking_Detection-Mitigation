import json
from pathlib import Path

calib_dir = Path("runs/v2_calibration_20261103_013443")
episodes_file = calib_dir / "episodes.jsonl"
episodes = [json.loads(line) for line in episodes_file.read_text(encoding="utf-8").splitlines() if line.strip()]

print(f"Total calibration episodes: {len(episodes)}")
for ep in episodes:
    print(f"[{ep['condition']}] {ep['episode_id']} | TrueSuccess={ep['true_success']} | Label={ep['ground_truth_label']} | Cause={ep['failure_cause']}")
    if ep['condition'] == 'target_scripted_explicit':
        for s in ep['trajectory']:
            print(f"   Step {s['step']}: {s['action']} ({s['path']}) -> msg: {s['message'][:60]}")
