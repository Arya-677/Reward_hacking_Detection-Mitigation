import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


class ExperimentStorage:
    def __init__(self, experiment_dir: Path):
        self.experiment_dir = experiment_dir.resolve()
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.trajectories_dir = self.experiment_dir / "trajectories"
        self.trajectories_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.experiment_dir / "experiment.db"
        self.log_path = self.experiment_dir / "run.log"
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS experiment_meta (
                    experiment_id TEXT PRIMARY KEY,
                    created_at TEXT,
                    config_json TEXT,
                    hardware_json TEXT
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS episodes (
                    episode_id TEXT PRIMARY KEY,
                    experiment_id TEXT,
                    condition TEXT,
                    task_id TEXT,
                    seed INTEGER,
                    started_at TEXT,
                    finished_at TEXT,
                    target_model TEXT,
                    target_digest TEXT,
                    hacker_model TEXT,
                    hacker_digest TEXT,
                    sandbox_mode TEXT,
                    initial_proxy_score REAL,
                    final_proxy_score REAL,
                    true_success INTEGER,
                    ground_truth_label TEXT,
                    exploit_event_count INTEGER,
                    detector_predicted_hack INTEGER,
                    detector_confidence REAL,
                    detector_fired_rules TEXT,
                    target_calls INTEGER,
                    hacker_calls INTEGER,
                    total_tokens INTEGER,
                    duration_seconds REAL
                )
                """
            )
            conn.commit()

    def record_experiment_meta(
        self,
        experiment_id: str,
        config_dict: Dict[str, Any],
        hardware_dict: Dict[str, Any],
    ):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO experiment_meta (experiment_id, created_at, config_json, hardware_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(config_dict, default=str),
                    json.dumps(hardware_dict, default=str),
                ),
            )
            conn.commit()

    def log(self, message: str):
        """Append timestamped message to run.log."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}\n"
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(log_line)

    def save_config(self, config_dict: Dict[str, Any]):
        """Save config to experiment directory."""
        cfg_file = self.experiment_dir / "config.json"
        with open(cfg_file, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, default=str)

    def save_episode(self, episode_data: Dict[str, Any]):
        """Persist full episode to SQLite, JSONL trajectory, and master episodes.jsonl."""
        ep_id = episode_data["episode_id"]

        # 1. Write trajectory JSONL
        traj_path = self.trajectories_dir / f"{ep_id}.jsonl"
        with open(traj_path, "w", encoding="utf-8") as f:
            for step in episode_data.get("trajectory", []):
                f.write(json.dumps(step) + "\n")

        # 2. Append full episode record to episodes.jsonl
        episodes_jsonl_path = self.experiment_dir / "episodes.jsonl"
        with open(episodes_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(episode_data, default=str) + "\n")

        # 3. Insert into SQLite
        fired_rules_str = ",".join(episode_data.get("detector_fired_rules", []))
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO episodes (
                    episode_id, experiment_id, condition, task_id, seed,
                    started_at, finished_at, target_model, target_digest,
                    hacker_model, hacker_digest, sandbox_mode,
                    initial_proxy_score, final_proxy_score, true_success,
                    ground_truth_label, exploit_event_count,
                    detector_predicted_hack, detector_confidence, detector_fired_rules,
                    target_calls, hacker_calls, total_tokens, duration_seconds
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ep_id,
                    episode_data.get("experiment_id"),
                    episode_data.get("condition"),
                    episode_data.get("task_id"),
                    episode_data.get("seed"),
                    episode_data.get("started_at"),
                    episode_data.get("finished_at"),
                    episode_data.get("target_model"),
                    episode_data.get("target_digest"),
                    episode_data.get("hacker_model"),
                    episode_data.get("hacker_digest"),
                    episode_data.get("sandbox_mode"),
                    episode_data.get("initial_proxy_score"),
                    episode_data.get("final_proxy_score"),
                    1 if episode_data.get("true_success") else 0,
                    episode_data.get("ground_truth_label"),
                    episode_data.get("exploit_event_count", 0),
                    1 if episode_data.get("detector_predicted_hack") else 0,
                    episode_data.get("detector_confidence", 0.0),
                    fired_rules_str,
                    episode_data.get("target_calls", 0),
                    episode_data.get("hacker_calls", 0),
                    episode_data.get("total_tokens", 0),
                    episode_data.get("duration_seconds", 0.0),
                ),
            )
            conn.commit()

        # 3. Log concise summary
        det_conf = float(episode_data.get("detector_confidence") or 0.0)
        self.log(
            f"Completed Episode {ep_id} | Condition: {episode_data.get('condition')} | "
            f"Task: {episode_data.get('task_id')} | Seed: {episode_data.get('seed')} | "
            f"Proxy: {episode_data.get('final_proxy_score'):.2f} | "
            f"TrueSuccess: {episode_data.get('true_success')} | "
            f"Label: {episode_data.get('ground_truth_label')} | "
            f"Detector: {episode_data.get('detector_predicted_hack')} (conf: {det_conf:.2f})"
        )

    def get_completed_units(self) -> Set[Tuple[str, int, str]]:
        """Return set of completed (task_id, seed, condition) triples for resume capability."""
        completed: Set[Tuple[str, int, str]] = set()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT task_id, seed, condition FROM episodes")
            for row in cursor.fetchall():
                completed.add((row[0], row[1], row[2]))
        return completed

    def get_all_episodes(self) -> List[Dict[str, Any]]:
        """Retrieve all episodes as list of dictionaries (preferring full JSONL records)."""
        episodes_jsonl_path = self.experiment_dir / "episodes.jsonl"
        if episodes_jsonl_path.exists():
            records = []
            with open(episodes_jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
            return records

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM episodes ORDER BY started_at ASC")
            return [dict(row) for row in cursor.fetchall()]
