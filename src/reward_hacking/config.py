import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


class OllamaConfig(BaseModel):
    base_url: str = Field(default="http://localhost:11434")
    num_ctx: int = Field(default=6144)
    keep_alive: str = Field(default="60m")
    request_timeout_s: int = Field(default=180)


class TargetConfig(BaseModel):
    model: str = Field(default="qwen2.5-coder:7b-instruct")
    policy: str = Field(default="permissive_v2")  # permissive_v2 (P0) or integrity_aware (P1)
    temperature: float = Field(default=0.4)
    max_tokens: int = Field(default=1024)
    max_steps: int = Field(default=10)


class HackerConfig(BaseModel):
    model: str = Field(default="qwen3:4b")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=350)
    max_interventions: int = Field(default=3)
    schedule: List[int] = Field(default_factory=lambda: [2, 5, 8])
    delivery_mode: str = Field(default="one_shot")
    attribution_window_target_actions: int = Field(default=2)
    disable_thinking: bool = Field(default=True)


class HackerInformationConfig(BaseModel):
    modes: List[str] = Field(
        default_factory=lambda: [
            "static",
            "raw_history",
            "structured_profile",
            "semantic_profile",
            "shuffled_profile",
        ]
    )
    max_public_events: int = Field(default=12)
    max_profile_tokens: int = Field(default=450)
    max_attack_tokens: int = Field(default=350)
    require_evidence_event_ids: bool = Field(default=True)
    forbid_private_evaluation_fields: bool = Field(default=True)


class ProfilerConfig(BaseModel):
    enabled: bool = Field(default=True)
    update_at_intervention_only: bool = Field(default=True)
    schema_version: int = Field(default=1)
    require_unknowns_field: bool = Field(default=True)
    max_inference_confidence: float = Field(default=0.95)


class SandboxConfig(BaseModel):
    mode: Literal["docker", "restricted_subprocess"] = Field(default="restricted_subprocess")
    allow_restricted: bool = Field(default=True)
    timeout_seconds: int = Field(default=10)
    docker_image: str = Field(default="python:3.11-slim")
    docker_memory: str = Field(default="256m")
    docker_cpus: float = Field(default=1.0)


class ExperimentConfig(BaseModel):
    time_budget_minutes: int = Field(default=135)
    base_seed: int = Field(default=20261102)
    tasks: List[str] = Field(
        default_factory=lambda: [
            "task_001", "task_002", "task_003", "task_004",
            "task_005", "task_006", "task_007", "task_008"
        ]
    )


class AppConfig(BaseModel):
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    target: TargetConfig = Field(default_factory=TargetConfig)
    hacker: HackerConfig = Field(default_factory=HackerConfig)
    hacker_information: HackerInformationConfig = Field(default_factory=HackerInformationConfig)
    profiler: ProfilerConfig = Field(default_factory=ProfilerConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    preset: str = Field(default="primary")
    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2])


def load_config(
    config_path: Optional[Path] = None,
    env_file: Optional[Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> AppConfig:
    """Load configuration from YAML and environment variables, with optional programmatic overrides."""
    root_dir = Path(__file__).resolve().parents[2]

    # 1. Load .env
    if env_file is None:
        default_env = root_dir / ".env"
        if default_env.exists():
            load_dotenv(default_env, override=True)
    elif env_file.exists():
        load_dotenv(env_file, override=True)

    # 2. Load YAML if present
    yaml_data: Dict[str, Any] = {}
    if config_path is None:
        default_yaml = root_dir / "configs" / "default.yaml"
        if default_yaml.exists():
            config_path = default_yaml

    if config_path and config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}

    # Preset mappings per Section 4.1
    model_presets = {
        "primary": {
            "target": "qwen2.5-coder:7b-instruct",
            "hacker": "qwen3:4b",
        },
        "alt1": {
            "target": "qwen2.5-coder:7b-instruct",
            "hacker": "llama3.2:3b",
        },
        "alt2": {
            "target": "qwen2.5-coder:3b-instruct",
            "hacker": "qwen3:4b",
        },
        "fallback_a": {
            "target": "qwen2.5-coder:3b-instruct",
            "hacker": "qwen3:4b",
        },
    }

    preset_name = os.getenv("PRESET", yaml_data.get("models", {}).get("preset", "primary"))

    ollama_cfg = yaml_data.get("ollama", {})
    target_cfg = yaml_data.get("target", {})
    hacker_cfg = yaml_data.get("hacker", {})
    sandbox_cfg = yaml_data.get("sandbox", {})
    exp_cfg = yaml_data.get("experiment", {})

    # Apply preset defaults if not explicitly set
    if preset_name in model_presets:
        if "model" not in target_cfg and not os.getenv("TARGET_MODEL"):
            target_cfg["model"] = model_presets[preset_name]["target"]
        if "model" not in hacker_cfg and not os.getenv("HACKER_MODEL"):
            hacker_cfg["model"] = model_presets[preset_name]["hacker"]

    # Environment variable overrides
    if os.getenv("OLLAMA_BASE_URL"):
        ollama_cfg["base_url"] = os.getenv("OLLAMA_BASE_URL")
    if os.getenv("NUM_CTX"):
        ollama_cfg["num_ctx"] = int(os.getenv("NUM_CTX"))
    if os.getenv("KEEP_ALIVE"):
        ollama_cfg["keep_alive"] = os.getenv("KEEP_ALIVE")
    if os.getenv("REQUEST_TIMEOUT_S"):
        ollama_cfg["request_timeout_s"] = int(os.getenv("REQUEST_TIMEOUT_S"))

    if os.getenv("TARGET_MODEL"):
        target_cfg["model"] = os.getenv("TARGET_MODEL")
    if os.getenv("TARGET_TEMPERATURE"):
        target_cfg["temperature"] = float(os.getenv("TARGET_TEMPERATURE"))
    if os.getenv("TARGET_MAX_TOKENS"):
        target_cfg["max_tokens"] = int(os.getenv("TARGET_MAX_TOKENS"))
    if os.getenv("MAX_TARGET_STEPS"):
        target_cfg["max_steps"] = int(os.getenv("MAX_TARGET_STEPS"))

    if os.getenv("HACKER_MODEL"):
        hacker_cfg["model"] = os.getenv("HACKER_MODEL")
    if os.getenv("HACKER_TEMPERATURE"):
        hacker_cfg["temperature"] = float(os.getenv("HACKER_TEMPERATURE"))
    if os.getenv("HACKER_MAX_TOKENS"):
        hacker_cfg["max_tokens"] = int(os.getenv("HACKER_MAX_TOKENS"))
    if os.getenv("MAX_HACKER_INTERVENTIONS"):
        hacker_cfg["max_interventions"] = int(os.getenv("MAX_HACKER_INTERVENTIONS"))

    if os.getenv("SANDBOX_MODE"):
        sandbox_cfg["mode"] = os.getenv("SANDBOX_MODE")
    if os.getenv("TIME_BUDGET_MINUTES"):
        exp_cfg["time_budget_minutes"] = int(os.getenv("TIME_BUDGET_MINUTES"))
    if os.getenv("BASE_SEED"):
        exp_cfg["base_seed"] = int(os.getenv("BASE_SEED"))

    if overrides:
        for k, v in overrides.items():
            if k == "preset":
                preset_name = v
                if v in model_presets:
                    target_cfg["model"] = model_presets[v]["target"]
                    hacker_cfg["model"] = model_presets[v]["hacker"]
            elif k == "target_model":
                target_cfg["model"] = v
            elif k == "hacker_model":
                hacker_cfg["model"] = v
            elif k == "sandbox_mode":
                sandbox_cfg["mode"] = v
            elif k == "allow_restricted":
                sandbox_cfg["allow_restricted"] = v
            elif k == "time_budget_minutes":
                exp_cfg["time_budget_minutes"] = int(v)
            elif k == "base_seed":
                exp_cfg["base_seed"] = int(v)
            elif k == "tasks" and isinstance(v, list):
                exp_cfg["tasks"] = v

    return AppConfig(
        ollama=OllamaConfig(**ollama_cfg),
        target=TargetConfig(**target_cfg),
        hacker=HackerConfig(**hacker_cfg),
        sandbox=SandboxConfig(**sandbox_cfg),
        experiment=ExperimentConfig(**exp_cfg),
        preset=preset_name,
        project_root=root_dir,
    )
