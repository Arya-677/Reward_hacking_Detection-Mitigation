"""GPU and Hardware Monitoring and Verification Module for V3.

Detects NVIDIA GPU, records VRAM, checks GPU utilization during inference,
verifies model CPU/GPU placement in Ollama, and benchmarks resident vs sequential
model loading strategies.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class GPUSnapshot:
    timestamp: float
    name: str
    memory_total_mb: float
    memory_used_mb: float
    memory_free_mb: float
    utilization_gpu_pct: float
    driver_version: str
    temperature_c: float = 0.0


@dataclass
class OllamaModelPlacement:
    name: str
    model_id: str
    size_bytes: int
    size_vram_bytes: int
    processor_string: str  # e.g. "100% GPU", "40%/60% CPU/GPU", "100% CPU"
    is_fully_gpu: bool
    vram_mb: float
    total_size_mb: float


@dataclass
class HardwareProfile:
    gpu_available: bool
    gpu_name: str
    gpu_uuid: str
    driver_version: str
    cuda_version: str
    total_vram_mb: float
    free_vram_mb: float
    system_ram_gb: float
    free_ram_gb: float
    cpu_name: str
    os_version: str
    ollama_version: str
    models_available: List[str] = field(default_factory=list)


class GPUMonitor:
    """Monitors NVIDIA GPU performance and Ollama model placement."""

    def __init__(self, ollama_host: str = "http://localhost:11434"):
        self.ollama_host = ollama_host.rstrip("/")
        self._sampling = False
        self._sample_thread: Optional[threading.Thread] = None
        self._samples: List[GPUSnapshot] = []

    @staticmethod
    def is_nvidia_smi_available() -> bool:
        return shutil.which("nvidia-smi") is not None

    def get_hardware_profile(self) -> HardwareProfile:
        """Inspect and return current host hardware specifications."""
        gpu_name = "Unavailable"
        gpu_uuid = "Unavailable"
        driver_version = "Unavailable"
        cuda_version = "Unavailable"
        total_vram = 0.0
        free_vram = 0.0
        gpu_avail = False

        if self.is_nvidia_smi_available():
            try:
                cmd = [
                    "nvidia-smi",
                    "--query-gpu=name,gpu_uuid,driver_version,memory.total,memory.free",
                    "--format=csv,noheader,nounits",
                ]
                res = subprocess.check_output(cmd, text=True, timeout=5).strip()
                if res:
                    parts = [p.strip() for p in res.split(",")]
                    gpu_name = parts[0]
                    gpu_uuid = parts[1] if len(parts) > 1 else ""
                    driver_version = parts[2] if len(parts) > 2 else ""
                    total_vram = float(parts[3]) if len(parts) > 3 else 0.0
                    free_vram = float(parts[4]) if len(parts) > 4 else 0.0
                    gpu_avail = True
            except Exception as e:
                logger.warning(f"Failed to query nvidia-smi: {e}")

            try:
                out = subprocess.check_output(["nvidia-smi"], text=True, timeout=5)
                for line in out.splitlines():
                    if "CUDA Version:" in line or "CUDA UMD Version:" in line:
                        cuda_version = line.split("CUDA")[-1].replace("Version:", "").replace("UMD", "").strip().split()[0]
                        break
            except Exception:
                pass

        # System RAM and CPU
        import ctypes
        import platform

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        try:
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            total_ram_gb = round(stat.ullTotalPhys / (1024**3), 2)
            free_ram_gb = round(stat.ullAvailPhys / (1024**3), 2)
        except Exception:
            total_ram_gb = 16.0
            free_ram_gb = 8.0

        cpu_name = platform.processor() or "Unknown CPU"
        os_ver = f"{platform.system()} {platform.release()} ({platform.version()})"

        # Ollama version
        ollama_ver = "Unknown"
        try:
            out = subprocess.check_output(["ollama", "--version"], text=True, timeout=5).strip()
            ollama_ver = out.replace("ollama version is", "").strip()
        except Exception:
            pass

        # Models available
        models_avail = []
        try:
            req = urllib.request.Request(f"{self.ollama_host}/api/tags")
            with urllib.request.urlopen(req, timeout=4) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models_avail = [m["name"] for m in data.get("models", [])]
        except Exception:
            pass

        return HardwareProfile(
            gpu_available=gpu_avail,
            gpu_name=gpu_name,
            gpu_uuid=gpu_uuid,
            driver_version=driver_version,
            cuda_version=cuda_version,
            total_vram_mb=total_vram,
            free_vram_mb=free_vram,
            system_ram_gb=total_ram_gb,
            free_ram_gb=free_ram_gb,
            cpu_name=cpu_name,
            os_version=os_ver,
            ollama_version=ollama_ver,
            models_available=models_avail,
        )

    def get_gpu_snapshot(self) -> Optional[GPUSnapshot]:
        """Fetch an instantaneous snapshot of GPU usage and temperature."""
        if not self.is_nvidia_smi_available():
            return None
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,driver_version,temperature.gpu",
                "--format=csv,noheader,nounits",
            ]
            res = subprocess.check_output(cmd, text=True, timeout=3).strip()
            if not res:
                return None
            p = [x.strip() for x in res.split(",")]
            return GPUSnapshot(
                timestamp=time.time(),
                name=p[0],
                memory_total_mb=float(p[1]),
                memory_used_mb=float(p[2]),
                memory_free_mb=float(p[3]),
                utilization_gpu_pct=float(p[4]),
                driver_version=p[5] if len(p) > 5 else "",
                temperature_c=float(p[6]) if len(p) > 6 else 0.0,
            )
        except Exception as e:
            logger.debug(f"Error reading GPU snapshot: {e}")
            return None

    sample_now = get_gpu_snapshot

    def get_ollama_loaded_models(self) -> List[OllamaModelPlacement]:
        """Inspect running models in Ollama and their processor placement."""
        # 1. First attempt CLI 'ollama ps' to parse the exact PROCESSOR column
        cli_processors: Dict[str, str] = {}
        try:
            out = subprocess.check_output(["ollama", "ps"], text=True, timeout=5)
            lines = out.strip().splitlines()
            if len(lines) > 1:
                header = lines[0]
                # NAME, ID, SIZE, PROCESSOR, UNTIL
                for line in lines[1:]:
                    parts = line.split()
                    if len(parts) >= 4:
                        m_name = parts[0]
                        # Find processor string (e.g. '100% GPU' or '100% CPU')
                        if "100% GPU" in line:
                            cli_processors[m_name] = "100% GPU"
                        elif "100% CPU" in line:
                            cli_processors[m_name] = "100% CPU"
                        else:
                            for p_tok in parts:
                                if "GPU" in p_tok or "CPU" in p_tok:
                                    cli_processors[m_name] = p_tok
                                    break
        except Exception as ex:
            logger.debug(f"Could not parse ollama ps CLI: {ex}")

        # 2. Query HTTP API /api/ps for exact byte sizes
        results: List[OllamaModelPlacement] = []
        try:
            req = urllib.request.Request(f"{self.ollama_host}/api/ps")
            with urllib.request.urlopen(req, timeout=4) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                for m in data.get("models", []):
                    m_name = m.get("name", "")
                    size_b = m.get("size", 0)
                    size_vram_b = m.get("size_vram", 0)
                    proc_str = cli_processors.get(m_name)
                    if not proc_str:
                        if size_b > 0 and size_vram_b >= size_b:
                            proc_str = "100% GPU"
                        elif size_vram_b == 0:
                            proc_str = "100% CPU"
                        else:
                            ratio = size_vram_b / max(1, size_b)
                            proc_str = f"{int(ratio*100)}% GPU"

                    is_gpu = "100% GPU" in proc_str or (size_vram_b > 0 and size_vram_b >= size_b * 0.98)
                    results.append(
                        OllamaModelPlacement(
                            name=m_name,
                            model_id=m.get("digest", "")[:12],
                            size_bytes=size_b,
                            size_vram_bytes=size_vram_b,
                            processor_string=proc_str,
                            is_fully_gpu=is_gpu,
                            vram_mb=round(size_vram_b / (1024**2), 1),
                            total_size_mb=round(size_b / (1024**2), 1),
                        )
                    )
        except Exception as ex:
            logger.warning(f"Error querying Ollama /api/ps: {ex}")

        return results

    def verify_gpu_placement(self, expected_models: List[str]) -> Tuple[bool, str]:
        """Verify that expected models are loaded and placed 100% on GPU."""
        loaded = self.get_ollama_loaded_models()
        loaded_dict = {m.name: m for m in loaded}

        details = []
        all_gpu = True
        for name in expected_models:
            # Check match by prefix or full name
            m = loaded_dict.get(name)
            if not m:
                # Try prefix match e.g. 'qwen2.5-coder:7b-instruct' vs 'qwen2.5-coder:7b-instruct:latest'
                for k, v in loaded_dict.items():
                    if k.startswith(name) or name.startswith(k):
                        m = v
                        break

            if not m:
                details.append(f"Model '{name}' is not currently loaded in memory.")
                all_gpu = False
            else:
                details.append(
                    f"Model '{m.name}': Placement={m.processor_string} (VRAM: {m.vram_mb:.1f} MB / Total: {m.total_size_mb:.1f} MB)"
                )
                if not m.is_fully_gpu:
                    all_gpu = False

        status_msg = "\n".join(details)
        return all_gpu, status_msg

    def start_sampling(self, interval_seconds: float = 0.5):
        """Start asynchronous GPU metrics sampling in background."""
        if self._sampling:
            return
        self._sampling = True
        self._samples.clear()

        def _loop():
            while self._sampling:
                snap = self.get_gpu_snapshot()
                if snap:
                    self._samples.append(snap)
                time.sleep(interval_seconds)

        self._sample_thread = threading.Thread(target=_loop, daemon=True)
        self._sample_thread.start()

    def stop_sampling(self) -> Dict[str, Any]:
        """Stop sampling and compute aggregate GPU statistics."""
        self._sampling = False
        if self._sample_thread:
            self._sample_thread.join(timeout=2.0)
            self._sample_thread = None

        if not self._samples:
            return {
                "sample_count": 0,
                "peak_vram_mb": 0.0,
                "avg_vram_mb": 0.0,
                "peak_gpu_util_pct": 0.0,
                "avg_gpu_util_pct": 0.0,
            }

        vram_vals = [s.memory_used_mb for s in self._samples]
        util_vals = [s.utilization_gpu_pct for s in self._samples]

        return {
            "sample_count": len(self._samples),
            "peak_vram_mb": round(max(vram_vals), 1),
            "avg_vram_mb": round(sum(vram_vals) / len(vram_vals), 1),
            "min_vram_mb": round(min(vram_vals), 1),
            "peak_gpu_util_pct": round(max(util_vals), 1),
            "avg_gpu_util_pct": round(sum(util_vals) / len(util_vals), 1),
        }

    def unload_model(self, model_name: str) -> bool:
        """Explicitly unload a model from memory using Ollama keep_alive=0."""
        try:
            req = urllib.request.Request(
                f"{self.ollama_host}/api/generate",
                data=json.dumps({"model": model_name, "keep_alive": 0}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                resp.read()
            time.sleep(0.5)
            return True
        except Exception as e:
            logger.warning(f"Failed to unload model {model_name}: {e}")
            return False

    def warm_model(self, model_name: str, keep_alive: str = "60m") -> Dict[str, Any]:
        """Warm up a model in GPU memory with an initial lightweight prompt."""
        t0 = time.perf_counter()
        vram_before = (self.get_gpu_snapshot().memory_used_mb if self.get_gpu_snapshot() else 0.0)
        req = urllib.request.Request(
            f"{self.ollama_host}/api/generate",
            data=json.dumps({
                "model": model_name,
                "prompt": "Hello",
                "keep_alive": keep_alive,
                "stream": False,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        load_duration = time.perf_counter() - t0
        vram_after = (self.get_gpu_snapshot().memory_used_mb if self.get_gpu_snapshot() else 0.0)

        # Check placement
        loaded = self.get_ollama_loaded_models()
        placement = next((m for m in loaded if m.name.startswith(model_name) or model_name.startswith(m.name)), None)

        return {
            "model": model_name,
            "load_duration_s": round(load_duration, 2),
            "vram_delta_mb": round(vram_after - vram_before, 1),
            "placement": placement.processor_string if placement else "Unknown",
            "is_gpu": placement.is_fully_gpu if placement else False,
        }


def run_gpu_benchmark_comparison(
    target_model: str = "qwen2.5-coder:7b-instruct",
    hacker_model: str = "qwen3:4b",
    test_prompts_count: int = 2,
) -> Dict[str, Any]:
    """Benchmark Strategy A (Both models resident) vs Strategy B (Sequential loading/unloading)."""
    monitor = GPUMonitor()
    hw = monitor.get_hardware_profile()

    print("================================================================================", flush=True)
    print("V3 GPU BENCHMARK & STRATEGY EVALUATION", flush=True)
    print(f"GPU: {hw.gpu_name} ({hw.total_vram_mb:.0f} MB VRAM) | Driver: {hw.driver_version} | CUDA: {hw.cuda_version}", flush=True)
    print(f"Target: {target_model} | Hacker: {hacker_model}", flush=True)
    print("================================================================================\n", flush=True)

    # -------------------------------------------------------------
    # Strategy A: Both Models Resident
    # -------------------------------------------------------------
    print("--- BENCHMARKING STRATEGY A: Both Models Resident ---", flush=True)
    monitor.start_sampling(interval_seconds=0.2)
    t0_strat_a = time.perf_counter()

    # Load Target
    res_t = monitor.warm_model(target_model, keep_alive="60m")
    print(f"  Target loaded in {res_t['load_duration_s']}s -> Placement: {res_t['placement']}", flush=True)

    # Load Hacker
    res_h = monitor.warm_model(hacker_model, keep_alive="60m")
    print(f"  Hacker loaded in {res_h['load_duration_s']}s -> Placement: {res_h['placement']}", flush=True)

    # Check both placement
    both_gpu, placement_status = monitor.verify_gpu_placement([target_model, hacker_model])
    print(f"  Verification:\n{placement_status}", flush=True)

    # Simulated turn interactions
    t_latencies_a = []
    h_latencies_a = []
    for turn in range(1, test_prompts_count + 1):
        # Target step
        t0 = time.perf_counter()
        req_t = urllib.request.Request(
            f"{monitor.ollama_host}/api/generate",
            data=json.dumps({"model": target_model, "prompt": f"Write a short python function for step {turn}.", "stream": False}).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req_t, timeout=120) as resp:
            resp.read()
        t_latencies_a.append(time.perf_counter() - t0)

        # Hacker step
        t0 = time.perf_counter()
        req_h = urllib.request.Request(
            f"{monitor.ollama_host}/api/generate",
            data=json.dumps({"model": hacker_model, "prompt": f"Suggest a test tweak for step {turn}.", "stream": False}).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req_h, timeout=120) as resp:
            resp.read()
        h_latencies_a.append(time.perf_counter() - t0)

    total_time_a = time.perf_counter() - t0_strat_a
    gpu_stats_a = monitor.stop_sampling()

    print(f"  Strategy A Complete: Total time={total_time_a:.2f}s | Peak VRAM={gpu_stats_a['peak_vram_mb']}MB | Avg Util={gpu_stats_a['avg_gpu_util_pct']}%\n", flush=True)

    # Unload both
    monitor.unload_model(target_model)
    monitor.unload_model(hacker_model)
    time.sleep(2.0)

    # -------------------------------------------------------------
    # Strategy B: Sequential Loading / Unloading
    # -------------------------------------------------------------
    print("--- BENCHMARKING STRATEGY B: Sequential Loading ---", flush=True)
    monitor.start_sampling(interval_seconds=0.2)
    t0_strat_b = time.perf_counter()
    t_latencies_b = []
    h_latencies_b = []
    switching_overhead = []

    for turn in range(1, test_prompts_count + 1):
        # Load Target, infer, unload
        t_switch_0 = time.perf_counter()
        monitor.warm_model(target_model, keep_alive="1m")
        t_switch_t = time.perf_counter() - t_switch_0

        t0 = time.perf_counter()
        req_t = urllib.request.Request(
            f"{monitor.ollama_host}/api/generate",
            data=json.dumps({"model": target_model, "prompt": f"Write a short python function for step {turn}.", "stream": False}).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req_t, timeout=120) as resp:
            resp.read()
        t_latencies_b.append(time.perf_counter() - t0)
        monitor.unload_model(target_model)

        # Load Hacker, infer, unload
        t_switch_1 = time.perf_counter()
        monitor.warm_model(hacker_model, keep_alive="1m")
        t_switch_h = time.perf_counter() - t_switch_1
        switching_overhead.extend([t_switch_t, t_switch_h])

        t0 = time.perf_counter()
        req_h = urllib.request.Request(
            f"{monitor.ollama_host}/api/generate",
            data=json.dumps({"model": hacker_model, "prompt": f"Suggest a test tweak for step {turn}.", "stream": False}).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req_h, timeout=120) as resp:
            resp.read()
        h_latencies_b.append(time.perf_counter() - t0)
        monitor.unload_model(hacker_model)

    total_time_b = time.perf_counter() - t0_strat_b
    gpu_stats_b = monitor.stop_sampling()

    print(f"  Strategy B Complete: Total time={total_time_b:.2f}s | Peak VRAM={gpu_stats_b['peak_vram_mb']}MB | Avg Util={gpu_stats_b['avg_gpu_util_pct']}%\n", flush=True)

    # Decision logic
    # If both models fit in VRAM with 100% GPU offload and peak VRAM < 10.5 GB, Strategy A is vastly superior
    recommended_strategy = "resident" if (both_gpu and gpu_stats_a["peak_vram_mb"] < 11000) else "sequential"

    comparison_report = {
        "timestamp": time.time(),
        "hardware": asdict(hw),
        "target_model": target_model,
        "hacker_model": hacker_model,
        "strategy_a_resident": {
            "total_duration_s": round(total_time_a, 2),
            "target_avg_latency_s": round(sum(t_latencies_a) / len(t_latencies_a), 2),
            "hacker_avg_latency_s": round(sum(h_latencies_a) / len(h_latencies_a), 2),
            "peak_vram_mb": gpu_stats_a["peak_vram_mb"],
            "avg_vram_mb": gpu_stats_a["avg_vram_mb"],
            "avg_gpu_util_pct": gpu_stats_a["avg_gpu_util_pct"],
            "both_fully_on_gpu": both_gpu,
        },
        "strategy_b_sequential": {
            "total_duration_s": round(total_time_b, 2),
            "target_avg_latency_s": round(sum(t_latencies_b) / len(t_latencies_b), 2),
            "hacker_avg_latency_s": round(sum(h_latencies_b) / len(h_latencies_b), 2),
            "avg_model_switch_s": round(sum(switching_overhead) / max(1, len(switching_overhead)), 2),
            "peak_vram_mb": gpu_stats_b["peak_vram_mb"],
            "avg_vram_mb": gpu_stats_b["avg_vram_mb"],
            "avg_gpu_util_pct": gpu_stats_b["avg_gpu_util_pct"],
        },
        "recommended_strategy": recommended_strategy,
        "speedup_factor": round(total_time_b / max(0.01, total_time_a), 2),
        "justification": (
            f"Strategy A (Resident) is {total_time_b / max(0.01, total_time_a):.1f}x faster because both models "
            f"fit comfortably in 12GB VRAM (Peak: {gpu_stats_a['peak_vram_mb']:.0f} MB < 12288 MB) with 100% GPU offload "
            f"and eliminate {sum(switching_overhead):.1f}s of redundant model disk/PCIe reload overhead."
            if recommended_strategy == "resident"
            else "Strategy B is recommended to prevent out-of-memory errors on limited VRAM."
        ),
    }

    print("================================================================================")
    print(f"BENCHMARK RESULT: Recommended Strategy is '{recommended_strategy.upper()}'")
    print(f"Speedup: {comparison_report['speedup_factor']}x faster ({total_time_a:.1f}s vs {total_time_b:.1f}s)")
    print(f"Justification: {comparison_report['justification']}")
    print("================================================================================\n")

    return comparison_report


if __name__ == "__main__":
    report = run_gpu_benchmark_comparison()
    out_file = Path("c:/demo_cap/reward_hacking_mvp/runs/gpu_benchmark_report.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved benchmark report to {out_file}")
