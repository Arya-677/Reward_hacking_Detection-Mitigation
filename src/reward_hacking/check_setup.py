import sys
from pathlib import Path
from reward_hacking.config import load_config
from reward_hacking.model_client import ModelClient
from reward_hacking.sandbox import Sandbox


def check_system_setup():
    print("================================================================================")
    print("REWARD HACKING MVP - ENVIRONMENT AND DEPENDENCY CHECK")
    print("================================================================================")

    config = load_config()
    all_ok = True

    # 1. Python Check
    py_ver = sys.version.split()[0]
    print(f"[*] Python Version: {py_ver} ({sys.executable})")
    if sys.version_info < (3, 10):
        print("    [!] Warning: Python 3.10+ is recommended.")

    # 2. Ollama Check
    client = ModelClient(base_url=config.ollama.base_url)
    ollama_running = client.check_ollama_running()
    if ollama_running:
        print(f"[+] Ollama Server: RUNNING at {config.ollama.base_url}")
        models = client.list_installed_models()
        model_names = [m.get("name") for m in models]
        print(f"[*] Installed Models ({len(model_names)}): {', '.join(model_names)}")

        # Check Target Model
        target = config.target.model
        if client.is_model_installed(target):
            t_digest = client.get_model_digest(target)
            print(f"[+] Target Model '{target}' is INSTALLED (Digest: {t_digest[:16] if t_digest else 'N/A'}...)")
        else:
            print(f"[-] Target Model '{target}' is MISSING: run 'ollama pull {target}'")
            all_ok = False

        # Check Hacker Model
        hacker = config.hacker.model
        if client.is_model_installed(hacker):
            h_digest = client.get_model_digest(hacker)
            print(f"[+] Hacker Model '{hacker}' is INSTALLED (Digest: {h_digest[:16] if h_digest else 'N/A'}...)")
        else:
            print(f"[-] Hacker Model '{hacker}' is MISSING: run 'ollama pull {hacker}'")
            all_ok = False
    else:
        print(f"[-] Ollama Server: NOT REACHABLE at {config.ollama.base_url}")
        print("    Please start Ollama with 'ollama serve' or open the Ollama desktop app.")
        all_ok = False

    # 3. Sandbox Check
    sandbox = Sandbox(
        mode=config.sandbox.mode,
        allow_restricted=config.sandbox.allow_restricted,
    )
    print(f"[*] Configured Sandbox Mode: {config.sandbox.mode}")
    print(f"[*] Effective Sandbox Mode:  {sandbox.effective_mode}")
    if sandbox.effective_mode == "restricted_subprocess":
        print("    Note: Running in restricted_subprocess fallback mode (Docker unavailable).")

    # 4. Tasks Check
    tasks_dir = config.project_root / "data" / "tasks"
    if tasks_dir.exists():
        task_dirs = [p.name for p in tasks_dir.iterdir() if p.is_dir() and (p / "task.yaml").exists()]
        print(f"[+] Task Suites Found ({len(task_dirs)}): {', '.join(task_dirs)}")
        if len(task_dirs) < 5:
            print(f"    [!] Warning: Expected at least 5 tasks, found {len(task_dirs)}.")
    else:
        print(f"[-] Task suites directory missing at {tasks_dir}")
        all_ok = False

    print("================================================================================")
    if all_ok:
        print("STATUS: ALL CHECKS PASSED. Ready to run experiments.")
    else:
        print("STATUS: SOME CHECKS FAILED. See guidance above before running.")
    print("================================================================================")
    return all_ok


if __name__ == "__main__":
    ok = check_system_setup()
    sys.exit(0 if ok else 1)
