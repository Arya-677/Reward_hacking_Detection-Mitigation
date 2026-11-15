import ast
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel


class SandboxResult(BaseModel):
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False
    rejected: bool = False
    rejection_reason: Optional[str] = None


FORBIDDEN_MODULES: Set[str] = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "shutil",
    "ctypes",
    "importlib",
    "pathlib",
    "requests",
    "urllib",
    "http",
    "builtins",
    "threading",
    "multiprocessing",
    "asyncio",
    "signal",
}

FORBIDDEN_CALLS: Set[str] = {
    "open",
    "eval",
    "exec",
    "__import__",
    "compile",
    "globals",
    "locals",
}


def check_ast_security(code: str, filename: str = "<unknown>") -> Tuple[bool, Optional[str]]:
    """Inspect Python AST for forbidden imports, dangerous builtins, or malicious calls."""
    try:
        tree = ast.parse(code, filename=filename)
    except SyntaxError:
        # Syntax errors are left to the test runner (pytest) to naturally fail
        return True, None

    for node in ast.walk(tree):
        # 1. Check 'import x'
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_pkg = alias.name.split(".")[0]
                if root_pkg in FORBIDDEN_MODULES:
                    return False, f"Forbidden import '{alias.name}' detected in {filename}"

        # 2. Check 'from x import y'
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root_pkg = node.module.split(".")[0]
                if root_pkg in FORBIDDEN_MODULES:
                    return False, f"Forbidden import from '{node.module}' detected in {filename}"

        # 3. Check calls to open, exec, eval, etc.
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in FORBIDDEN_CALLS:
                    return False, f"Forbidden call to '{node.func.id}()' detected in {filename}"
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr in FORBIDDEN_CALLS:
                    return False, f"Forbidden attribute call to '{node.func.attr}()' detected in {filename}"

    return True, None


class Sandbox:
    def __init__(
        self,
        mode: str = "restricted_subprocess",
        allow_restricted: bool = True,
        timeout_seconds: int = 10,
        docker_image: str = "python:3.11-slim",
    ):
        self.mode = mode
        self.allow_restricted = allow_restricted
        self.timeout_seconds = timeout_seconds
        self.docker_image = docker_image
        self._docker_available = False
        self._check_docker()

    def _check_docker(self):
        try:
            res = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=5,
                shell=False,
            )
            self._docker_available = res.returncode == 0
        except Exception:
            self._docker_available = False

    @property
    def effective_mode(self) -> str:
        if self.mode == "docker" and self._docker_available:
            return "docker"
        return "restricted_subprocess"

    def print_security_banner(self):
        eff = self.effective_mode
        if eff == "restricted_subprocess":
            banner = (
                "\n"
                "================================================================================\n"
                "SANDBOX DISCLOSURE: RUNNING IN RESTRICTED_SUBPROCESS MODE (FALLBACK)\n"
                "Docker daemon is not active on this host. Executing code in isolated temporary\n"
                "directories using 'python -I' with sanitized environment, strict AST import\n"
                "validation, and a hard execution timeout.\n"
                "================================================================================\n"
            )
            print(banner)
        else:
            banner = (
                "\n"
                "================================================================================\n"
                "SANDBOX NOTICE: RUNNING IN DOCKER CONTAINER ISOLATION MODE\n"
                "Disposable container with --network none, memory and CPU constraints.\n"
                "================================================================================\n"
            )
            print(banner)

    def run_tests(
        self,
        workspace_dir: Path,
        test_file_name: str,
        timeout: Optional[int] = None,
    ) -> SandboxResult:
        """Run pytest against test_file_name within the workspace in an isolated sandbox."""
        to = timeout or self.timeout_seconds
        eff = self.effective_mode

        if eff == "docker" and self._docker_available:
            return self._run_in_docker(workspace_dir, test_file_name, to)
        else:
            if not self.allow_restricted:
                raise PermissionError("Docker is unavailable and restricted_subprocess is not permitted.")
            return self._run_in_restricted_subprocess(workspace_dir, test_file_name, to)

    def _run_in_restricted_subprocess(
        self,
        workspace_dir: Path,
        test_file_name: str,
        timeout: int,
    ) -> SandboxResult:
        """Execute pytest in a clean temporary directory with python -I and AST verification."""
        t0 = time.perf_counter()

        # 1. AST pre-check on all python files in workspace
        for py_path in workspace_dir.glob("*.py"):
            try:
                code_content = py_path.read_text(encoding="utf-8")
                is_safe, reason = check_ast_security(code_content, filename=py_path.name)
                if not is_safe:
                    return SandboxResult(
                        exit_code=1,
                        stdout="",
                        stderr=f"SECURITY VIOLATION: {reason}",
                        duration_seconds=time.perf_counter() - t0,
                        rejected=True,
                        rejection_reason=reason,
                    )
            except Exception as e:
                return SandboxResult(
                    exit_code=1,
                    stdout="",
                    stderr=f"File read error during AST check: {e}",
                    duration_seconds=time.perf_counter() - t0,
                    rejected=True,
                    rejection_reason=str(e),
                )

        # 2. Copy workspace to isolated temp directory
        temp_dir = tempfile.mkdtemp(prefix="rh_sandbox_")
        try:
            shutil.copytree(workspace_dir, temp_dir, dirs_exist_ok=True)

            # Minimal sanitized environment
            clean_env = {
                "SYSTEMROOT": os.environ.get("SYSTEMROOT", "C:\\Windows"),
                "SYSTEMDRIVE": os.environ.get("SYSTEMDRIVE", "C:"),
                "COMSPEC": os.environ.get("COMSPEC", "C:\\Windows\\system32\\cmd.exe"),
                "PATH": os.path.dirname(sys.executable),
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUNBUFFERED": "1",
            }

            cmd = [
                sys.executable,
                "-I",
                "-m",
                "pytest",
                test_file_name,
                "-v",
                "--tb=short",
                "-p",
                "no:cacheprovider",
            ]

            proc = subprocess.run(
                cmd,
                cwd=temp_dir,
                env=clean_env,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=False,
            )
            duration = time.perf_counter() - t0
            return SandboxResult(
                exit_code=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                duration_seconds=duration,
                timed_out=False,
            )

        except subprocess.TimeoutExpired as te:
            duration = time.perf_counter() - t0
            return SandboxResult(
                exit_code=-1,
                stdout=te.stdout or "",
                stderr=f"Execution timed out after {timeout} seconds.",
                duration_seconds=duration,
                timed_out=True,
            )
        except Exception as e:
            duration = time.perf_counter() - t0
            return SandboxResult(
                exit_code=1,
                stdout="",
                stderr=f"Sandbox execution error: {e}",
                duration_seconds=duration,
                rejected=False,
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _run_in_docker(
        self,
        workspace_dir: Path,
        test_file_name: str,
        timeout: int,
    ) -> SandboxResult:
        """Run evaluation inside an isolated Docker container."""
        t0 = time.perf_counter()
        abs_ws = str(workspace_dir.resolve()).replace("\\", "/")

        cmd = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--memory",
            "256m",
            "--cpus",
            "1",
            "--pids-limit",
            "64",
            "-v",
            f"{abs_ws}:/workspace:ro",
            "-w",
            "/workspace",
            self.docker_image,
            "python",
            "-m",
            "pytest",
            test_file_name,
            "-v",
        ]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=False,
            )
            return SandboxResult(
                exit_code=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                duration_seconds=time.perf_counter() - t0,
                timed_out=False,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                exit_code=-1,
                stdout="",
                stderr=f"Docker container timed out after {timeout}s",
                duration_seconds=time.perf_counter() - t0,
                timed_out=True,
            )
        except Exception as e:
            return SandboxResult(
                exit_code=1,
                stdout="",
                stderr=f"Docker execution error: {e}",
                duration_seconds=time.perf_counter() - t0,
            )
