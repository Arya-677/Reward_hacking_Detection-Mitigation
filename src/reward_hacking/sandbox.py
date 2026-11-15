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


ALLOWED_MODULES_SOLUTION: Set[str] = {
    "re",
    "math",
    "string",
    "collections",
    "itertools",
    "functools",
    "statistics",
    "typing",
    "operator",
    "dataclasses",
    "decimal",
    "fractions",
    "heapq",
    "bisect",
    "solution",
}

ALLOWED_MODULES_TEST: Set[str] = ALLOWED_MODULES_SOLUTION | {
    "pytest",
    "unittest",
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

DANGEROUS_DUNDER_ATTRS: Set[str] = {
    "__subclasses__",
    "__globals__",
    "__code__",
    "__dict__",
    "__bases__",
    "__class__",
    "__builtins__",
    "__import__",
}


def check_ast_security(code: str, filename: str = "<unknown>") -> Tuple[bool, Optional[str]]:
    """Inspect Python AST using strict ALLOWLIST for imports and rejecting forbidden calls and dunders."""
    try:
        tree = ast.parse(code, filename=filename)
    except SyntaxError:
        # Syntax errors are left to pytest to report naturally
        return True, None

    is_test_file = "test" in filename.lower()
    allowed_modules = ALLOWED_MODULES_TEST if is_test_file else ALLOWED_MODULES_SOLUTION

    for node in ast.walk(tree):
        # 1. Check 'import x'
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_pkg = alias.name.split(".")[0]
                if root_pkg not in allowed_modules:
                    return False, f"Import '{alias.name}' is not permitted by sandbox allowlist in {filename}."

        # 2. Check 'from x import y'
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root_pkg = node.module.split(".")[0]
                if root_pkg not in allowed_modules:
                    return False, f"Import from '{node.module}' is not permitted by sandbox allowlist in {filename}."

        # 3. Check calls to open, exec, eval, compile, etc.
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_id = node.func.id
                if func_id in FORBIDDEN_CALLS:
                    return False, f"Forbidden call to '{func_id}()' detected in {filename}."
                if func_id == "getattr" and len(node.args) >= 2:
                    if isinstance(node.args[1], ast.Constant) and str(node.args[1].value) in DANGEROUS_DUNDER_ATTRS:
                        return False, f"Forbidden getattr lookup for '{node.args[1].value}' in {filename}."
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr in FORBIDDEN_CALLS:
                    return False, f"Forbidden attribute call to '{node.func.attr}()' detected in {filename}."
                if node.func.attr in DANGEROUS_DUNDER_ATTRS:
                    return False, f"Forbidden call on dunder attribute '{node.func.attr}' in {filename}."

        # 4. Check dunder attribute access
        elif isinstance(node, ast.Attribute):
            if node.attr in DANGEROUS_DUNDER_ATTRS:
                return False, f"Access to dunder attribute '{node.attr}' is forbidden in {filename}."

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
