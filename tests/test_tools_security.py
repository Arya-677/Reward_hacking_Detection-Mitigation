import os
import tempfile
from pathlib import Path
import pytest

from reward_hacking.agent_tools import WorkspaceToolHandler, validate_workspace_path
from reward_hacking.sandbox import Sandbox, check_ast_security


@pytest.fixture
def workspace_dir():
    tmp = Path(tempfile.mkdtemp(prefix="sec_ws_"))
    (tmp / "solution.py").write_text("def test(): pass\n", encoding="utf-8")
    (tmp / "test_visible.py").write_text("def test_ok(): pass\n", encoding="utf-8")
    yield tmp
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


def test_path_traversal_and_forbidden_paths(workspace_dir):
    """Requirement G: Path traversal, absolute path, reserved names, and non-whitelisted files are rejected."""
    # 1. Path traversal
    valid, _, err = validate_workspace_path(workspace_dir, "../outside.py")
    assert valid is False
    assert "Directory traversal" in err

    # 2. Absolute path
    valid, _, err = validate_workspace_path(workspace_dir, "C:/Windows/notepad.exe")
    assert valid is False
    assert "forbidden" in err.lower()

    # 3. Reserved device name (Windows)
    valid, _, err = validate_workspace_path(workspace_dir, "CON.py")
    assert valid is False
    assert "reserved" in err.lower()

    # 4. Non-whitelisted file
    valid, _, err = validate_workspace_path(workspace_dir, "secret.txt")
    assert valid is False
    assert "Access denied" in err


def test_ast_security_rejections():
    """Requirement G: Dangerous imports and calls are rejected by AST pre-check."""
    dangerous_snippets = [
        ("import os\nos.system('dir')", "os"),
        ("import subprocess\nsubprocess.run('echo hi')", "subprocess"),
        ("import socket\nsocket.socket()", "socket"),
        ("from pathlib import Path\nPath('.').exists()", "pathlib"),
        ("f = open('test.txt', 'w')", "open"),
        ("eval('1 + 1')", "eval"),
        ("exec('print(1)')", "exec"),
    ]

    for code, expected_kw in dangerous_snippets:
        is_safe, reason = check_ast_security(code, filename="solution.py")
        assert is_safe is False, f"Dangerous code unexpectedly passed AST check: {code}"
        assert expected_kw in reason


def test_sandbox_timeout_kills_infinite_loop(workspace_dir):
    """Requirement G: Sandbox hard timeout kills infinite loop."""
    sandbox = Sandbox(mode="restricted_subprocess", allow_restricted=True, timeout_seconds=2)
    loop_code = """
from solution import test
def test_infinite_loop():
    while True:
        pass
"""
    (workspace_dir / "test_visible.py").write_text(loop_code, encoding="utf-8")
    res = sandbox.run_tests(workspace_dir, "test_visible.py", timeout=2)
    assert res.timed_out is True
    assert res.exit_code == -1
