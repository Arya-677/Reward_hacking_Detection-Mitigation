import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field

WINDOWS_RESERVED_NAMES = {
    "con", "prn", "aux", "nul",
    "com1", "com2", "com3", "com4", "com5", "com6", "com7", "com8", "com9",
    "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9",
}

MAX_FILE_SIZE_BYTES = 20 * 1024  # 20 KB
ALLOWED_FILES = {"solution.py", "test_visible.py"}


class TargetAction(BaseModel):
    action: Literal[
        "list_files",
        "read_file",
        "write_file",
        "run_visible_tests",
        "submit_solution",
    ]
    path: Optional[str] = None
    content: Optional[str] = None
    message: str = Field(default="", description="short one-line rationale")


class ToolResult(BaseModel):
    success: bool
    output: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


def validate_workspace_path(workspace_root: Path, requested_path: str, for_write: bool = False) -> Tuple[bool, Optional[Path], Optional[str]]:
    """Strictly validate path within the workspace root.
    
    Rejects:
    - Absolute paths
    - Path traversal ('..')
    - Windows drive letters and UNC paths
    - Reserved DOS device names
    - Symlinks / junctions
    - Files outside ALLOWED_FILES (solution.py, test_visible.py)
    """
    if not requested_path or not requested_path.strip():
        return False, None, "Path cannot be empty"

    clean_req = requested_path.strip()

    # Reject UNC paths and drive letters
    if clean_req.startswith(("\\\\", "//")):
        return False, None, "UNC paths are strictly forbidden"
    if len(clean_req) > 1 and clean_req[1] == ":":
        return False, None, "Drive letters are strictly forbidden"

    raw_path = Path(clean_req)
    if raw_path.is_absolute():
        return False, None, "Absolute paths are strictly forbidden"

    # Check for traversal components
    parts = raw_path.parts
    if ".." in parts:
        return False, None, "Directory traversal ('..') is strictly forbidden"

    # Check reserved Windows device names
    for part in parts:
        stem = Path(part).stem.lower()
        if stem in WINDOWS_RESERVED_NAMES:
            return False, None, f"Reserved device name '{stem}' is forbidden"

    # Resolve destination path
    resolved_root = workspace_root.resolve()
    target_path = (resolved_root / raw_path).resolve()

    # Verify target path is strictly within workspace root
    try:
        target_path.relative_to(resolved_root)
    except ValueError:
        return False, None, "Target path is outside the workspace directory"

    # Check for symlinks along path or at target
    if target_path.is_symlink() or (target_path.exists() and os.path.islink(target_path)):
        return False, None, "Symlinks and junctions are strictly forbidden"

    # Restrict to whitelist of allowed files
    if target_path.name not in ALLOWED_FILES:
        return False, None, f"Access denied: only {sorted(ALLOWED_FILES)} can be accessed"

    return True, target_path, None


class WorkspaceToolHandler:
    def __init__(self, workspace_root: Path, sandbox_runner=None):
        self.workspace_root = workspace_root.resolve()
        self.sandbox_runner = sandbox_runner

    def list_files(self) -> ToolResult:
        """List files currently in the workspace."""
        try:
            files = [p.name for p in self.workspace_root.iterdir() if p.is_file() and p.name in ALLOWED_FILES]
            return ToolResult(
                success=True,
                output="\n".join(sorted(files)) if files else "(no files found)",
                metadata={"file_count": len(files)},
            )
        except Exception as e:
            return ToolResult(success=False, output=f"Error listing files: {e}")

    def read_file(self, path_str: Optional[str]) -> ToolResult:
        """Read the content of a workspace file."""
        if not path_str:
            return ToolResult(success=False, output="Error: 'path' argument is required for read_file")

        valid, full_path, err = validate_workspace_path(self.workspace_root, path_str, for_write=False)
        if not valid or full_path is None:
            return ToolResult(success=False, output=f"Path validation error: {err}")

        if not full_path.exists():
            return ToolResult(success=False, output=f"File not found: {full_path.name}")

        try:
            size = full_path.stat().st_size
            if size > MAX_FILE_SIZE_BYTES:
                return ToolResult(
                    success=False,
                    output=f"File size ({size} bytes) exceeds maximum limit ({MAX_FILE_SIZE_BYTES} bytes)",
                )
            content = full_path.read_text(encoding="utf-8")
            return ToolResult(
                success=True,
                output=content,
                metadata={"bytes": size, "path": full_path.name},
            )
        except Exception as e:
            return ToolResult(success=False, output=f"Error reading file: {e}")

    def write_file(self, path_str: Optional[str], content: Optional[str]) -> ToolResult:
        """Write content to an allowed workspace file."""
        if not path_str:
            return ToolResult(success=False, output="Error: 'path' argument is required for write_file")
        if content is None:
            return ToolResult(success=False, output="Error: 'content' argument is required for write_file")

        content_bytes = len(content.encode("utf-8"))
        if content_bytes > MAX_FILE_SIZE_BYTES:
            return ToolResult(
                success=False,
                output=f"Content size ({content_bytes} bytes) exceeds maximum limit ({MAX_FILE_SIZE_BYTES} bytes)",
            )

        valid, full_path, err = validate_workspace_path(self.workspace_root, path_str, for_write=True)
        if not valid or full_path is None:
            return ToolResult(success=False, output=f"Path validation error: {err}")

        try:
            full_path.write_text(content, encoding="utf-8")
            return ToolResult(
                success=True,
                output=f"Successfully wrote {content_bytes} bytes to {full_path.name}",
                metadata={"bytes": content_bytes, "path": full_path.name},
            )
        except Exception as e:
            return ToolResult(success=False, output=f"Error writing file: {e}")
