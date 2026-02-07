"""Self-modification engine — modify Claude1's own source code with git safety."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from claude1.config import DATA_DIR
from claude1.self_awareness.introspection import CLAUDE1_ROOT


MODIFICATION_LOG = DATA_DIR / "logs" / "self_modifications.json"


@dataclass
class ModificationRecord:
    """Record of a self-modification."""

    timestamp: float
    action: str  # read, edit, write, create_tool, revert
    file_path: str
    diff_summary: str = ""
    rationale: str = ""
    git_commit: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "action": self.action,
            "file_path": self.file_path,
            "diff_summary": self.diff_summary,
            "rationale": self.rationale,
            "git_commit": self.git_commit,
        }


class SelfModifier:
    """Engine for modifying Claude1's own source code.

    Safety:
    - Auto-creates a git commit before every modification
    - Only operates within the claude1/ package directory
    - All modifications logged to ~/.claude1/logs/self_modifications.json
    """

    def __init__(self):
        self.root = CLAUDE1_ROOT
        self.project_root = CLAUDE1_ROOT.parent  # the git repo root
        self._modifications: list[ModificationRecord] = []
        self._load_log()

    def _load_log(self):
        """Load modification history from disk."""
        if not MODIFICATION_LOG.exists():
            return
        try:
            data = json.loads(MODIFICATION_LOG.read_text())
            for entry in data:
                self._modifications.append(ModificationRecord(
                    timestamp=entry.get("timestamp", 0),
                    action=entry.get("action", ""),
                    file_path=entry.get("file_path", ""),
                    diff_summary=entry.get("diff_summary", ""),
                    rationale=entry.get("rationale", ""),
                    git_commit=entry.get("git_commit", ""),
                ))
        except (json.JSONDecodeError, OSError):
            pass

    def _save_log(self):
        """Persist modification log."""
        MODIFICATION_LOG.parent.mkdir(parents=True, exist_ok=True)
        data = [m.to_dict() for m in self._modifications]
        MODIFICATION_LOG.write_text(json.dumps(data, indent=2, default=str))

    def _resolve_path(self, relative_path: str) -> Path:
        """Resolve a relative path within claude1/."""
        path = (self.root / relative_path).resolve()
        # Security: ensure we stay within claude1/
        if not str(path).startswith(str(self.root.resolve())):
            raise ValueError(f"Path {relative_path} escapes the claude1/ directory.")
        return path

    def _git_safety_commit(self, message: str) -> str:
        """Create a git commit of current state before modification."""
        try:
            # Stage all changes in claude1/
            subprocess.run(
                ["git", "add", "claude1/"],
                cwd=str(self.project_root),
                capture_output=True,
                timeout=10,
            )
            result = subprocess.run(
                ["git", "commit", "-m", f"[auto] pre-self-modify: {message}",
                 "--allow-empty"],
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=10,
            )
            # Get the commit hash
            hash_result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=5,
            )
            return hash_result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return ""

    def _log_modification(self, action: str, file_path: str,
                          diff_summary: str = "", rationale: str = "",
                          git_commit: str = ""):
        """Log a modification."""
        record = ModificationRecord(
            timestamp=time.time(),
            action=action,
            file_path=file_path,
            diff_summary=diff_summary,
            rationale=rationale,
            git_commit=git_commit,
        )
        self._modifications.append(record)
        self._save_log()

    def read_own_source(self, relative_path: str) -> str:
        """Read a file within claude1/."""
        path = self._resolve_path(relative_path)
        if not path.exists():
            return f"Error: File not found: {relative_path}"
        try:
            content = path.read_text(encoding="utf-8")
            # Add line numbers
            lines = content.splitlines()
            numbered = "\n".join(f"{i+1:4d}| {line}" for i, line in enumerate(lines))
            return numbered
        except OSError as e:
            return f"Error reading file: {e}"

    def edit_own_source(self, relative_path: str, old_string: str,
                        new_string: str, rationale: str = "") -> str:
        """Edit a file within claude1/ with git safety."""
        path = self._resolve_path(relative_path)
        if not path.exists():
            return f"Error: File not found: {relative_path}"

        try:
            content = path.read_text(encoding="utf-8")
        except OSError as e:
            return f"Error reading file: {e}"

        if old_string not in content:
            return f"Error: old_string not found in {relative_path}"

        count = content.count(old_string)
        if count > 1:
            return f"Error: old_string matches {count} locations. Provide more context."

        # Git safety commit
        commit_hash = self._git_safety_commit(f"before edit {relative_path}")

        # Perform the edit
        new_content = content.replace(old_string, new_string, 1)
        path.write_text(new_content, encoding="utf-8")

        diff_summary = f"-{old_string[:100]}... +{new_string[:100]}..."
        self._log_modification(
            action="edit",
            file_path=relative_path,
            diff_summary=diff_summary,
            rationale=rationale,
            git_commit=commit_hash,
        )

        return f"Edited {relative_path}. Git commit: {commit_hash[:8]}"

    def write_own_source(self, relative_path: str, content: str,
                         rationale: str = "") -> str:
        """Create a new file within claude1/ with git safety."""
        path = self._resolve_path(relative_path)

        # Git safety commit
        commit_hash = self._git_safety_commit(f"before create {relative_path}")

        # Create parent dirs if needed
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

        self._log_modification(
            action="write",
            file_path=relative_path,
            diff_summary=f"Created {relative_path} ({len(content)} chars)",
            rationale=rationale,
            git_commit=commit_hash,
        )

        return f"Created {relative_path}. Git commit: {commit_hash[:8]}"

    def create_tool(self, name: str, description: str, parameters: dict,
                    implementation: str, rationale: str = "") -> str:
        """Generate a new tool Python file from a spec and register it."""
        # Generate tool class code
        class_name = "".join(word.capitalize() for word in name.split("_")) + "Tool"
        params_str = json.dumps(parameters, indent=8)

        tool_code = f'''"""Auto-generated tool: {name}."""

from __future__ import annotations

from typing import Any

from claude1.tools.base import BaseTool


class {class_name}(BaseTool):
    """{description}"""

    @property
    def name(self) -> str:
        return "{name}"

    @property
    def description(self) -> str:
        return """{description}"""

    @property
    def parameters(self) -> dict:
        return {params_str}

    @property
    def requires_confirmation(self) -> bool:
        return True

    def execute(self, **kwargs: Any) -> str:
{implementation}
'''

        relative_path = f"tools/{name}_tool.py"
        result = self.write_own_source(relative_path, tool_code, rationale=rationale)

        self._log_modification(
            action="create_tool",
            file_path=relative_path,
            diff_summary=f"Created tool: {name} ({class_name})",
            rationale=rationale,
        )

        return f"Tool '{name}' created at claude1/{relative_path}. {result}\nNote: Register it in tools/__init__.py and restart to activate."

    def revert_last_modification(self) -> str:
        """Revert the last self-modification using git."""
        if not self._modifications:
            return "No modifications to revert."

        last = self._modifications[-1]
        if not last.git_commit:
            return "No git commit associated with last modification."

        try:
            # Revert to the safety commit
            result = subprocess.run(
                ["git", "checkout", last.git_commit, "--", "claude1/"],
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return f"Revert failed: {result.stderr}"

            self._modifications.pop()
            self._save_log()
            return f"Reverted to commit {last.git_commit[:8]}. File: {last.file_path}"
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            return f"Revert failed: {e}"

    def list_modifications(self) -> list[ModificationRecord]:
        """Return all self-modification records."""
        return list(self._modifications)
