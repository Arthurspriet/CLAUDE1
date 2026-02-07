"""Bash command execution tool with safety classification."""

import re
import subprocess
from typing import Any

from claude1.config import (
    BLOCKED_BASH_PATTERNS,
    DEFAULT_BASH_TIMEOUT,
    MAX_TOOL_OUTPUT_CHARS,
    WARNED_BASH_PATTERNS,
)
from claude1.tools.base import BaseTool
from claude1.undo import BashUndoManager


class CommandSafety:
    """Classifies bash commands into blocked, warned, or allowed."""

    BLOCKED = "blocked"
    WARNED = "warned"
    ALLOWED = "allowed"

    _blocked_re = [re.compile(p, re.IGNORECASE) for p in BLOCKED_BASH_PATTERNS]
    _warned_re = [re.compile(p, re.IGNORECASE) for p in WARNED_BASH_PATTERNS]

    @classmethod
    def classify(cls, command: str) -> tuple[str, str]:
        """Classify a command. Returns (level, reason).

        level is one of BLOCKED, WARNED, ALLOWED.
        reason is the matched pattern description (empty for ALLOWED).
        """
        for pattern in cls._blocked_re:
            if pattern.search(command):
                return cls.BLOCKED, pattern.pattern
        for pattern in cls._warned_re:
            if pattern.search(command):
                return cls.WARNED, pattern.pattern
        return cls.ALLOWED, ""


class BashTool(BaseTool):
    def __init__(self, working_dir: str, bash_undo: BashUndoManager | None = None, **kwargs: Any):
        super().__init__(working_dir, **kwargs)
        self.bash_undo = bash_undo

    @property
    def name(self) -> str:
        return "bash"

    @property
    def description(self) -> str:
        return (
            "Execute a bash command in the working directory. "
            f"Output is truncated at {MAX_TOOL_OUTPUT_CHARS} characters. "
            f"Default timeout: {DEFAULT_BASH_TIMEOUT}s (override with timeout parameter). "
            "Use for: running scripts, git, package managers, builds, tests. "
            "Prefer grep_search over 'bash grep' and glob_search over 'bash find'. "
            "SAFETY: Do not run destructive commands (rm -rf, git reset --hard) without user approval. "
            "If a command fails, read stderr and adjust — don't blindly retry."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                },
                "timeout": {
                    "type": "integer",
                    "description": f"Timeout in seconds (default: {DEFAULT_BASH_TIMEOUT})",
                },
            },
            "required": ["command"],
        }

    @property
    def requires_confirmation(self) -> bool:
        return True

    def execute(self, **kwargs: Any) -> str:
        command = kwargs.get("command", "")
        timeout = kwargs.get("timeout", DEFAULT_BASH_TIMEOUT)

        if not command:
            return "Error: command is required"

        # ── Safety classification ──
        level, reason = CommandSafety.classify(command)
        if level == CommandSafety.BLOCKED:
            return (
                f"BLOCKED: This command matches a dangerous pattern and cannot be executed.\n"
                f"Pattern: {reason}\n"
                f"Command: {command}"
            )
        # WARNED commands are handled by the confirmation callback
        # (the _warn_level attribute is checked by the REPL's confirm callback)
        self._last_safety_level = level
        self._last_safety_reason = reason

        # ── Bash undo: capture snapshot before warned (destructive) commands ──
        if level == CommandSafety.WARNED and self.bash_undo is not None:
            self.bash_undo.capture(command, str(self.working_dir))

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.working_dir),
            )

            output_parts = []
            if result.stdout:
                output_parts.append(result.stdout)
            if result.stderr:
                output_parts.append(f"[stderr]\n{result.stderr}")

            output = "\n".join(output_parts) if output_parts else "(no output)"

            if result.returncode != 0:
                output = f"[exit code: {result.returncode}]\n{output}"

            if len(output) > MAX_TOOL_OUTPUT_CHARS:
                output = output[:MAX_TOOL_OUTPUT_CHARS] + f"\n... [truncated at {MAX_TOOL_OUTPUT_CHARS} chars]"

            return output

        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after {timeout} seconds"
        except Exception as e:
            return f"Error executing command: {e}"

    @property
    def safety_level(self) -> str:
        """Return the safety classification of the last command checked."""
        return getattr(self, "_last_safety_level", CommandSafety.ALLOWED)

    @property
    def safety_reason(self) -> str:
        """Return the reason for the last safety classification."""
        return getattr(self, "_last_safety_reason", "")
