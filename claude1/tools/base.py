"""Abstract base class for all tools."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from claude1.undo import UndoStack


class PathSandboxError(Exception):
    """Raised when a file path violates sandboxing rules."""


class BaseTool(ABC):
    """Base class that all tools must inherit from."""

    def __init__(self, working_dir: str, undo_stack: UndoStack | None = None):
        self.working_dir = Path(working_dir).resolve()
        self.undo_stack = undo_stack

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name as the model will call it."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description for the model."""
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """JSON Schema for the tool's parameters."""
        ...

    @property
    def requires_confirmation(self) -> bool:
        """Whether this tool needs user confirmation before executing."""
        return False

    @abstractmethod
    def execute(self, **kwargs: Any) -> str:
        """Execute the tool. Must always return a string, never raise."""
        ...

    def _resolve_path(self, path: str) -> Path:
        """Resolve a path relative to working_dir, with sandboxing.

        After resolving, verifies:
        1. The path is a descendant of working_dir OR an allowed extra path.
        2. The path does not point to a known sensitive location.

        Raises PathSandboxError if the path is disallowed.
        """
        p = Path(path)
        if p.is_absolute():
            resolved = p.resolve()
        else:
            resolved = (self.working_dir / p).resolve()

        self._check_sandbox(resolved)
        return resolved

    def _check_sandbox(self, resolved: Path) -> None:
        """Verify a resolved path is within the sandbox."""
        from claude1.config import ALLOWED_EXTRA_PATHS, SENSITIVE_PATHS

        resolved_str = str(resolved)

        # Check sensitive paths first (block regardless of location)
        for sensitive in SENSITIVE_PATHS:
            expanded = str(Path(os.path.expanduser(sensitive)).resolve())
            if resolved_str == expanded or resolved_str.startswith(expanded + os.sep):
                raise PathSandboxError(
                    f"Access denied: {resolved} is in a sensitive location ({sensitive})"
                )

        # Allow if within working directory
        try:
            resolved.relative_to(self.working_dir)
            return
        except ValueError:
            pass

        # Allow if within an allowed extra path
        for allowed in ALLOWED_EXTRA_PATHS:
            allowed_resolved = Path(os.path.expanduser(allowed)).resolve()
            try:
                resolved.relative_to(allowed_resolved)
                return
            except ValueError:
                continue

        raise PathSandboxError(
            f"Access denied: {resolved} is outside the working directory ({self.working_dir}). "
            f"Only paths within the working directory or {ALLOWED_EXTRA_PATHS} are allowed."
        )

    def to_tool_definition(self) -> dict:
        """Convert to OpenAI-compatible tool-calling format (used by both Ollama and HuggingFace)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    # Backwards compatibility alias
    to_ollama_tool = to_tool_definition
