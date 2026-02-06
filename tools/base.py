"""Abstract base class for all tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from undo import UndoStack


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
        """Resolve a path relative to working_dir, or return absolute path."""
        p = Path(path)
        if p.is_absolute():
            return p.resolve()
        return (self.working_dir / p).resolve()

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
