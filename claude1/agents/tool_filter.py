"""Filtered tool registry for role-based tool access."""

from __future__ import annotations

from claude1.tools import ToolRegistry
from claude1.tools.base import BaseTool


class FilteredToolRegistry:
    """Wraps a ToolRegistry and exposes only allowed tools based on agent role.

    Respects read_only flag by removing write_file, edit_file, and bash.
    Provides the same interface as ToolRegistry for tool lookup and definitions.
    """

    WRITE_TOOLS = {"write_file", "edit_file", "bash"}

    def __init__(
        self,
        registry: ToolRegistry,
        allowed_tool_names: list[str],
        read_only: bool = False,
    ):
        self._registry = registry
        self._allowed = set(allowed_tool_names)
        if read_only:
            self._allowed -= self.WRITE_TOOLS

    def get(self, name: str) -> BaseTool | None:
        """Get a tool by name if it's in the allowed set."""
        if name not in self._allowed:
            return None
        return self._registry.get(name)

    def all_tools(self) -> list[BaseTool]:
        """Return all allowed tools."""
        return [
            tool for tool in self._registry.all_tools()
            if tool.name in self._allowed
        ]

    def ollama_tool_definitions(self) -> list[dict]:
        """Return tool definitions in Ollama API format, filtered to allowed tools."""
        return [tool.to_ollama_tool() for tool in self.all_tools()]

    @property
    def tool_names(self) -> list[str]:
        """Return list of allowed tool names."""
        return sorted(self._allowed)
