"""Tool registry - instantiates and manages all available tools."""

from __future__ import annotations

from typing import Callable, TYPE_CHECKING

from tools.base import BaseTool
from tools.bash_tool import BashTool
from tools.file_tools import EditFileTool, ReadFileTool, WriteFileTool
from tools.search_tools import GlobSearchTool, GrepSearchTool, ListDirTool
from tools.task_tool import TaskTool
from undo import UndoStack

if TYPE_CHECKING:
    from config import AppConfig
    from task_manager import TaskManager


class ToolRegistry:
    """Registry that holds all tool instances and provides lookup."""

    def __init__(
        self,
        working_dir: str,
        config: AppConfig | None = None,
        task_manager: TaskManager | None = None,
        confirm_callback: Callable[[str, dict], bool] | None = None,
        is_subtask: bool = False,
    ):
        self._tools: dict[str, BaseTool] = {}
        self.undo_stack = UndoStack()
        self._register_all(
            working_dir,
            config=config,
            task_manager=task_manager,
            confirm_callback=confirm_callback,
            is_subtask=is_subtask,
        )

    def _register_all(
        self,
        working_dir: str,
        config: AppConfig | None = None,
        task_manager: TaskManager | None = None,
        confirm_callback: Callable[[str, dict], bool] | None = None,
        is_subtask: bool = False,
    ):
        tools: list[BaseTool] = [
            ReadFileTool(working_dir),
            WriteFileTool(working_dir, undo_stack=self.undo_stack),
            EditFileTool(working_dir, undo_stack=self.undo_stack),
            BashTool(working_dir),
            GlobSearchTool(working_dir),
            GrepSearchTool(working_dir),
            ListDirTool(working_dir),
        ]

        # Register TaskTool if we have the required dependencies
        if config is not None and task_manager is not None:
            tools.append(
                TaskTool(
                    working_dir,
                    config=config,
                    task_manager=task_manager,
                    tool_registry=self,
                    confirm_callback=confirm_callback,
                    is_subtask=is_subtask,
                )
            )

        for tool in tools:
            self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def all_tools(self) -> list[BaseTool]:
        """Return all registered tools."""
        return list(self._tools.values())

    def tool_definitions(self) -> list[dict]:
        """Return tool definitions in OpenAI-compatible format (used by both Ollama and HuggingFace)."""
        return [tool.to_tool_definition() for tool in self._tools.values()]

    # Backwards compatibility alias
    ollama_tool_definitions = tool_definitions
