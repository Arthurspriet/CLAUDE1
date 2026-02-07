"""Tool registry - instantiates and manages all available tools."""

from __future__ import annotations

from typing import Callable, TYPE_CHECKING

from claude1.tools.base import BaseTool
from claude1.tools.bash_tool import BashTool
from claude1.tools.file_tools import EditFileTool, ReadFileTool, WriteFileTool
from claude1.tools.search_tools import GlobSearchTool, GrepSearchTool, ListDirTool
from claude1.tools.task_tool import TaskTool
from claude1.undo import UndoStack

if TYPE_CHECKING:
    from claude1.config import AppConfig
    from claude1.task_engine import TaskEngine


class ToolRegistry:
    """Registry that holds all tool instances and provides lookup."""

    def __init__(
        self,
        working_dir: str,
        config: AppConfig | None = None,
        task_manager: TaskEngine | None = None,
        confirm_callback: Callable[[str, dict], bool] | None = None,
        is_subtask: bool = False,
        mcp_tools: list[BaseTool] | None = None,
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
        if mcp_tools:
            for tool in mcp_tools:
                self._tools[tool.name] = tool

    def _register_all(
        self,
        working_dir: str,
        config: AppConfig | None = None,
        task_manager: TaskEngine | None = None,
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

        # Add video tools if dependencies are available
        try:
            from claude1.tools.video_tools import ImageToVideoTool, TextToVideoTool
            tools.append(TextToVideoTool(working_dir))
            tools.append(ImageToVideoTool(working_dir))
        except ImportError:
            # Video generation dependencies not installed - skip video tools
            pass

        # Web tools (optional: httpx, beautifulsoup4)
        try:
            from claude1.tools.web_tools import WebFetchTool, HttpRequestTool, WebSearchTool
            tools.append(WebFetchTool(working_dir))
            tools.append(HttpRequestTool(working_dir))
            tools.append(WebSearchTool(working_dir))
        except ImportError:
            pass

        # Communication tools
        try:
            from claude1.tools.comm_tools import SendEmailTool, WebhookTool
            tools.append(SendEmailTool(working_dir))
            tools.append(WebhookTool(working_dir))
        except ImportError:
            pass

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

        # Register self-awareness tools
        try:
            from claude1.tools.self_tools import CreateSkillTool, CreateAgentRoleTool, ModifySelfTool
            from claude1.self_awareness.skill_factory import SkillFactory
            from claude1.self_awareness.role_factory import RoleFactory
            from claude1.self_awareness.self_modifier import SelfModifier

            skill_factory = SkillFactory(working_dir)
            role_factory = RoleFactory()
            self_modifier = SelfModifier()

            tools.extend([
                CreateSkillTool(working_dir, skill_factory=skill_factory),
                CreateAgentRoleTool(working_dir, role_factory=role_factory),
                ModifySelfTool(working_dir, self_modifier=self_modifier),
            ])
        except Exception:
            pass  # Self-awareness module not available

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
