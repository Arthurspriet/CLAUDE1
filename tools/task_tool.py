"""Task tool - spawns isolated sub-conversations for focused work."""

from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING

from tools.base import BaseTool

if TYPE_CHECKING:
    from config import AppConfig
    from task_manager import TaskManager
    from tools import ToolRegistry


class TaskTool(BaseTool):
    """Spawn an isolated sub-conversation for a focused task."""

    def __init__(
        self,
        working_dir: str,
        config: AppConfig,
        task_manager: TaskManager,
        tool_registry: ToolRegistry,
        confirm_callback: Callable[[str, dict], bool] | None = None,
        is_subtask: bool = False,
    ):
        super().__init__(working_dir)
        self.config = config
        self.task_manager = task_manager
        self.parent_registry = tool_registry
        self.confirm_callback = confirm_callback
        self.is_subtask = is_subtask

    @property
    def name(self) -> str:
        return "task"

    @property
    def description(self) -> str:
        return "Spawn an isolated sub-conversation for a focused task. Use for tasks that benefit from a fresh context window."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Brief title for the task",
                },
                "prompt": {
                    "type": "string",
                    "description": "Detailed task instructions",
                },
                "tools": {
                    "type": "string",
                    "description": "Comma-separated tool names to allow (default: all except task)",
                },
            },
            "required": ["description", "prompt"],
        }

    @property
    def requires_confirmation(self) -> bool:
        return True

    def execute(self, **kwargs: Any) -> str:
        description = kwargs.get("description", "Subtask")
        prompt = kwargs.get("prompt", "")
        tools_str = kwargs.get("tools", "")

        # Prevent nested subtasks
        if self.is_subtask:
            return "Error: Subtasks cannot spawn further subtasks (max 1 level of nesting)."

        if not prompt:
            return "Error: prompt is required."

        task = self.task_manager.create(description)

        try:
            # Import here to avoid circular imports
            from llm import LLMInterface
            from tools import ToolRegistry

            # Create a filtered tool registry for the subtask
            sub_registry = ToolRegistry(
                self.config.working_dir,
                config=self.config,
                task_manager=self.task_manager,
                confirm_callback=self.confirm_callback,
                is_subtask=True,
            )

            # If specific tools requested, note it (filter will be applied in LLM)
            sub_llm = LLMInterface(
                config=self.config,
                tool_registry=sub_registry,
                confirm_callback=self.confirm_callback,
            )

            # Apply tool filter if specified
            if tools_str:
                tool_names = [t.strip() for t in tools_str.split(",") if t.strip()]
                # Always exclude 'task' from subtask to prevent nesting
                tool_names = [t for t in tool_names if t != "task"]
                sub_llm.set_tool_filter(tool_names)

            # Run to completion (collect all text chunks)
            result_text = ""
            for chunk in sub_llm.send_message(prompt):
                if chunk.type == "text":
                    result_text += chunk.content
                elif chunk.type == "error":
                    self.task_manager.fail(task.id, chunk.content)
                    return f"Task failed: {chunk.content}"
                elif chunk.type == "done":
                    break

            self.task_manager.complete(task.id, result_text)

            # Return truncated result to parent
            if len(result_text) > 2000:
                return result_text[:2000] + f"\n... [{len(result_text)} chars total]"
            return result_text

        except Exception as e:
            self.task_manager.fail(task.id, str(e))
            return f"Task failed: {e}"
