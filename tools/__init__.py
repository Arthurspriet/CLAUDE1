"""Tool registry - instantiates and manages all available tools."""

from tools.base import BaseTool
from tools.bash_tool import BashTool
from tools.file_tools import EditFileTool, ReadFileTool, WriteFileTool
from tools.search_tools import GlobSearchTool, GrepSearchTool, ListDirTool
from undo import UndoStack


class ToolRegistry:
    """Registry that holds all tool instances and provides lookup."""

    def __init__(self, working_dir: str):
        self._tools: dict[str, BaseTool] = {}
        self.undo_stack = UndoStack()
        self._register_all(working_dir)

    def _register_all(self, working_dir: str):
        tools = [
            ReadFileTool(working_dir),
            WriteFileTool(working_dir, undo_stack=self.undo_stack),
            EditFileTool(working_dir, undo_stack=self.undo_stack),
            BashTool(working_dir),
            GlobSearchTool(working_dir),
            GrepSearchTool(working_dir),
            ListDirTool(working_dir),
        ]
        for tool in tools:
            self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def all_tools(self) -> list[BaseTool]:
        """Return all registered tools."""
        return list(self._tools.values())

    def ollama_tool_definitions(self) -> list[dict]:
        """Return tool definitions in Ollama API format."""
        return [tool.to_ollama_tool() for tool in self._tools.values()]
