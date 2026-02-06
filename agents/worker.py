"""Worker agent that executes individual tasks with role-specific tools."""

from __future__ import annotations

import json

import ollama

from agents.types import AgentRole, TaskSpec, TaskResult, TaskStatus
from agents.config import AgentRoleConfig, DEFAULT_AGENT_CONFIGS
from agents.file_lock import FileLockManager
from agents.tool_filter import FilteredToolRegistry
from system_prompt import build_system_prompt
from tools import ToolRegistry


class WorkerAgent:
    """Async worker that executes a single task using role-specific config.

    Uses ollama.AsyncClient for non-blocking I/O. Each worker gets a filtered
    tool subset and acquires file locks for write operations.
    """

    def __init__(
        self,
        role: AgentRole,
        tool_registry: ToolRegistry,
        file_lock: FileLockManager,
        working_dir: str,
        role_config: AgentRoleConfig | None = None,
        ollama_host: str = "http://localhost:11434",
    ):
        self.role = role
        self.working_dir = working_dir
        self._cancelled = False

        # Resolve config
        self.config = role_config or DEFAULT_AGENT_CONFIGS.get(
            role.value,
            DEFAULT_AGENT_CONFIGS["code_edit"],
        )

        # Filtered tools for this role
        self.tools = FilteredToolRegistry(
            registry=tool_registry,
            allowed_tool_names=self.config.tool_names,
            read_only=self.config.read_only,
        )

        self.file_lock = file_lock
        self.client = ollama.AsyncClient(host=ollama_host)

    def cancel(self):
        """Signal cancellation."""
        self._cancelled = True

    def _build_system_prompt(self, task: TaskSpec) -> str:
        """Build a role-specific system prompt."""
        base = build_system_prompt(
            self.working_dir,
            self.config.model,
            compact=True,  # workers use concise prompts
        )
        extra = self.config.system_prompt_extra
        role_header = f"\n## Agent Role: {self.role.value}\n\n{extra}\n"
        return base + role_header

    async def execute(self, task: TaskSpec) -> TaskResult:
        """Execute a task and return the result."""
        messages: list[dict] = [
            {"role": "system", "content": self._build_system_prompt(task)},
        ]

        # Build user message with context
        user_content = task.instruction
        if task.context:
            user_content = f"Context from previous tasks:\n{task.context}\n\nTask: {task.instruction}"
        messages.append({"role": "user", "content": user_content})

        tool_defs = self.tools.ollama_tool_definitions()
        files_modified: list[str] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        full_output = ""

        try:
            for _iteration in range(self.config.max_iterations):
                if self._cancelled:
                    return TaskResult(
                        task_id=task.id,
                        role=self.role,
                        status=TaskStatus.FAILED,
                        error="Cancelled",
                    )

                # Non-streaming call for simpler aggregation
                response = await self.client.chat(
                    model=self.config.model,
                    messages=messages,
                    tools=tool_defs if tool_defs else None,
                    stream=False,
                    options={"num_ctx": 8192},
                )

                # Extract response data
                msg = response.get("message", response.message) if isinstance(response, dict) else response.message
                content = msg.get("content", "") if isinstance(msg, dict) else (msg.content or "")
                tool_calls = msg.get("tool_calls", None) if isinstance(msg, dict) else getattr(msg, "tool_calls", None)

                # Track tokens
                if isinstance(response, dict):
                    total_prompt_tokens += response.get("prompt_eval_count", 0)
                    total_completion_tokens += response.get("eval_count", 0)
                else:
                    total_prompt_tokens += getattr(response, "prompt_eval_count", 0) or 0
                    total_completion_tokens += getattr(response, "eval_count", 0) or 0

                if content:
                    full_output += content

                # Build assistant message for history
                assistant_msg: dict = {"role": "assistant", "content": content}
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls
                messages.append(assistant_msg)

                # If no tool calls, we're done
                if not tool_calls:
                    break

                # Execute tool calls
                for tc in tool_calls:
                    if self._cancelled:
                        return TaskResult(
                            task_id=task.id,
                            role=self.role,
                            status=TaskStatus.FAILED,
                            error="Cancelled",
                        )

                    func = tc.get("function", tc) if isinstance(tc, dict) else tc.function
                    tool_name = func.get("name", "") if isinstance(func, dict) else func.name
                    tool_args = func.get("arguments", {}) if isinstance(func, dict) else func.arguments

                    if isinstance(tool_args, str):
                        try:
                            tool_args = json.loads(tool_args)
                        except json.JSONDecodeError:
                            tool_args = {"raw": tool_args}

                    tool = self.tools.get(tool_name)
                    if tool is None:
                        result = f"Error: Tool '{tool_name}' not available for role {self.role.value}"
                    else:
                        # Acquire file lock for write operations
                        locked_path = None
                        if self.file_lock.is_write_tool(tool_name):
                            path = self.file_lock.get_path_from_args(tool_name, tool_args)
                            if path:
                                locked_path = path
                                await self.file_lock.acquire(path)

                        try:
                            result = tool.execute(**tool_args)
                            # Track file modifications
                            if tool_name in ("write_file", "edit_file"):
                                path = tool_args.get("path", "")
                                if path and path not in files_modified:
                                    files_modified.append(path)
                        finally:
                            if locked_path:
                                self.file_lock.release(locked_path)

                    messages.append({"role": "tool", "content": result})

            return TaskResult(
                task_id=task.id,
                role=self.role,
                status=TaskStatus.COMPLETED,
                output=full_output,
                files_modified=files_modified,
                token_usage={
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens,
                },
            )

        except Exception as e:
            return TaskResult(
                task_id=task.id,
                role=self.role,
                status=TaskStatus.FAILED,
                output=full_output,
                error=str(e),
            )
