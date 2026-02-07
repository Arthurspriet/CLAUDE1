"""Agent role configuration and model mappings."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentRoleConfig:
    """Configuration for a specific agent role."""

    model: str
    tool_names: list[str] = field(default_factory=list)
    read_only: bool = False
    max_iterations: int = 10
    system_prompt_extra: str = ""


# All available tool names for reference
ALL_TOOLS = [
    "read_file", "write_file", "edit_file", "bash",
    "glob_search", "grep_search", "list_dir",
]

READ_ONLY_TOOLS = [
    "read_file", "glob_search", "grep_search", "list_dir",
]

# Default mapping: role name -> config
DEFAULT_AGENT_CONFIGS: dict[str, AgentRoleConfig] = {
    "coordinator": AgentRoleConfig(
        model="deepseek-r1:8b",
        tool_names=[],
        read_only=True,
        max_iterations=1,
        system_prompt_extra=(
            "You are a task coordinator. Your job is to decompose user requests "
            "into subtasks that can be executed by specialized agents. "
            "Output ONLY valid JSON — no extra text."
        ),
    ),
    "code_edit": AgentRoleConfig(
        model="devstral-small-2:24b",
        tool_names=ALL_TOOLS,
        read_only=False,
        max_iterations=15,
        system_prompt_extra=(
            "You are a code editing agent. You have full access to file tools and bash. "
            "Complete the assigned task precisely and report what you changed."
        ),
    ),
    "search": AgentRoleConfig(
        model="qwen3:4b",
        tool_names=READ_ONLY_TOOLS,
        read_only=True,
        max_iterations=10,
        system_prompt_extra=(
            "You are a search agent with read-only access to the codebase. "
            "Find the requested information and report your findings concisely."
        ),
    ),
    "reasoning": AgentRoleConfig(
        model="deepseek-r1:8b",
        tool_names=[],
        read_only=True,
        max_iterations=1,
        system_prompt_extra=(
            "You are a reasoning agent. Analyze the given context and provide "
            "clear, structured analysis. You have no tools — work only from context provided."
        ),
    ),
    "quick_lookup": AgentRoleConfig(
        model="llama3.2:1b",
        tool_names=READ_ONLY_TOOLS,
        read_only=True,
        max_iterations=5,
        system_prompt_extra=(
            "You are a fast lookup agent. Quickly find and return the requested information. "
            "Be concise."
        ),
    ),
}
