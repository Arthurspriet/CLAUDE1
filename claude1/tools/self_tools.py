"""Self-awareness tools — create skills, agent roles, and modify source code."""

from __future__ import annotations

import json
import time
from typing import Any

from claude1.tools.base import BaseTool


class CreateSkillTool(BaseTool):
    """Create a new skill programmatically."""

    def __init__(self, working_dir: str, skill_factory=None):
        super().__init__(working_dir)
        self.skill_factory = skill_factory

    @property
    def name(self) -> str:
        return "create_skill"

    @property
    def description(self) -> str:
        return "Create a new slash-command skill. The skill will be available as /<name> immediately."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Skill name (becomes the slash command, e.g. 'deploy' -> /deploy)",
                },
                "description": {
                    "type": "string",
                    "description": "Brief description of what the skill does",
                },
                "instructions": {
                    "type": "string",
                    "description": "Markdown instructions the model follows when the skill is invoked",
                },
                "allowed_tools": {
                    "type": "string",
                    "description": "Comma-separated tool names to restrict to (optional, default: all)",
                },
            },
            "required": ["name", "description", "instructions"],
        }

    @property
    def requires_confirmation(self) -> bool:
        return True

    def execute(self, **kwargs: Any) -> str:
        if not self.skill_factory:
            return "Error: SkillFactory not initialized."

        name = kwargs.get("name", "")
        description = kwargs.get("description", "")
        instructions = kwargs.get("instructions", "")
        tools_str = kwargs.get("allowed_tools", "")

        if not name or not instructions:
            return "Error: name and instructions are required."

        allowed_tools = None
        if tools_str:
            allowed_tools = [t.strip() for t in tools_str.split(",") if t.strip()]

        try:
            path = self.skill_factory.create_skill_from_spec(
                name=name,
                description=description,
                instructions=instructions,
                allowed_tools=allowed_tools,
            )
            return f"Skill '{name}' created at {path}. Use /{name} to invoke it."
        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error creating skill: {e}"


class CreateAgentRoleTool(BaseTool):
    """Create a new custom agent role."""

    def __init__(self, working_dir: str, role_factory=None):
        super().__init__(working_dir)
        self.role_factory = role_factory

    @property
    def name(self) -> str:
        return "create_agent_role"

    @property
    def description(self) -> str:
        return "Create a new agent role for the multi-agent system. Available immediately in /agents mode."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Role name (e.g. 'security_auditor')",
                },
                "model": {
                    "type": "string",
                    "description": "Ollama model to use for this role",
                },
                "tool_names": {
                    "type": "string",
                    "description": "Comma-separated tool names (e.g. 'read_file,grep_search,bash')",
                },
                "read_only": {
                    "type": "boolean",
                    "description": "If true, prevents write operations",
                },
                "max_iterations": {
                    "type": "integer",
                    "description": "Max tool-calling iterations (default: 10)",
                },
                "system_prompt_extra": {
                    "type": "string",
                    "description": "Additional system prompt instructions for this role",
                },
            },
            "required": ["name", "model"],
        }

    @property
    def requires_confirmation(self) -> bool:
        return True

    def execute(self, **kwargs: Any) -> str:
        if not self.role_factory:
            return "Error: RoleFactory not initialized."

        name = kwargs.get("name", "")
        model = kwargs.get("model", "")
        tools_str = kwargs.get("tool_names", "")
        read_only = kwargs.get("read_only", False)
        max_iterations = kwargs.get("max_iterations", 10)
        system_prompt_extra = kwargs.get("system_prompt_extra", "")

        if not name or not model:
            return "Error: name and model are required."

        tool_names = [t.strip() for t in tools_str.split(",") if t.strip()] if tools_str else []

        try:
            config = self.role_factory.create_role(
                name=name,
                model=model,
                tool_names=tool_names,
                read_only=read_only,
                max_iterations=max_iterations,
                system_prompt_extra=system_prompt_extra,
            )
            return f"Agent role '{name}' created (model={config.model}, tools={len(config.tool_names)}). Available in /agents mode."
        except Exception as e:
            return f"Error creating role: {e}"


class ModifySelfTool(BaseTool):
    """Read, edit, or create Claude1's own source files."""

    def __init__(self, working_dir: str, self_modifier=None):
        super().__init__(working_dir)
        self.self_modifier = self_modifier

    @property
    def name(self) -> str:
        return "modify_self"

    @property
    def description(self) -> str:
        return (
            "Read, edit, or create Claude1's own source code. "
            "All modifications are git-backed and reversible. "
            "Use action='read' to view source, 'edit' to modify, 'write' to create, "
            "'create_tool' to generate a new tool, 'revert' to undo, 'list' to see history."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Action: read, edit, write, create_tool, revert, list",
                    "enum": ["read", "edit", "write", "create_tool", "revert", "list"],
                },
                "file_path": {
                    "type": "string",
                    "description": "Relative path within claude1/ (e.g. 'tools/bash_tool.py')",
                },
                "old_string": {
                    "type": "string",
                    "description": "String to find and replace (for action=edit)",
                },
                "new_string": {
                    "type": "string",
                    "description": "Replacement string (for action=edit)",
                },
                "content": {
                    "type": "string",
                    "description": "File content (for action=write) or tool implementation (for action=create_tool)",
                },
                "tool_spec": {
                    "type": "string",
                    "description": "JSON tool spec with name, description, parameters, implementation (for action=create_tool)",
                },
            },
            "required": ["action"],
        }

    @property
    def requires_confirmation(self) -> bool:
        # Always require confirmation — no bypass
        return True

    def execute(self, **kwargs: Any) -> str:
        if not self.self_modifier:
            return "Error: SelfModifier not initialized."

        action = kwargs.get("action", "")

        if action == "read":
            file_path = kwargs.get("file_path", "")
            if not file_path:
                return "Error: file_path is required for read action."
            return self.self_modifier.read_own_source(file_path)

        elif action == "edit":
            file_path = kwargs.get("file_path", "")
            old_string = kwargs.get("old_string", "")
            new_string = kwargs.get("new_string", "")
            if not file_path or not old_string or not new_string:
                return "Error: file_path, old_string, and new_string are required for edit action."
            return self.self_modifier.edit_own_source(file_path, old_string, new_string)

        elif action == "write":
            file_path = kwargs.get("file_path", "")
            content = kwargs.get("content", "")
            if not file_path or not content:
                return "Error: file_path and content are required for write action."
            return self.self_modifier.write_own_source(file_path, content)

        elif action == "create_tool":
            tool_spec_str = kwargs.get("tool_spec", "")
            if not tool_spec_str:
                return "Error: tool_spec (JSON) is required for create_tool action."
            try:
                spec = json.loads(tool_spec_str)
            except json.JSONDecodeError:
                return "Error: tool_spec must be valid JSON."
            name = spec.get("name", "")
            description = spec.get("description", "")
            parameters = spec.get("parameters", {})
            implementation = spec.get("implementation", "        return 'Not implemented'")
            if not name:
                return "Error: tool_spec must include 'name'."
            return self.self_modifier.create_tool(name, description, parameters, implementation)

        elif action == "revert":
            return self.self_modifier.revert_last_modification()

        elif action == "list":
            mods = self.self_modifier.list_modifications()
            if not mods:
                return "No self-modifications recorded."
            lines = []
            for m in mods[-10:]:  # last 10
                ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(m.timestamp)) if m.timestamp else "?"
                lines.append(f"  [{ts}] {m.action}: {m.file_path} — {m.diff_summary[:80]}")
            return f"Recent modifications ({len(mods)} total):\n" + "\n".join(lines)

        else:
            return f"Error: Unknown action '{action}'. Use: read, edit, write, create_tool, revert, list"
