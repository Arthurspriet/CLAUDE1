"""Codebase introspection — parse Claude1's own source to understand capabilities."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


CLAUDE1_ROOT = Path(__file__).parent.parent  # the claude1/ package directory


@dataclass
class ModuleInfo:
    """Info about a Python module."""

    path: str
    classes: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    docstring: str = ""


@dataclass
class Capability:
    """A discovered capability."""

    type: str  # "tool", "skill", "agent_role", "slash_command"
    name: str
    description: str = ""


class CodebaseIntrospector:
    """Introspects Claude1's own source code."""

    def __init__(self):
        self.root = CLAUDE1_ROOT

    def get_module_map(self) -> dict[str, ModuleInfo]:
        """Parse all .py files and extract classes, functions, docstrings."""
        modules: dict[str, ModuleInfo] = {}

        for py_file in self.root.rglob("*.py"):
            relative = str(py_file.relative_to(self.root))
            try:
                source = py_file.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(py_file))

                classes = []
                functions = []
                docstring = ast.get_docstring(tree) or ""

                for node in ast.iter_child_nodes(tree):
                    if isinstance(node, ast.ClassDef):
                        classes.append(node.name)
                    elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                        functions.append(node.name)

                modules[relative] = ModuleInfo(
                    path=relative,
                    classes=classes,
                    functions=functions,
                    docstring=docstring,
                )
            except (SyntaxError, OSError):
                continue

        return modules

    def get_capabilities(self) -> list[Capability]:
        """Enumerate tools, skills, agent roles, and slash commands."""
        capabilities: list[Capability] = []

        # Discover tools from tools/__init__.py
        try:
            from claude1.tools import ToolRegistry
            # Create a minimal registry to enumerate tools
            registry = ToolRegistry(str(self.root.parent))
            for tool in registry.all_tools():
                capabilities.append(Capability(
                    type="tool",
                    name=tool.name,
                    description=tool.description[:100],
                ))
        except Exception:
            pass

        # Discover skills
        try:
            from claude1.skills import SkillRegistry
            sr = SkillRegistry(str(self.root.parent))
            for skill in sr.list_skills():
                capabilities.append(Capability(
                    type="skill",
                    name=skill.name,
                    description=skill.description[:100],
                ))
        except Exception:
            pass

        # Discover agent roles
        try:
            from claude1.agents.config import DEFAULT_AGENT_CONFIGS
            for role_name, config in DEFAULT_AGENT_CONFIGS.items():
                capabilities.append(Capability(
                    type="agent_role",
                    name=role_name,
                    description=f"model={config.model}, tools={len(config.tool_names)}",
                ))
        except Exception:
            pass

        # Discover custom roles
        try:
            from claude1.self_awareness.role_factory import RoleFactory
            rf = RoleFactory()
            for role in rf.list_custom_roles():
                capabilities.append(Capability(
                    type="agent_role",
                    name=role.model,
                    description="custom role",
                ))
        except Exception:
            pass

        # Parse slash commands from repl.py
        try:
            repl_path = self.root / "repl.py"
            if repl_path.exists():
                source = repl_path.read_text()
                import re
                commands = re.findall(r'cmd\s*==\s*"(/[a-z_-]+)"', source)
                for cmd in commands:
                    capabilities.append(Capability(
                        type="slash_command",
                        name=cmd,
                    ))
        except Exception:
            pass

        return capabilities

    def get_config_state(self) -> dict[str, Any]:
        """Return current configuration state."""
        try:
            from claude1.config import DEFAULT_MODEL, NUM_CTX, MAX_TOOL_ITERATIONS
            return {
                "default_model": DEFAULT_MODEL,
                "num_ctx": NUM_CTX,
                "max_tool_iterations": MAX_TOOL_ITERATIONS,
                "claude1_root": str(self.root),
            }
        except Exception:
            return {"claude1_root": str(self.root)}

    def generate_self_description(self) -> str:
        """Generate a natural-language summary for system prompt injection."""
        modules = self.get_module_map()
        capabilities = self.get_capabilities()
        config = self.get_config_state()

        tools = [c for c in capabilities if c.type == "tool"]
        skills = [c for c in capabilities if c.type == "skill"]
        roles = [c for c in capabilities if c.type == "agent_role"]
        commands = [c for c in capabilities if c.type == "slash_command"]

        lines = [
            f"You are Claude1, running from {config.get('claude1_root', 'unknown')}.",
            f"Source: {len(modules)} Python modules.",
            f"Tools: {', '.join(t.name for t in tools) or 'none loaded'}.",
            f"Skills: {', '.join(s.name for s in skills) or 'none loaded'}.",
            f"Agent roles: {', '.join(r.name for r in roles) or 'none loaded'}.",
            f"Slash commands: {', '.join(c.name for c in commands) or 'none loaded'}.",
        ]
        return "\n".join(lines)
