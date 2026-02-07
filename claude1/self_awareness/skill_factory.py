"""Programmatic skill creation — write SKILL.md files without an editor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from claude1.config import DATA_DIR


_SKILL_TEMPLATE = """---
name: {name}
description: "{description}"
{tools_line}---

{instructions}
"""


@dataclass
class SkillSuggestion:
    """A suggested skill based on patterns."""

    name: str
    description: str
    instructions: str
    confidence: float = 0.0


class SkillFactory:
    """Create skills programmatically without requiring an editor."""

    def __init__(self, working_dir: str | None = None):
        self.working_dir = Path(working_dir) if working_dir else None

    def create_skill_from_spec(
        self,
        name: str,
        description: str,
        instructions: str,
        allowed_tools: list[str] | None = None,
        project_level: bool = True,
    ) -> Path:
        """Write a complete SKILL.md programmatically.

        Returns the path to the created SKILL.md file.
        Raises ValueError if the name conflicts with existing commands.
        """
        # Validate name
        _RESERVED_COMMANDS = {
            "help", "model", "models", "clear", "save", "load", "sessions",
            "resume", "auto", "temp", "compact", "plan", "undo", "export",
            "doctor", "debug", "context", "stats", "cost", "copy", "rewind",
            "memory", "init", "skills", "tasks", "agents", "exit", "quit",
            "hf-search", "hf-import", "video-text", "video-image", "video-models",
            "verify", "bandit",
        }
        if name.lower() in _RESERVED_COMMANDS:
            raise ValueError(f"Skill name '{name}' conflicts with a built-in command.")

        # Determine base directory
        if project_level and self.working_dir:
            base = self.working_dir / ".claude1" / "skills"
        else:
            base = DATA_DIR / "skills"

        skill_dir = base / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_file = skill_dir / "SKILL.md"

        # Build tools line
        tools_line = ""
        if allowed_tools:
            tools_line = f"allowed-tools: {', '.join(allowed_tools)}\n"

        content = _SKILL_TEMPLATE.format(
            name=name,
            description=description,
            tools_line=tools_line,
            instructions=instructions,
        )

        skill_file.write_text(content, encoding="utf-8")

        # Reload skills registry if available
        try:
            from claude1.skills import SkillRegistry
            sr = SkillRegistry(str(self.working_dir) if self.working_dir else ".")
            sr.reload()
        except Exception:
            pass

        return skill_file

    def suggest_skills_from_history(
        self,
        tool_calls: list[dict],
        messages: list[dict],
    ) -> list[SkillSuggestion]:
        """Analyze conversation history for recurring tool call patterns.

        Returns suggested skills based on repeated patterns.
        """
        # Count tool call sequences
        patterns: dict[str, int] = {}
        for i in range(len(tool_calls) - 1):
            pair = f"{tool_calls[i].get('tool', '')} -> {tool_calls[i+1].get('tool', '')}"
            patterns[pair] = patterns.get(pair, 0) + 1

        suggestions: list[SkillSuggestion] = []
        for pattern, count in sorted(patterns.items(), key=lambda x: -x[1]):
            if count >= 3:
                tools = pattern.split(" -> ")
                name = f"auto-{tools[0].replace('_', '-')}"
                suggestions.append(SkillSuggestion(
                    name=name,
                    description=f"Automates: {pattern} (seen {count}x)",
                    instructions=f"Follow this tool sequence: {pattern}",
                    confidence=min(1.0, count / 10.0),
                ))

        return suggestions
