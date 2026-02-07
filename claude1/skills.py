"""Skills system - markdown-based capability extensions."""

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from claude1.config import DATA_DIR


@dataclass
class SkillDefinition:
    """A parsed skill from a SKILL.md file."""

    name: str
    description: str
    body: str  # markdown instructions
    path: Path  # directory containing SKILL.md
    allowed_tools: list[str] | None = None  # None = all tools


_SKILL_TEMPLATE = """---
name: {name}
description: "Describe what this skill does"
# allowed-tools: bash, read_file, grep_search
---

## Instructions

Write your skill instructions here.
The model will follow these when `/{name}` is invoked.

Use $ARGUMENTS to reference what the user passed after the command.
"""


class SkillRegistry:
    """Discovers, loads, and manages skills from SKILL.md files."""

    def __init__(self, working_dir: str):
        self.working_dir = Path(working_dir)
        self._skills: dict[str, SkillDefinition] = {}
        self._discover_skills()

    def _discover_skills(self):
        """Scan project and user skill directories."""
        self._skills.clear()
        # Project-level skills (higher priority)
        project_dir = self.working_dir / ".claude1" / "skills"
        # User-level skills
        user_dir = DATA_DIR / "skills"

        # Load user-level first (lower priority), then project overrides
        for skills_dir in [user_dir, project_dir]:
            if not skills_dir.is_dir():
                continue
            for child in sorted(skills_dir.iterdir()):
                if child.is_dir():
                    skill_file = child / "SKILL.md"
                    if skill_file.is_file():
                        try:
                            skill = self._parse_skill(skill_file)
                            if skill:
                                self._skills[skill.name] = skill
                        except Exception:
                            pass  # Skip malformed skills

    def _parse_skill(self, path: Path) -> SkillDefinition | None:
        """Parse a SKILL.md file into a SkillDefinition."""
        text = path.read_text(encoding="utf-8")

        # Parse YAML frontmatter
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", text, re.DOTALL)
        if not match:
            return None

        frontmatter_text = match.group(1)
        body = match.group(2).strip()

        # Simple YAML parsing (no external dependency)
        frontmatter: dict[str, str] = {}
        for line in frontmatter_text.splitlines():
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            if ":" in line:
                key, _, value = line.partition(":")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                frontmatter[key] = value

        name = frontmatter.get("name")
        description = frontmatter.get("description", "")
        if not name:
            return None

        # Parse allowed-tools
        allowed_tools = None
        tools_str = frontmatter.get("allowed-tools")
        if tools_str:
            allowed_tools = [t.strip() for t in tools_str.split(",") if t.strip()]

        return SkillDefinition(
            name=name,
            description=description,
            body=body,
            path=path.parent,
            allowed_tools=allowed_tools,
        )

    def get(self, name: str) -> SkillDefinition | None:
        """Get a skill by name."""
        return self._skills.get(name)

    def list_skills(self) -> list[SkillDefinition]:
        """Return all discovered skills."""
        return list(self._skills.values())

    def render(self, skill: SkillDefinition, arguments: str) -> str:
        """Render a skill's body, substituting $ARGUMENTS."""
        rendered = skill.body.replace("$ARGUMENTS", arguments)
        return rendered

    def reload(self):
        """Re-scan directories and reload all skills."""
        self._discover_skills()

    def create_skill(self, name: str, project_level: bool = True) -> Path | None:
        """Create a new skill from template. Returns the SKILL.md path, or None on error."""
        if project_level:
            base = self.working_dir / ".claude1" / "skills"
        else:
            base = DATA_DIR / "skills"

        skill_dir = base / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_file = skill_dir / "SKILL.md"

        if skill_file.exists():
            return None  # Already exists

        skill_file.write_text(_SKILL_TEMPLATE.format(name=name), encoding="utf-8")
        return skill_file

    def open_in_editor(self, path: Path) -> bool:
        """Open a file in the user's $EDITOR. Returns True on success."""
        editor = os.environ.get("EDITOR", os.environ.get("VISUAL", "nano"))
        try:
            subprocess.run([editor, str(path)], check=False)
            return True
        except FileNotFoundError:
            return False
