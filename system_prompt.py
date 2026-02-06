"""System prompt template for the coding assistant."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from config import DATA_DIR

if TYPE_CHECKING:
    from model_profiles import ModelProfile


def _get_git_info(working_dir: str) -> str | None:
    """Get git branch and status info if in a git repo."""
    try:
        # Check if we're in a git repo
        subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )

        # Get current branch
        branch_result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=5,
        )
        branch = branch_result.stdout.strip() or "HEAD (detached)"

        # Get short status
        status_result = subprocess.run(
            ["git", "status", "--short"],
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=5,
        )
        status = status_result.stdout.strip()
        status_summary = f"\n{status}" if status else " (clean)"

        return f"- Git branch: {branch}\n- Git status:{status_summary}"
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None


def _load_claude_md(working_dir: str) -> str | None:
    """Load CLAUDE.md from global config and/or project directory."""
    sections = []

    # Global user-level instructions
    global_path = DATA_DIR / "CLAUDE.md"
    if global_path.exists():
        try:
            content = global_path.read_text().strip()
            if content:
                sections.append(f"## User Instructions (Global)\n\n{content}")
        except OSError:
            pass

    # Project-level instructions (overrides/supplements global)
    project_path = Path(working_dir) / "CLAUDE.md"
    if project_path.exists():
        try:
            content = project_path.read_text().strip()
            if content:
                sections.append(f"## Project Instructions (CLAUDE.md)\n\n{content}")
        except OSError:
            pass

    return "\n\n".join(sections) if sections else None


def build_system_prompt(working_dir: str, model_name: str, compact: bool = False,
                        profile: ModelProfile | None = None) -> str:
    prompt = f"""You are Claude1, a local coding assistant running in the terminal. You help users with software engineering tasks by reading, writing, and editing files, running commands, and searching codebases.

## Environment
- Working directory: {working_dir}
- Model: {model_name}"""

    # Add git info
    git_info = _get_git_info(working_dir)
    if git_info:
        prompt += f"\n{git_info}"

    # Conditionally include tools section based on profile
    supports_tools = profile.supports_tools if profile else True

    if supports_tools:
        prompt += """

## Available Tools

You have access to these tools. Use them proactively to help the user:

1. **read_file** - Read file contents with line numbers. Always read a file before editing it.
2. **write_file** - Create or overwrite a file. Use for new files only. Prefer edit_file for modifications.
3. **edit_file** - Edit a file by exact string replacement. The old_string must match exactly one location. Always read the file first to get the exact content.
4. **bash** - Execute a shell command in a shell. Use for running scripts, git, package managers, etc.
5. **glob_search** - Find files by glob pattern (e.g., '**/*.py').
6. **grep_search** - Search file contents with regex. Returns matching lines with file paths.
7. **list_dir** - List directory contents with file sizes.

## Rules

1. **Read before edit**: Always use read_file before edit_file to see exact content including whitespace.
2. **Prefer edit over write**: Use edit_file to modify existing files. Only use write_file for new files.
3. **Be precise with edits**: The old_string in edit_file must match exactly one location, including indentation.
4. **Explain your actions**: Briefly explain what you're doing and why before using tools.
5. **Handle errors gracefully**: If a tool call fails, read the error, adjust, and retry.
6. **Stay in scope**: Only modify files the user asks about. Don't make unnecessary changes.
7. **Be concise**: Keep responses short and focused. Don't over-explain simple operations.
"""
    else:
        prompt += """

## Text-Only Mode

You are running in text-only mode — you do NOT have access to any tools.
Provide code for the user to copy and run manually.
When showing file changes, use clear before/after code blocks with full file paths.
"""

    # Append profile format rules
    if profile and profile.system_prompt_suffix:
        prompt += f"\n## Output Format\n\n{profile.system_prompt_suffix}\n"

    # Add CLAUDE.md content
    claude_md = _load_claude_md(working_dir)
    if claude_md:
        prompt += f"\n{claude_md}\n"

    # Compact mode instruction
    if compact:
        prompt += "\n## IMPORTANT: Compact Mode Active\nBe maximally concise. Give the shortest possible responses. Omit pleasantries and explanations unless asked. Use terse language.\n"

    return prompt
