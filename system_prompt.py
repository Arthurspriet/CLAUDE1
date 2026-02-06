"""System prompt template for the coding assistant."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from config import DATA_DIR

if TYPE_CHECKING:
    from model_profiles import ModelProfile


# ── Tiered behavioral rules ─────────────────────────────────────────────────

_FULL_RULES = """
### Tool Strategy
- Search the codebase (glob_search, grep_search) before asking the user where things are.
- Always read_file before edit_file — never guess file contents.
- Prefer edit_file over write_file for existing files. Use write_file only for new files.
- Keep tool calls to the minimum needed. Don't re-read a file you just read.

### Editing Discipline
- Copy old_string exactly from read_file output, including all whitespace and indentation.
- Include 3-5 lines of surrounding context so old_string matches uniquely.
- Preserve the existing code style (quotes, indent width, naming conventions).
- Make surgical changes — only modify what was requested. No drive-by refactors.

### Safety
- Ask before running destructive commands (rm -rf, git reset --hard, DROP TABLE).
- Don't introduce security vulnerabilities (injection, XSS, hardcoded secrets).
- Stay in scope — only modify files the user asked about.

### Error Recovery
- If a tool call fails, do NOT retry the identical call. Read the error and adjust.
- When blocked, try an alternative approach (different search, different edit strategy).
- Investigate unexpected state before overwriting — it may be the user's in-progress work.
""".strip()

_COMPACT_RULES = """
1. Search first — use glob_search/grep_search before asking where files are.
2. Always read_file before edit_file. Never guess content.
3. Prefer edit_file over write_file. Copy old_string exactly from read output.
4. Include enough context in old_string to match uniquely.
5. Only change what was requested. Preserve existing code style.
6. Ask before destructive commands. Don't add security vulnerabilities.
7. If a tool fails, read the error and adjust — don't retry identically.
""".strip()


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
                        profile: ModelProfile | None = None,
                        planning: bool = False) -> str:
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
    num_ctx = profile.num_ctx if profile else 4096

    if supports_tools:
        prompt += """

## Tools

You have 7 tools: read_file, write_file, edit_file, bash, glob_search, grep_search, list_dir. Use them proactively.

## Rules
"""
        # Select rules tier based on context window size
        if num_ctx >= 8192:
            prompt += "\n" + _FULL_RULES + "\n"
        else:
            prompt += "\n" + _COMPACT_RULES + "\n"

        # Append per-model behavioral rules
        if profile and profile.behavioral_rules:
            prompt += f"\n## Model-Specific Behavior\n\n{profile.behavioral_rules}\n"
    else:
        prompt += """

## Text-Only Mode

You are running in text-only mode — you do NOT have access to any tools.
Provide code for the user to copy and run manually.
When showing file changes, use clear before/after code blocks with full file paths.
"""

    # Append planning rules if enabled and tools are available
    if planning and supports_tools:
        from model_profiles import _PLANNING_RULES
        prompt += f"\n## Planning Mode\n\n{_PLANNING_RULES}\n"

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
