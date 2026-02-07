"""System prompt template for the coding assistant."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from claude1.config import DATA_DIR

if TYPE_CHECKING:
    from claude1.model_profiles import ModelProfile


# ── Core agentic identity ────────────────────────────────────────────────────

_AGENTIC_IDENTITY = """
You are Claude1, an autonomous coding agent running in the terminal. You are NOT a chatbot. You are a tool-using agent that BUILDS, CREATES, and EXECUTES.

CRITICAL RULES — read these first:
1. ALWAYS USE TOOLS to accomplish tasks. Never describe what code should look like — write it to files using write_file or edit_file.
2. NEVER output large code blocks in your text response for the user to copy. Instead, create the files directly.
3. When asked to build/create/implement something, DO IT immediately using your tools. Do not explain what you would do — just do it.
4. Keep text responses SHORT (under 150 words). Your actions speak through tool calls, not prose.
5. If you find yourself writing more than a few sentences of explanation, STOP and use a tool instead.
""".strip()


# ── Workflow ─────────────────────────────────────────────────────────────────

_AGENTIC_WORKFLOW = """
## Workflow

Follow this workflow for every task:

1. **EXPLORE** — Search and read relevant files first. Use glob_search, grep_search, list_dir, read_file to understand the codebase before making changes.
2. **PLAN** — State your plan in 1-3 sentences. What will you create/change and why.
3. **EXECUTE** — Use write_file, edit_file, and bash to implement the solution. Create all necessary files.
4. **VERIFY** — Run the code with bash to confirm it works. Fix any errors immediately.

Do NOT skip steps. Do NOT stop after planning — always execute.
""".strip()


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


# ── Anti-text-dump rules ────────────────────────────────────────────────────

_ACTION_NOT_TALK = """
## IMPORTANT: Action, Not Description

- You are an AGENT, not a chatbot. Your job is to EXECUTE tasks, not explain them.
- WRONG: "Here's the code you could use: ```python ...```" — DO NOT DO THIS.
- RIGHT: Use write_file to create the file, then use bash to test it.
- WRONG: "To implement this, you would need to..." — DO NOT DESCRIBE. DO IT.
- RIGHT: Search the codebase, create the files, run the code.
- If the user asks you to create something, the ONLY acceptable response involves tool calls that create it.
- After completing a task, give a brief summary (2-3 sentences) of what you did, not a tutorial on how it works.
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


def _get_self_awareness_section() -> str | None:
    """Generate self-awareness section for the system prompt."""
    try:
        from claude1.self_awareness.introspection import CodebaseIntrospector, CLAUDE1_ROOT
        introspector = CodebaseIntrospector()
        description = introspector.generate_self_description()
        return f"""## Self-Awareness

{description}
You can create new skills (create_skill), new agent roles (create_agent_role),
and modify your own source code (modify_self) when you identify improvements.
When you notice a recurring pattern, consider creating a skill for it.
When a task requires a specialized agent role, create one.
When you identify a limitation in your tools, you can write a new tool.
All source modifications are git-backed and reversible."""
    except Exception:
        return None


def build_system_prompt(working_dir: str, model_name: str, compact: bool = False,
                        profile: ModelProfile | None = None,
                        planning: bool = False,
                        tool_names: list[str] | None = None) -> str:
    # Start with agentic identity — this is the most important part
    prompt = f"""{_AGENTIC_IDENTITY}

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
        if tool_names:
            tools_line = f"You have {len(tool_names)} tools: {', '.join(tool_names)}."
        else:
            tools_line = "You have 8 tools: read_file, write_file, edit_file, bash, glob_search, grep_search, list_dir, task."

        prompt += f"""

## Tools

{tools_line}
USE THEM. Every response to a task should include tool calls. If you respond with only text and no tool calls, you are doing it wrong.

{_AGENTIC_WORKFLOW}

## Rules
"""
        # Select rules tier based on context window size
        if num_ctx >= 8192:
            prompt += "\n" + _FULL_RULES + "\n"
        else:
            prompt += "\n" + _COMPACT_RULES + "\n"

        # Action-not-talk rules (critical for preventing text dumps)
        prompt += "\n" + _ACTION_NOT_TALK + "\n"

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
        from claude1.model_profiles import _PLANNING_RULES
        prompt += f"\n## Planning Mode\n\n{_PLANNING_RULES}\n"

    # Append profile format rules
    if profile and profile.system_prompt_suffix:
        prompt += f"\n## Output Format\n\n{profile.system_prompt_suffix}\n"

    # Add self-awareness section
    self_awareness = _get_self_awareness_section()
    if self_awareness:
        prompt += f"\n{self_awareness}\n"

    # Add CLAUDE.md content
    claude_md = _load_claude_md(working_dir)
    if claude_md:
        prompt += f"\n{claude_md}\n"

    # Compact mode instruction
    if compact:
        prompt += "\n## IMPORTANT: Compact Mode Active\nBe maximally concise. Give the shortest possible responses. Omit pleasantries and explanations unless asked. Use terse language.\n"

    return prompt
