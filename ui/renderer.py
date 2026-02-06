"""Rich-based terminal UI rendering."""

import difflib
import json
import os

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.status import Status
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text


console = Console()


def show_welcome(model: str, working_dir: str, model_info: dict | None = None):
    """Display welcome banner."""
    console.print()

    info_line = f"Model: [green]{model}[/green]"
    if model_info:
        parts = []
        if model_info.get("parameter_size"):
            parts.append(model_info["parameter_size"])
        if model_info.get("quantization_level"):
            parts.append(model_info["quantization_level"])
        if model_info.get("family"):
            parts.append(model_info["family"])
        if parts:
            info_line += f" [dim]({', '.join(parts)})[/dim]"

    console.print(
        Panel(
            f"[bold cyan]Claude1[/bold cyan] - Local Coding Assistant\n"
            f"{info_line}  |  Dir: [dim]{working_dir}[/dim]\n"
            f"Type [bold]/help[/bold] for commands, [bold]Ctrl+C[/bold] to cancel, [bold]Ctrl+D[/bold] to exit",
            border_style="cyan",
            padding=(1, 2),
        )
    )
    console.print()


def show_help():
    """Display help table."""
    table = Table(title="Commands", border_style="dim")
    table.add_column("Command", style="bold cyan", no_wrap=True)
    table.add_column("Description")
    table.add_row("/help", "Show this help message")
    table.add_row("/model <name>", "Switch to a different model")
    table.add_row("/models", "List available Ollama models")
    table.add_row("/clear", "Clear conversation history")
    table.add_row("/save <name>", "Save current session")
    table.add_row("/load <name>", "Load a saved session")
    table.add_row("/sessions", "List saved sessions")
    table.add_row("/resume", "Resume last auto-saved session")
    table.add_row("/auto", "Toggle auto-accept for tool confirmations")
    table.add_row("/temp [value]", "Get/set temperature (0.0-2.0)")
    table.add_row("/profile", "Show current model profile settings")
    table.add_row("/compact", "Toggle compact mode (terse responses)")
    table.add_row("/plan", "Toggle planning mode (reflect before tools)")
    table.add_row("/undo", "Undo last file edit/write")
    table.add_row("/export [file]", "Export conversation as markdown")
    table.add_row("/doctor", "Run health checks")
    table.add_row("/debug", "Show session debug info")
    table.add_row("/context", "Visualize context window usage")
    table.add_row("/stats (/cost)", "Show detailed token stats")
    table.add_row("/copy", "Copy last response to clipboard")
    table.add_row("/rewind [N]", "Rewind conversation by N user turns")
    table.add_row("/memory [cmd]", "Manage CLAUDE.md (show/add/edit/global/reset)")
    table.add_row("/init", "Initialize project CLAUDE.md")
    table.add_row("/skills [cmd]", "Manage skills (list/create/edit/reload/info)")
    table.add_row("/tasks", "List spawned subtasks")
    table.add_row("/agents", "Toggle multi-agent mode")
    table.add_row("/agents config", "Show agent role -> model mappings")
    table.add_row("/agents set <role> <model>", "Change model for an agent role")
    table.add_row("/exit", "Exit Claude1")
    table.add_row("", "")
    table.add_row("[bold]Shortcuts[/bold]", "")
    table.add_row("Ctrl+C", "Cancel current generation")
    table.add_row("Ctrl+D", "Exit")
    table.add_row("Esc+Enter", "Insert newline in input")
    table.add_row("Up/Down", "Navigate command history")
    table.add_row("Ctrl+R", "Search command history")
    console.print(table)
    console.print()


def show_plan(plan_text: str):
    """Display the planning step panel."""
    console.print(
        Panel(
            Markdown(plan_text),
            title="[bold magenta]Plan[/bold magenta]",
            border_style="magenta",
            padding=(0, 1),
        )
    )


def show_tool_call(tool_name: str, tool_args: dict):
    """Display a tool call panel."""
    args_display = _format_tool_args(tool_name, tool_args)
    console.print(
        Panel(
            args_display,
            title=f"[bold yellow]Tool: {tool_name}[/bold yellow]",
            border_style="yellow",
            padding=(0, 1),
        )
    )


def show_tool_result(tool_name: str, result: str, tool_args: dict | None = None):
    """Display a tool result panel with optional syntax highlighting."""
    # Truncate for display
    if len(result) > 2000:
        display = result[:2000] + f"\n... [{len(result)} chars total]"
    else:
        display = result

    renderable = _format_tool_result(tool_name, display, tool_args)

    console.print(
        Panel(
            renderable,
            title=f"[dim]Result: {tool_name}[/dim]",
            border_style="dim",
            padding=(0, 1),
        )
    )


def _format_tool_result(tool_name: str, display: str, tool_args: dict | None = None) -> Text | Syntax:
    """Format tool result with syntax highlighting where appropriate."""
    if tool_name == "read_file" and tool_args:
        # Extract file extension for syntax highlighting
        path = tool_args.get("path", "")
        ext = os.path.splitext(path)[1].lstrip(".")
        if ext:
            # Strip line number prefixes for syntax highlighting
            lines = display.splitlines()
            code_lines = []
            for line in lines:
                # Skip header line
                if line.startswith("File:"):
                    continue
                # Remove line number prefix (e.g., "     1\t")
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    code_lines.append(parts[1])
                else:
                    code_lines.append(line)
            code = "\n".join(code_lines)
            try:
                return Syntax(code, ext, theme="monokai", line_numbers=False)
            except Exception:
                pass

    if tool_name == "bash":
        try:
            return Syntax(display, "bash", theme="monokai", line_numbers=False)
        except Exception:
            pass

    return Text(display, style="dim")


def confirm_tool(tool_name: str, tool_args: dict) -> tuple[bool, bool]:
    """Show tool call and ask for confirmation. Returns (approved, always)."""
    args_display = _format_tool_args(tool_name, tool_args)
    console.print(
        Panel(
            args_display,
            title=f"[bold yellow]Tool: {tool_name}[/bold yellow]",
            border_style="yellow",
            padding=(0, 1),
        )
    )
    try:
        response = console.input("[bold yellow]Allow? (y)es / (n)o / (a)lways: [/bold yellow]").strip().lower()
        if response in ("a", "always"):
            return True, True
        return response in ("y", "yes", ""), False
    except (EOFError, KeyboardInterrupt):
        return False, False


def render_final_text(text: str):
    """Render the final markdown text."""
    if text.strip():
        console.print(Markdown(text.strip()))


def _format_tool_args(tool_name: str, tool_args: dict) -> str | Text:
    """Format tool arguments for display."""
    if tool_name == "bash":
        return tool_args.get("command", str(tool_args))
    elif tool_name == "read_file":
        path = tool_args.get("path", "")
        extra = ""
        if "offset" in tool_args:
            extra += f" (from line {tool_args['offset']})"
        if "limit" in tool_args:
            extra += f" (limit {tool_args['limit']} lines)"
        return f"{path}{extra}"
    elif tool_name == "write_file":
        path = tool_args.get("path", "")
        content = tool_args.get("content", "")
        lines = content.count("\n") + 1
        preview = content[:500]
        if len(content) > 500:
            preview += f"\n... [{len(content)} chars total]"
        return f"{path} ({lines} lines)\n{preview}"
    elif tool_name == "edit_file":
        path = tool_args.get("path", "")
        old = tool_args.get("old_string", "")
        new = tool_args.get("new_string", "")
        return _format_edit_diff(path, old, new)
    elif tool_name in ("glob_search", "grep_search"):
        return tool_args.get("pattern", str(tool_args))
    elif tool_name == "list_dir":
        return tool_args.get("path", ".") or "."
    else:
        return json.dumps(tool_args, indent=2)


def _format_edit_diff(path: str, old: str, new: str) -> Text:
    """Format edit_file args as a unified diff."""
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)

    diff = difflib.unified_diff(
        old_lines, new_lines,
        fromfile=path, tofile=path,
        lineterm="",
    )

    text = Text()
    text.append(f"{path}\n", style="bold")
    has_diff = False
    for line in diff:
        has_diff = True
        line_stripped = line.rstrip("\n")
        if line_stripped.startswith("---") or line_stripped.startswith("+++"):
            text.append(line_stripped + "\n", style="bold")
        elif line_stripped.startswith("@@"):
            text.append(line_stripped + "\n", style="cyan")
        elif line_stripped.startswith("-"):
            text.append(line_stripped + "\n", style="red")
        elif line_stripped.startswith("+"):
            text.append(line_stripped + "\n", style="green")
        else:
            text.append(line_stripped + "\n")

    if not has_diff:
        # Fallback if diff is empty (identical strings)
        text.append(f"[red]- {old}[/red]\n[green]+ {new}[/green]\n")

    return text


def start_thinking_spinner() -> Status:
    """Start a thinking spinner. Returns the Status object to stop later."""
    status = Status("[dim]Thinking...[/dim]", spinner="dots", console=console)
    status.start()
    return status


def stop_thinking_spinner(status: Status | None):
    """Stop the thinking spinner."""
    if status is not None:
        status.stop()


def show_stats(turn_prompt: int, turn_completion: int, eval_duration: int,
               session_total: int, session_cost: float):
    """Display token stats for the turn and session."""
    # Calculate tokens/s from eval_duration (nanoseconds)
    tok_s = ""
    if eval_duration > 0 and turn_completion > 0:
        speed = turn_completion / (eval_duration / 1e9)
        tok_s = f" ({speed:.1f} tok/s)"

    parts = [
        f"tokens: {turn_prompt} in / {turn_completion} out{tok_s}",
        f"session: {session_total} total",
    ]
    if session_cost > 0:
        parts.append(f"~${session_cost:.4f} saved vs API")

    console.print(f"[dim]{' | '.join(parts)}[/dim]")


def show_debug(message: str):
    """Display a debug message."""
    console.print(f"[dim magenta][debug] {message}[/dim magenta]")


def show_error(message: str):
    """Display an error message."""
    console.print(f"[bold red]Error:[/bold red] {message}")


def show_info(message: str):
    """Display an info message."""
    console.print(f"[dim]{message}[/dim]")


def show_model_changed(model: str):
    """Display model change confirmation."""
    console.print(f"[green]Switched to model:[/green] [bold]{model}[/bold]")


def show_models(models: list[str], current: str):
    """Display available models."""
    console.print("[bold]Available models:[/bold]")
    for m in models:
        marker = " [cyan]<-- current[/cyan]" if m == current else ""
        console.print(f"  {m}{marker}")
    console.print()


# ── New rendering functions ──────────────────────────────────────────────────


def show_doctor_results(results):
    """Display health check results."""
    table = Table(title="Health Checks", border_style="dim")
    table.add_column("Status", width=4, justify="center")
    table.add_column("Check", style="bold")
    table.add_column("Result")
    table.add_column("Details", style="dim")

    passed = 0
    total = len(results)

    for r in results:
        icon = "[green]OK[/green]" if r.passed else "[red]FAIL[/red]"
        if r.passed:
            passed += 1
        table.add_row(icon, r.name, r.message, r.details or "")

    console.print(table)
    color = "green" if passed == total else "yellow" if passed >= total - 2 else "red"
    console.print(f"[{color}]{passed}/{total} checks passed[/{color}]")
    console.print()


def show_debug_info(data: dict):
    """Display session debug information."""
    # Model info
    model_text = Text()
    model_text.append(f"Model: {data.get('model', 'unknown')}\n")
    model_text.append(f"Working dir: {data.get('working_dir', 'unknown')}\n")
    model_text.append(f"Auto-accept: {data.get('auto_accept', False)}\n")
    model_text.append(f"Compact: {data.get('compact', False)}\n")
    model_text.append(f"Planning: {data.get('planning', False)}\n")
    model_text.append(f"Temperature: {data.get('temperature', 'default')}\n")
    console.print(Panel(model_text, title="[bold]Config[/bold]", border_style="blue"))

    # Conversation stats
    conv_text = Text()
    conv_text.append(f"Messages: {data.get('message_count', 0)}\n")
    roles = data.get('role_breakdown', {})
    for role, count in roles.items():
        conv_text.append(f"  {role}: {count}\n")
    conv_text.append(f"Estimated tokens: {data.get('estimated_tokens', 0)}\n")
    console.print(Panel(conv_text, title="[bold]Conversation[/bold]", border_style="blue"))

    # Recent tool calls
    tool_history = data.get('tool_history', [])
    if tool_history:
        tools_text = Text()
        for entry in tool_history[-10:]:
            tools_text.append(f"  {entry}\n")
        console.print(Panel(tools_text, title="[bold]Recent Tool Calls[/bold]", border_style="blue"))

    console.print()


def show_context_usage(usage: dict):
    """Display context window usage visualization."""
    estimated = usage.get("estimated_tokens", 0)
    num_ctx = usage.get("num_ctx", 0)
    usable = usage.get("usable", 0)
    reserve = usage.get("reserve", 0)
    pct = usage.get("usage_pct", 0)
    msg_count = usage.get("message_count", 0)

    # Summary
    console.print(f"[bold]Context Window:[/bold] {estimated} / {usable} tokens ({pct:.1f}% used)")
    console.print(f"[dim]Total: {num_ctx} | Reserve: {reserve} | Messages: {msg_count}[/dim]")

    # Colored bar
    bar_width = 50
    filled = int(bar_width * min(pct, 100) / 100)
    empty = bar_width - filled

    if pct < 60:
        color = "green"
    elif pct < 85:
        color = "yellow"
    else:
        color = "red"

    bar = Text()
    bar.append("[")
    bar.append("=" * filled, style=color)
    bar.append(" " * empty)
    bar.append(f"] {pct:.0f}%")
    console.print(bar)

    # Role breakdown
    roles = usage.get("role_breakdown", {})
    if roles:
        table = Table(border_style="dim", show_header=True)
        table.add_column("Role", style="bold")
        table.add_column("Est. Tokens", justify="right")
        for role, tokens in sorted(roles.items()):
            table.add_row(role, str(tokens))
        console.print(table)

    console.print()


def show_session_stats(stats):
    """Display detailed session statistics."""
    # Session summary
    table = Table(title="Session Summary", border_style="dim")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Total requests", str(stats.total_requests))
    table.add_row("Prompt tokens", f"{stats.total_prompt_tokens:,}")
    table.add_row("Completion tokens", f"{stats.total_completion_tokens:,}")
    table.add_row("Total tokens", f"{stats.total_tokens:,}")
    table.add_row("Est. API cost saved", f"${stats.estimated_cost:.4f}")
    console.print(table)

    # Per-turn table
    if stats.turns:
        turn_table = Table(title="Per-Turn Stats", border_style="dim")
        turn_table.add_column("#", justify="right", style="dim")
        turn_table.add_column("Prompt", justify="right")
        turn_table.add_column("Completion", justify="right")
        turn_table.add_column("Speed", justify="right")

        for i, turn in enumerate(stats.turns, 1):
            speed = f"{turn.tokens_per_second:.1f} tok/s" if turn.tokens_per_second > 0 else "-"
            turn_table.add_row(
                str(i),
                f"{turn.prompt_tokens:,}",
                f"{turn.completion_tokens:,}",
                speed,
            )

        console.print(turn_table)

        # Unicode bar chart of completion tokens
        if len(stats.turns) > 1:
            max_comp = max(t.completion_tokens for t in stats.turns) or 1
            console.print("[bold]Completion tokens per turn:[/bold]")
            for i, turn in enumerate(stats.turns, 1):
                bar_len = int(30 * turn.completion_tokens / max_comp)
                bar = "#" * bar_len
                console.print(f"  [dim]{i:>3}[/dim] [cyan]{bar}[/cyan] {turn.completion_tokens}")

    console.print()


def show_rewind_options(turns: list[tuple[int, str]]):
    """Show numbered list of user turns for rewind selection."""
    console.print("[bold]User turns (most recent first):[/bold]")
    for num, preview in turns:
        console.print(f"  [cyan]{num}[/cyan]. {preview}")
    console.print("[dim]Use /rewind N to rewind to after turn N[/dim]")
    console.print()


def show_skills(skills):
    """Display list of available skills."""
    if not skills:
        console.print("[dim]No skills found. Use /skills create <name> to create one.[/dim]")
        console.print()
        return

    table = Table(title="Available Skills", border_style="dim")
    table.add_column("Command", style="bold cyan")
    table.add_column("Description")
    table.add_column("Location", style="dim")

    for skill in skills:
        table.add_row(f"/{skill.name}", skill.description, str(skill.path))

    console.print(table)
    console.print()


def show_skill_activated(name: str, description: str):
    """Show panel when a skill is activated."""
    console.print(
        Panel(
            f"{description}",
            title=f"[bold green]Skill: {name}[/bold green]",
            border_style="green",
            padding=(0, 1),
        )
    )


def show_skill_created(name: str, path):
    """Show confirmation after skill creation."""
    console.print(f"[green]Skill created:[/green] [bold]{name}[/bold]")
    console.print(f"[dim]File: {path}[/dim]")
    console.print()


def show_skill_info(skill):
    """Show detailed info about a skill."""
    text = Text()
    text.append(f"Name: {skill.name}\n", style="bold")
    text.append(f"Description: {skill.description}\n")
    text.append(f"Path: {skill.path}\n", style="dim")
    if skill.allowed_tools:
        text.append(f"Allowed tools: {', '.join(skill.allowed_tools)}\n")
    else:
        text.append("Allowed tools: all\n")
    text.append(f"\nBody preview:\n", style="bold")
    preview = skill.body[:500]
    if len(skill.body) > 500:
        preview += f"\n... [{len(skill.body)} chars total]"
    text.append(preview, style="dim")
    console.print(Panel(text, title=f"[bold]Skill: {skill.name}[/bold]", border_style="cyan"))
    console.print()


def show_tasks(tasks):
    """Display list of spawned subtasks."""
    if not tasks:
        console.print("[dim]No subtasks have been spawned yet.[/dim]")
        console.print()
        return

    table = Table(title="Subtasks", border_style="dim")
    table.add_column("ID", justify="right", style="bold")
    table.add_column("Description")
    table.add_column("Status")
    table.add_column("Duration", justify="right")
    table.add_column("Result Preview", style="dim", max_width=40)

    for task in tasks:
        # Status color
        if task.status == "completed":
            status = "[green]completed[/green]"
        elif task.status == "failed":
            status = "[red]failed[/red]"
        else:
            status = "[yellow]running[/yellow]"

        # Duration
        if task.completed_at > 0:
            duration = f"{task.completed_at - task.started_at:.1f}s"
        else:
            import time
            duration = f"{time.time() - task.started_at:.1f}s (running)"

        # Result preview
        preview = task.result[:80].replace("\n", " ") if task.result else "-"

        table.add_row(str(task.id), task.description, status, duration, preview)

    console.print(table)
    console.print()


# ── Agent mode rendering ───────────────────────────────────────────────────


def show_agent_plan(summary: str, tasks: list) -> None:
    """Display the agent decomposition plan."""
    lines = [f"[bold]{summary}[/bold]\n"]
    for i, task in enumerate(tasks, 1):
        role = task.role.value if hasattr(task.role, "value") else task.role
        deps = f" (after: {', '.join(task.depends_on)})" if task.depends_on else ""
        lines.append(f"  {i}. [{role}] {task.instruction}{deps}")

    console.print(
        Panel(
            "\n".join(lines),
            title="[bold blue]Agent Plan[/bold blue]",
            border_style="blue",
            padding=(0, 1),
        )
    )


def show_agent_progress(task_id: str, role: str, model: str, status: str) -> None:
    """Display a status update for an agent task."""
    status_styles = {
        "running": "[yellow]running[/yellow]",
        "completed": "[green]completed[/green]",
        "failed": "[red]failed[/red]",
    }
    styled = status_styles.get(status, status)
    console.print(f"  [{task_id}] {role} ({model}): {styled}")


def render_agent_result(output: str) -> None:
    """Render the final synthesized agent output as markdown."""
    if output.strip():
        console.print(
            Panel(
                Markdown(output.strip()),
                title="[bold green]Agent Result[/bold green]",
                border_style="green",
                padding=(0, 1),
            )
        )


def show_agent_config(configs: dict) -> None:
    """Display the current agent role -> model mappings."""
    table = Table(title="Agent Configuration", border_style="dim")
    table.add_column("Role", style="bold cyan", no_wrap=True)
    table.add_column("Model", style="green")
    table.add_column("Tools", style="dim")
    table.add_column("Max Iter", style="dim", justify="right")

    for role_name, config in configs.items():
        tool_count = len(config.tool_names) if config.tool_names else 0
        tool_label = f"{tool_count} tools" if tool_count else "none"
        if config.read_only and tool_count:
            tool_label += " (read-only)"
        table.add_row(role_name, config.model, tool_label, str(config.max_iterations))

    console.print(table)
    console.print()
