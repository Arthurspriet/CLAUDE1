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
    table.add_row("/undo", "Undo last file edit/write")
    table.add_row("/export [file]", "Export conversation as markdown")
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
