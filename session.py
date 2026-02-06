"""Session save/load for conversation history."""

import json
from datetime import datetime
from pathlib import Path

from config import SESSIONS_DIR


def save_session(messages: list[dict], name: str) -> str:
    """Save conversation history to a JSON file. Returns the file path."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Sanitize name
    safe_name = "".join(c for c in name if c.isalnum() or c in "-_").strip()
    if not safe_name:
        safe_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    filepath = SESSIONS_DIR / f"{safe_name}.json"

    data = {
        "name": name,
        "saved_at": datetime.now().isoformat(),
        "messages": messages,
    }

    filepath.write_text(json.dumps(data, indent=2))
    return str(filepath)


def load_session(name: str) -> list[dict] | None:
    """Load conversation history from a saved session. Returns messages or None."""
    filepath = SESSIONS_DIR / f"{name}.json"

    if not filepath.exists():
        # Try without extension
        matches = list(SESSIONS_DIR.glob(f"{name}*"))
        if matches:
            filepath = matches[0]
        else:
            return None

    try:
        data = json.loads(filepath.read_text())
        return data.get("messages", [])
    except (json.JSONDecodeError, KeyError):
        return None


def list_sessions() -> list[dict]:
    """List all saved sessions with metadata."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sessions = []

    for f in sorted(SESSIONS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        if f.stem == "_autosave":
            continue
        try:
            data = json.loads(f.read_text())
            msg_count = len(data.get("messages", []))
            sessions.append({
                "name": f.stem,
                "saved_at": data.get("saved_at", "unknown"),
                "messages": msg_count,
            })
        except (json.JSONDecodeError, OSError):
            continue

    return sessions


def auto_save_session(messages: list[dict]):
    """Auto-save conversation to _autosave.json."""
    if not messages:
        return
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = SESSIONS_DIR / "_autosave.json"
    data = {
        "name": "_autosave",
        "saved_at": datetime.now().isoformat(),
        "messages": messages,
    }
    try:
        filepath.write_text(json.dumps(data, indent=2))
    except OSError:
        pass


def get_latest_session() -> list[dict] | None:
    """Load the autosaved session. Returns messages or None."""
    filepath = SESSIONS_DIR / "_autosave.json"
    if not filepath.exists():
        return None
    try:
        data = json.loads(filepath.read_text())
        return data.get("messages", [])
    except (json.JSONDecodeError, OSError):
        return None


def add_agent_run_message(messages: list[dict], role: str, model: str,
                          task_id: str, output: str, files_modified: list[str] | None = None) -> None:
    """Append an agent run record to the message history (backward-compatible)."""
    messages.append({
        "role": "assistant",
        "content": output,
        "agent_metadata": {
            "type": "agent_run",
            "agent_role": role,
            "model": model,
            "task_id": task_id,
            "files_modified": files_modified or [],
        },
    })


def export_as_markdown(messages: list[dict], filepath: str | None = None) -> str:
    """Export conversation as a markdown file. Returns the file path."""
    if filepath is None:
        filepath = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

    path = Path(filepath)

    lines = [f"# Conversation Export ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n"]

    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        if role == "system":
            lines.append("## System Prompt\n")
            lines.append(f"```\n{content}\n```\n")
        elif role == "user":
            lines.append("## User\n")
            lines.append(f"{content}\n")
        elif role == "assistant":
            lines.append("## Assistant\n")
            lines.append(f"{content}\n")
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                for tc in tool_calls:
                    func = tc.get("function", {})
                    name = func.get("name", "unknown")
                    args = func.get("arguments", {})
                    lines.append(f"\n**Tool Call: {name}**\n")
                    lines.append(f"```json\n{json.dumps(args, indent=2)}\n```\n")
        elif role == "tool":
            lines.append("## Tool Result\n")
            lines.append(f"```\n{content}\n```\n")

    path.write_text("\n".join(lines))
    return str(path)
