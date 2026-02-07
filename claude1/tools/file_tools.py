"""File operation tools: read, write, edit — with path sandboxing and atomic writes."""

import difflib
import os
import tempfile
from pathlib import Path
from typing import Any

from claude1.config import MAX_FILE_READ_CHARS
from claude1.tools.base import BaseTool, PathSandboxError


def _atomic_write(path: Path, content: str) -> None:
    """Write content to path atomically using a temp file + os.replace().

    Creates a temp file in the same directory as the target, writes content,
    then atomically replaces the target. This prevents corruption on crash/Ctrl+C.
    """
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        os.replace(tmp_path, str(path))
    except BaseException:
        # Clean up temp file on any error
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


class ReadFileTool(BaseTool):
    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return (
            "Read file contents with line numbers (format: '  LINE\\tCONTENT'). "
            f"IMPORTANT: ALWAYS call this before edit_file to see exact content. "
            f"Truncates at {MAX_FILE_READ_CHARS} chars. "
            "Use offset (1-based line number) and limit (number of lines) for large files. "
            "Example: offset=50, limit=20 reads lines 50-69."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path (absolute or relative to working directory)",
                },
                "offset": {
                    "type": "integer",
                    "description": "Starting line number (1-based). Optional.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read. Optional.",
                },
            },
            "required": ["path"],
        }

    def execute(self, **kwargs: Any) -> str:
        path_str = kwargs.get("path", "")
        offset = kwargs.get("offset", 1)
        limit = kwargs.get("limit")

        if not path_str:
            return "Error: path is required"

        try:
            path = self._resolve_path(path_str)
        except PathSandboxError as e:
            return f"Error: {e}"

        if not path.exists():
            return f"Error: File not found: {path}"
        if not path.is_file():
            return f"Error: Not a file: {path}"

        try:
            content = path.read_text(errors="replace")
        except PermissionError:
            return f"Error: Permission denied: {path}"
        except Exception as e:
            return f"Error reading file: {e}"

        lines = content.splitlines()
        total_lines = len(lines)

        # Apply offset (1-based)
        if offset and offset > 1:
            start = offset - 1
        else:
            start = 0

        # Apply limit
        if limit:
            end = start + limit
        else:
            end = total_lines

        selected = lines[start:end]

        # Format with line numbers
        numbered = []
        for i, line in enumerate(selected, start=start + 1):
            numbered.append(f"{i:>6}\t{line}")
        result = "\n".join(numbered)

        if len(result) > MAX_FILE_READ_CHARS:
            result = result[:MAX_FILE_READ_CHARS] + f"\n... [truncated at {MAX_FILE_READ_CHARS} chars]"

        header = f"File: {path} ({total_lines} lines)"
        if start > 0 or end < total_lines:
            header += f" [showing lines {start + 1}-{min(end, total_lines)}]"

        return f"{header}\n{result}"


class WriteFileTool(BaseTool):
    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return (
            "WARNING: Overwrites the ENTIRE file. Creates parent directories if needed. "
            "Use ONLY for brand-new files. For modifying existing files, use edit_file instead. "
            "If you use this on an existing file, all previous content is lost."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path (absolute or relative to working directory)",
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file",
                },
            },
            "required": ["path", "content"],
        }

    @property
    def requires_confirmation(self) -> bool:
        return True

    def execute(self, **kwargs: Any) -> str:
        path_str = kwargs.get("path", "")
        content = kwargs.get("content", "")

        if not path_str:
            return "Error: path is required"

        try:
            path = self._resolve_path(path_str)
        except PathSandboxError as e:
            return f"Error: {e}"

        try:
            # Capture snapshot for undo before writing
            if self.undo_stack is not None:
                self.undo_stack.push(path)

            _atomic_write(path, content)
            lines = content.count("\n") + (1 if content else 0)
            return f"Successfully wrote {len(content)} bytes ({lines} lines) to {path}"
        except PermissionError:
            return f"Error: Permission denied: {path}"
        except Exception as e:
            return f"Error writing file: {e}"


class EditFileTool(BaseTool):
    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return (
            "Edit a file by replacing an exact string match. Critical rules: "
            "(1) ALWAYS read_file first — never guess content. "
            "(2) old_string must match EXACTLY, including whitespace and indentation. "
            "(3) Include 3-5 lines of context so old_string matches uniquely. "
            "(4) If old_string is empty, new_string is inserted at the beginning. "
            "(5) On error, the tool shows partial matches — use them to fix old_string. "
            "Returns a unified diff of the change."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path (absolute or relative to working directory)",
                },
                "old_string": {
                    "type": "string",
                    "description": "The exact string to find and replace. Must match exactly once.",
                },
                "new_string": {
                    "type": "string",
                    "description": "The replacement string.",
                },
            },
            "required": ["path", "old_string", "new_string"],
        }

    @property
    def requires_confirmation(self) -> bool:
        return True

    def execute(self, **kwargs: Any) -> str:
        path_str = kwargs.get("path", "")
        old_string = kwargs.get("old_string", "")
        new_string = kwargs.get("new_string", "")

        if not path_str:
            return "Error: path is required"

        try:
            path = self._resolve_path(path_str)
        except PathSandboxError as e:
            return f"Error: {e}"

        if not path.exists():
            return f"Error: File not found: {path}"
        if not path.is_file():
            return f"Error: Not a file: {path}"

        try:
            content = path.read_text()
        except Exception as e:
            return f"Error reading file: {e}"

        # Empty old_string means insert at beginning
        if old_string == "":
            if self.undo_stack is not None:
                self.undo_stack.push(path)
            new_content = new_string + content
            _atomic_write(path, new_content)
            return f"Successfully inserted text at the beginning of {path}"

        # Count occurrences
        count = content.count(old_string)

        if count == 0:
            # Provide helpful context
            lines = content.splitlines()
            # Try to find partial matches
            first_line = old_string.splitlines()[0] if old_string.splitlines() else old_string
            partial = [
                f"  Line {i + 1}: {line.rstrip()}"
                for i, line in enumerate(lines)
                if first_line.strip() in line
            ]
            hint = ""
            if partial:
                hint = "\n\nPartial matches found:\n" + "\n".join(partial[:5])
            return (
                f"Error: old_string not found in {path}. "
                f"Make sure the string matches exactly, including whitespace and indentation.{hint}"
            )

        if count > 1:
            return (
                f"Error: old_string found {count} times in {path}. "
                f"Provide more surrounding context to make the match unique."
            )

        # Exactly one match - do the replacement
        if self.undo_stack is not None:
            self.undo_stack.push(path)
        new_content = content.replace(old_string, new_string, 1)
        try:
            _atomic_write(path, new_content)
        except Exception as e:
            return f"Error writing file: {e}"

        # Generate unified diff for the tool result
        diff_text = _make_diff(path, content, new_content)
        return f"Successfully edited {path} (replaced 1 occurrence)\n\n{diff_text}"


def _make_diff(path: Path, old_content: str, new_content: str) -> str:
    """Generate a unified diff string between old and new content."""
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines, new_lines,
        fromfile=str(path), tofile=str(path),
        lineterm="",
    )
    return "".join(diff)
