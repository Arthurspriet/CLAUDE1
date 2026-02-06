"""Undo stack for file edit operations."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class FileSnapshot:
    """Snapshot of a file's state before modification."""
    path: Path
    content: str | None  # None means file didn't exist (was newly created)


class UndoStack:
    """Stack of file snapshots for undo support. Max 50 snapshots."""

    def __init__(self, max_size: int = 50):
        self._stack: list[FileSnapshot] = []
        self._max_size = max_size

    def push(self, path: Path):
        """Capture current file content before modification."""
        path = Path(path).resolve()
        if path.exists() and path.is_file():
            try:
                content = path.read_text()
            except OSError:
                content = None
        else:
            content = None  # File doesn't exist yet (new file)

        self._stack.append(FileSnapshot(path=path, content=content))

        # Trim to max size
        if len(self._stack) > self._max_size:
            self._stack = self._stack[-self._max_size:]

    def undo_last(self) -> str:
        """Undo the last file modification. Returns status message."""
        if not self._stack:
            return "Nothing to undo."

        snapshot = self._stack.pop()

        if snapshot.content is None:
            # File was newly created — delete it
            if snapshot.path.exists():
                try:
                    snapshot.path.unlink()
                    return f"Deleted newly created file: {snapshot.path}"
                except OSError as e:
                    return f"Error deleting {snapshot.path}: {e}"
            else:
                return f"File already gone: {snapshot.path}"
        else:
            # Restore previous content
            try:
                snapshot.path.write_text(snapshot.content)
                return f"Restored: {snapshot.path}"
            except OSError as e:
                return f"Error restoring {snapshot.path}: {e}"

    @property
    def size(self) -> int:
        return len(self._stack)
