"""File-level locking for concurrent agent writes."""

from __future__ import annotations

import asyncio
from pathlib import Path


class FileLockManager:
    """Manages per-file asyncio locks to prevent concurrent write conflicts.

    Read operations don't acquire locks (concurrent reads are safe).
    Write operations (write_file, edit_file) acquire a lock keyed by absolute path.
    """

    def __init__(self):
        self._locks: dict[str, asyncio.Lock] = {}

    def _get_lock(self, path: str) -> asyncio.Lock:
        """Get or create a lock for the given absolute path."""
        resolved = str(Path(path).resolve())
        if resolved not in self._locks:
            self._locks[resolved] = asyncio.Lock()
        return self._locks[resolved]

    async def acquire(self, path: str) -> None:
        """Acquire the lock for a file path."""
        lock = self._get_lock(path)
        await lock.acquire()

    def release(self, path: str) -> None:
        """Release the lock for a file path."""
        resolved = str(Path(path).resolve())
        lock = self._locks.get(resolved)
        if lock and lock.locked():
            lock.release()

    def is_write_tool(self, tool_name: str) -> bool:
        """Check if a tool name is a write operation that needs locking."""
        return tool_name in ("write_file", "edit_file")

    def get_path_from_args(self, tool_name: str, tool_args: dict) -> str | None:
        """Extract the file path from tool arguments."""
        if tool_name in ("write_file", "edit_file", "read_file"):
            return tool_args.get("path")
        return None
