"""Undo stack for file edit operations and bash command undo via git snapshots."""

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class FileSnapshot:
    """Snapshot of a file's state before modification."""
    path: Path
    content: str | None  # None means file didn't exist (was newly created)


@dataclass
class BashSnapshot:
    """Snapshot of working directory state before a destructive bash command."""
    command: str
    stash_ref: str  # git stash ref (from `git stash create`)
    working_dir: str
    had_untracked: bool = False
    untracked_stash_ref: str = ""  # separate stash for untracked files


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


class FileTransaction:
    """Context manager for multi-file atomic operations.

    Captures snapshots of all files before modification. If any operation
    fails (exception is raised), all files are rolled back to their
    pre-transaction state.

    Usage:
        with FileTransaction() as txn:
            txn.track(path1)
            path1.write_text("new content")
            txn.track(path2)
            path2.write_text("other content")
            # If an exception occurs here, both files are restored
    """

    def __init__(self):
        self._snapshots: list[FileSnapshot] = []
        self._committed = False

    def track(self, path: Path) -> None:
        """Capture the current state of a file before modifying it."""
        path = Path(path).resolve()
        if path.exists() and path.is_file():
            try:
                content = path.read_text()
            except OSError:
                content = None
        else:
            content = None
        self._snapshots.append(FileSnapshot(path=path, content=content))

    def commit(self) -> None:
        """Mark the transaction as successful. No rollback will occur."""
        self._committed = True

    def rollback(self) -> list[str]:
        """Manually roll back all tracked files. Returns status messages."""
        messages = []
        for snapshot in reversed(self._snapshots):
            if snapshot.content is None:
                # File was newly created — delete it
                if snapshot.path.exists():
                    try:
                        snapshot.path.unlink()
                        messages.append(f"Deleted: {snapshot.path}")
                    except OSError as e:
                        messages.append(f"Error deleting {snapshot.path}: {e}")
            else:
                # Restore previous content
                try:
                    snapshot.path.write_text(snapshot.content)
                    messages.append(f"Restored: {snapshot.path}")
                except OSError as e:
                    messages.append(f"Error restoring {snapshot.path}: {e}")
        self._snapshots.clear()
        return messages

    @property
    def tracked_files(self) -> list[Path]:
        """Return list of tracked file paths."""
        return [s.path for s in self._snapshots]

    def __enter__(self) -> "FileTransaction":
        return self

    def __exit__(self, exc_type, _exc_val, _exc_tb) -> bool:
        if exc_type is not None and not self._committed:
            # Exception occurred — roll back
            self.rollback()
        return False  # Don't suppress the exception


class BashUndoManager:
    """Captures git snapshots before destructive bash commands for rollback.

    Uses `git stash create` to capture the working tree state without
    modifying the index. Snapshots are stored in a stack with max 10 entries.
    """

    def __init__(self, max_snapshots: int = 10):
        self._stack: list[BashSnapshot] = []
        self._max_snapshots = max_snapshots

    @staticmethod
    def _is_git_repo(working_dir: str) -> bool:
        """Check if working_dir is inside a git repository."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=working_dir,
                capture_output=True, text=True, timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def capture(self, command: str, working_dir: str) -> Optional[str]:
        """Capture a git snapshot before running a destructive command.

        Returns a status message, or None if capture was not possible.
        """
        if not self._is_git_repo(working_dir):
            return None

        # Create a stash entry without modifying the working tree
        try:
            result = subprocess.run(
                ["git", "stash", "create"],
                cwd=working_dir,
                capture_output=True, text=True, timeout=10,
            )
            stash_ref = result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None

        if not stash_ref:
            # No changes to stash (clean working tree) — use HEAD
            try:
                result = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=working_dir,
                    capture_output=True, text=True, timeout=5,
                )
                stash_ref = result.stdout.strip()
            except (FileNotFoundError, subprocess.TimeoutExpired):
                return None

        if not stash_ref:
            return None

        # Store the snapshot ref so it won't be garbage-collected
        try:
            subprocess.run(
                ["git", "stash", "store", "-m", f"bash-undo: {command[:80]}", stash_ref],
                cwd=working_dir,
                capture_output=True, text=True, timeout=5,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass  # Not critical — the ref still exists temporarily

        snapshot = BashSnapshot(
            command=command,
            stash_ref=stash_ref,
            working_dir=working_dir,
        )
        self._stack.append(snapshot)

        # Trim to max size
        if len(self._stack) > self._max_snapshots:
            self._stack = self._stack[-self._max_snapshots:]

        return f"Snapshot captured before: {command[:60]}"

    def undo_last(self) -> str:
        """Restore the last bash snapshot. Returns status message."""
        if not self._stack:
            return "No bash snapshots to undo."

        snapshot = self._stack.pop()

        if not self._is_git_repo(snapshot.working_dir):
            return "Error: directory is no longer a git repository."

        try:
            # Apply the stash to restore the working tree
            result = subprocess.run(
                ["git", "stash", "apply", snapshot.stash_ref],
                cwd=snapshot.working_dir,
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                return (
                    f"Restored working tree to state before: {snapshot.command[:60]}\n"
                    f"(ref: {snapshot.stash_ref[:12]})"
                )
            else:
                return (
                    f"Error restoring snapshot: {result.stderr.strip()}\n"
                    f"You can try manually: git stash apply {snapshot.stash_ref}"
                )
        except subprocess.TimeoutExpired:
            return f"Timeout restoring snapshot. Try: git stash apply {snapshot.stash_ref}"
        except FileNotFoundError:
            return "Error: git not found."

    def list_snapshots(self) -> list[dict]:
        """Return a list of available bash snapshots."""
        return [
            {
                "command": s.command[:80],
                "ref": s.stash_ref[:12],
                "working_dir": s.working_dir,
            }
            for s in reversed(self._stack)
        ]

    @property
    def size(self) -> int:
        return len(self._stack)
