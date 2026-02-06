"""Search and directory listing tools."""

import os
import re
from pathlib import Path
from typing import Any

from config import MAX_GLOB_RESULTS, MAX_GREP_MATCHES
from tools.base import BaseTool

# Directories to always skip
SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
    ".eggs", "*.egg-info",
}

# Binary file extensions to skip for grep
BINARY_EXTENSIONS = {
    ".pyc", ".pyo", ".so", ".o", ".a", ".lib", ".dll", ".exe",
    ".bin", ".dat", ".db", ".sqlite", ".sqlite3",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg",
    ".pdf", ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z",
    ".mp3", ".mp4", ".avi", ".mov", ".wav",
    ".woff", ".woff2", ".ttf", ".eot",
}


def _should_skip_dir(name: str) -> bool:
    return name in SKIP_DIRS or name.endswith(".egg-info")


class GlobSearchTool(BaseTool):
    @property
    def name(self) -> str:
        return "glob_search"

    @property
    def description(self) -> str:
        return (
            "Search for files matching a glob pattern. Returns relative file paths. "
            f"Capped at {MAX_GLOB_RESULTS} results. "
            "Auto-skips: .git, __pycache__, node_modules, .venv, dist, build. "
            "Examples: '**/*.py' (all Python files), 'src/**/*.ts', 'tests/test_*.py'. "
            "Prefer this over 'bash find'."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match files (e.g., '**/*.py')",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (default: working directory)",
                },
            },
            "required": ["pattern"],
        }

    def execute(self, **kwargs: Any) -> str:
        pattern = kwargs.get("pattern", "")
        path_str = kwargs.get("path", "")

        if not pattern:
            return "Error: pattern is required"

        search_dir = self._resolve_path(path_str) if path_str else self.working_dir

        if not search_dir.exists():
            return f"Error: Directory not found: {search_dir}"

        try:
            matches = []
            for match in sorted(search_dir.glob(pattern)):
                # Skip hidden/unwanted directories
                parts = match.relative_to(search_dir).parts
                if any(_should_skip_dir(p) for p in parts):
                    continue
                try:
                    rel = match.relative_to(self.working_dir)
                except ValueError:
                    rel = match
                matches.append(str(rel))
                if len(matches) >= MAX_GLOB_RESULTS:
                    break

            if not matches:
                return f"No files found matching pattern: {pattern}"

            result = "\n".join(matches)
            if len(matches) == MAX_GLOB_RESULTS:
                result += f"\n... [capped at {MAX_GLOB_RESULTS} results]"

            return f"Found {len(matches)} file(s):\n{result}"

        except Exception as e:
            return f"Error during glob search: {e}"


class GrepSearchTool(BaseTool):
    @property
    def name(self) -> str:
        return "grep_search"

    @property
    def description(self) -> str:
        return (
            "Search file contents using regex. Returns 'file:line: content' for each match. "
            f"Capped at {MAX_GREP_MATCHES} matches. Skips binary files and files >1MB. "
            "Use the 'glob' parameter to filter files (e.g., glob='*.py'). "
            "Use 'context_lines' to show surrounding code. "
            "Prefer this over 'bash grep' or 'bash rg'."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regular expression pattern to search for",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search in (default: working directory)",
                },
                "glob": {
                    "type": "string",
                    "description": "Optional glob to filter files (e.g., '*.py')",
                },
                "case_insensitive": {
                    "type": "boolean",
                    "description": "If true, search is case-insensitive (default: false)",
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Number of context lines to show before and after each match",
                },
                "type": {
                    "type": "string",
                    "description": "File type filter (e.g., 'py', 'js', 'ts'). Maps to glob pattern.",
                },
            },
            "required": ["pattern"],
        }

    def execute(self, **kwargs: Any) -> str:
        pattern_str = kwargs.get("pattern", "")
        path_str = kwargs.get("path", "")
        file_glob = kwargs.get("glob", "")
        case_insensitive = kwargs.get("case_insensitive", False)
        context_lines = kwargs.get("context_lines", 0)
        file_type = kwargs.get("type", "")

        if not pattern_str:
            return "Error: pattern is required"

        # Map type to glob
        if file_type and not file_glob:
            file_glob = f"*.{file_type}"

        flags = re.IGNORECASE if case_insensitive else 0
        try:
            regex = re.compile(pattern_str, flags)
        except re.error as e:
            return f"Error: Invalid regex pattern: {e}"

        search_path = self._resolve_path(path_str) if path_str else self.working_dir

        if not search_path.exists():
            return f"Error: Path not found: {search_path}"

        matches = []
        match_count = 0

        # If it's a single file, search just that file
        if search_path.is_file():
            files = [search_path]
        else:
            files = self._collect_files(search_path, file_glob)

        for fpath in files:
            if match_count >= MAX_GREP_MATCHES:
                break
            try:
                content = fpath.read_text(errors="replace")
                lines = content.splitlines()
                try:
                    rel = fpath.relative_to(self.working_dir)
                except ValueError:
                    rel = fpath

                if context_lines and context_lines > 0:
                    # Context mode: collect ranges around matches
                    match_indices = []
                    for i, line in enumerate(lines):
                        if regex.search(line):
                            match_indices.append(i)
                            match_count += 1
                            if match_count >= MAX_GREP_MATCHES:
                                break

                    if match_indices:
                        # Build ranges with context
                        ranges = []
                        for idx in match_indices:
                            start = max(0, idx - context_lines)
                            end = min(len(lines), idx + context_lines + 1)
                            ranges.append((start, end, idx))

                        # Merge overlapping ranges
                        merged = [ranges[0]]
                        for start, end, idx in ranges[1:]:
                            prev_start, prev_end, _ = merged[-1]
                            if start <= prev_end:
                                merged[-1] = (prev_start, max(prev_end, end), idx)
                            else:
                                merged.append((start, end, idx))

                        for i, (start, end, _) in enumerate(merged):
                            if i > 0:
                                matches.append("---")
                            for j in range(start, end):
                                prefix = ">" if j in match_indices else " "
                                matches.append(f"{prefix} {rel}:{j + 1}: {lines[j].rstrip()}")
                else:
                    # Simple mode: just matching lines
                    for i, line in enumerate(lines):
                        if regex.search(line):
                            matches.append(f"{rel}:{i + 1}: {line.rstrip()}")
                            match_count += 1
                            if match_count >= MAX_GREP_MATCHES:
                                break
            except (PermissionError, OSError):
                continue

        if not matches:
            return f"No matches found for pattern: {pattern_str}"

        result = "\n".join(matches)
        if match_count >= MAX_GREP_MATCHES:
            result += f"\n... [capped at {MAX_GREP_MATCHES} matches]"

        return f"Found {match_count} match(es):\n{result}"

    def _collect_files(self, directory: Path, file_glob: str) -> list[Path]:
        """Collect files to search, respecting skip rules."""
        files = []
        for root, dirs, filenames in os.walk(directory):
            # Filter out directories in-place
            dirs[:] = [d for d in dirs if not _should_skip_dir(d) and not d.startswith(".")]

            for fname in filenames:
                fpath = Path(root) / fname
                # Skip binary files
                if fpath.suffix.lower() in BINARY_EXTENSIONS:
                    continue
                # Skip large files (>1MB)
                try:
                    if fpath.stat().st_size > 1_000_000:
                        continue
                except OSError:
                    continue
                # Apply glob filter
                if file_glob and not fpath.match(file_glob):
                    continue
                files.append(fpath)

        return sorted(files)


class ListDirTool(BaseTool):
    @property
    def name(self) -> str:
        return "list_dir"

    @property
    def description(self) -> str:
        return (
            "List directory contents (non-recursive). Shows files with sizes and subdirectories. "
            "Directories listed first. For finding files by pattern, use glob_search instead."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path (default: working directory)",
                },
            },
            "required": [],
        }

    def execute(self, **kwargs: Any) -> str:
        path_str = kwargs.get("path", "")
        directory = self._resolve_path(path_str) if path_str else self.working_dir

        if not directory.exists():
            return f"Error: Directory not found: {directory}"
        if not directory.is_dir():
            return f"Error: Not a directory: {directory}"

        try:
            entries = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except PermissionError:
            return f"Error: Permission denied: {directory}"

        if not entries:
            return f"Directory is empty: {directory}"

        lines = []
        for entry in entries:
            if entry.is_dir():
                lines.append(f"  {entry.name}/")
            else:
                try:
                    size = entry.stat().st_size
                    lines.append(f"  {entry.name}  ({_format_size(size)})")
                except OSError:
                    lines.append(f"  {entry.name}")

        return f"Contents of {directory}:\n" + "\n".join(lines)


def _format_size(size: int) -> str:
    """Format file size in human-readable form."""
    if size < 1024:
        return f"{size} B"
    elif size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    elif size < 1024 * 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    else:
        return f"{size / (1024 * 1024 * 1024):.1f} GB"
