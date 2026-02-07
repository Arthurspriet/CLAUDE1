"""Indexed, searchable view of Claude1's own source code."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

from claude1.self_awareness.introspection import CLAUDE1_ROOT


@dataclass
class SearchResult:
    """A search match in Claude1's source."""

    file: str  # relative path
    line: int
    content: str
    context: str = ""  # surrounding lines


class CodebaseIndex:
    """Searchable index of Claude1's own source code."""

    def __init__(self):
        self.root = CLAUDE1_ROOT

    def search(self, query: str, max_results: int = 20) -> list[SearchResult]:
        """Grep within Claude1's own source code."""
        results: list[SearchResult] = []
        pattern = re.compile(re.escape(query), re.IGNORECASE)

        for py_file in sorted(self.root.rglob("*.py")):
            try:
                lines = py_file.read_text(encoding="utf-8").splitlines()
                relative = str(py_file.relative_to(self.root))

                for i, line in enumerate(lines):
                    if pattern.search(line):
                        # Get context (2 lines before and after)
                        start = max(0, i - 2)
                        end = min(len(lines), i + 3)
                        context = "\n".join(lines[start:end])

                        results.append(SearchResult(
                            file=relative,
                            line=i + 1,
                            content=line.strip(),
                            context=context,
                        ))

                        if len(results) >= max_results:
                            return results
            except OSError:
                continue

        return results

    def get_file(self, relative_path: str) -> str | None:
        """Read a Claude1 source file by relative path."""
        path = self.root / relative_path
        if not path.exists():
            return None
        # Safety: ensure we stay within claude1/
        try:
            path.resolve().relative_to(self.root.resolve())
        except ValueError:
            return None
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return None

    def get_class_source(self, class_name: str) -> str | None:
        """Extract source code of a specific class from the codebase."""
        for py_file in self.root.rglob("*.py"):
            try:
                source = py_file.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(py_file))
                lines = source.splitlines()

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef) and node.name == class_name:
                        start = node.lineno - 1
                        end = node.end_lineno if node.end_lineno else start + 1
                        class_source = "\n".join(lines[start:end])
                        relative = str(py_file.relative_to(self.root))
                        return f"# {relative}:{node.lineno}\n{class_source}"
            except (SyntaxError, OSError):
                continue

        return None

    def list_files(self) -> list[str]:
        """List all Python files in the codebase."""
        files = []
        for py_file in sorted(self.root.rglob("*.py")):
            files.append(str(py_file.relative_to(self.root)))
        return files
