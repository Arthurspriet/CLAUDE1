"""Tests for file tools: path sandboxing, atomic writes, edit correctness, diff display."""

import os
from pathlib import Path

import pytest

from claude1.tools.base import PathSandboxError
from claude1.tools.file_tools import (
    EditFileTool,
    ReadFileTool,
    WriteFileTool,
    _atomic_write,
)
from claude1.undo import UndoStack


class TestPathSandboxing:
    """Test that file tools enforce path sandboxing."""

    def test_read_within_workdir(self, tmp_workdir, sample_file):
        tool = ReadFileTool(str(tmp_workdir))
        result = tool.execute(path=str(sample_file))
        assert "Hello" in result

    def test_read_relative_path(self, tmp_workdir, sample_file):
        tool = ReadFileTool(str(tmp_workdir))
        result = tool.execute(path="hello.py")
        assert "Hello" in result

    def test_read_outside_workdir_blocked(self, tmp_workdir):
        tool = ReadFileTool(str(tmp_workdir))
        result = tool.execute(path="/etc/hostname")
        assert "Access denied" in result or "Error" in result

    def test_read_parent_traversal_blocked(self, tmp_workdir):
        tool = ReadFileTool(str(tmp_workdir))
        result = tool.execute(path="../../.ssh/id_rsa")
        assert "Access denied" in result or "Error" in result

    def test_write_outside_workdir_blocked(self, tmp_workdir):
        tool = WriteFileTool(str(tmp_workdir))
        result = tool.execute(path="/etc/test_file", content="hack")
        assert "Access denied" in result or "Error" in result

    def test_tmp_path_allowed(self, tmp_workdir):
        """Paths under /tmp should be allowed."""
        tool = ReadFileTool(str(tmp_workdir))
        # Create a temp file to test
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test content")
            tmp_file = f.name
        try:
            result = tool.execute(path=tmp_file)
            assert "test content" in result or "Error" not in result
        finally:
            os.unlink(tmp_file)

    def test_sensitive_ssh_blocked(self, tmp_workdir):
        tool = ReadFileTool(str(tmp_workdir))
        home = Path.home()
        ssh_path = home / ".ssh" / "id_rsa"
        result = tool.execute(path=str(ssh_path))
        assert "Access denied" in result or "sensitive" in result.lower() or "Error" in result


class TestAtomicWrite:
    """Test atomic file write behavior."""

    def test_basic_atomic_write(self, tmp_workdir):
        target = tmp_workdir / "output.txt"
        _atomic_write(target, "hello world")
        assert target.read_text() == "hello world"

    def test_atomic_write_creates_parents(self, tmp_workdir):
        target = tmp_workdir / "sub" / "dir" / "file.txt"
        _atomic_write(target, "nested content")
        assert target.read_text() == "nested content"

    def test_atomic_write_overwrites(self, tmp_workdir):
        target = tmp_workdir / "overwrite.txt"
        target.write_text("old content")
        _atomic_write(target, "new content")
        assert target.read_text() == "new content"

    def test_no_temp_file_left_on_success(self, tmp_workdir):
        target = tmp_workdir / "clean.txt"
        _atomic_write(target, "clean")
        tmp_files = list(tmp_workdir.glob("*.tmp"))
        assert len(tmp_files) == 0


class TestReadFileTool:
    """Test read_file tool."""

    def test_read_with_line_numbers(self, tmp_workdir, sample_file):
        tool = ReadFileTool(str(tmp_workdir))
        result = tool.execute(path="hello.py")
        assert "1\t" in result
        assert "def greet" in result

    def test_read_with_offset_and_limit(self, tmp_workdir, sample_file):
        tool = ReadFileTool(str(tmp_workdir))
        result = tool.execute(path="hello.py", offset=2, limit=1)
        assert "return" in result
        assert "showing lines 2-2" in result

    def test_read_nonexistent(self, tmp_workdir):
        tool = ReadFileTool(str(tmp_workdir))
        result = tool.execute(path="nonexistent.py")
        assert "Error" in result

    def test_read_empty_path(self, tmp_workdir):
        tool = ReadFileTool(str(tmp_workdir))
        result = tool.execute(path="")
        assert "Error" in result


class TestWriteFileTool:
    """Test write_file tool."""

    def test_write_new_file(self, tmp_workdir):
        tool = WriteFileTool(str(tmp_workdir), undo_stack=UndoStack())
        result = tool.execute(path="new.txt", content="hello")
        assert "Successfully wrote" in result
        assert (tmp_workdir / "new.txt").read_text() == "hello"

    def test_write_with_undo(self, tmp_workdir):
        stack = UndoStack()
        tool = WriteFileTool(str(tmp_workdir), undo_stack=stack)
        tool.execute(path="new.txt", content="hello")
        assert stack.size == 1

    def test_write_creates_parents(self, tmp_workdir):
        tool = WriteFileTool(str(tmp_workdir))
        result = tool.execute(path="sub/dir/new.txt", content="nested")
        assert "Successfully wrote" in result
        assert (tmp_workdir / "sub" / "dir" / "new.txt").read_text() == "nested"


class TestEditFileTool:
    """Test edit_file tool."""

    def test_basic_edit(self, tmp_workdir, sample_file):
        tool = EditFileTool(str(tmp_workdir), undo_stack=UndoStack())
        result = tool.execute(
            path="hello.py",
            old_string='return f"Hello, {name}!"',
            new_string='return f"Hi, {name}!"',
        )
        assert "Successfully edited" in result
        assert "Hi" in sample_file.read_text()

    def test_edit_returns_diff(self, tmp_workdir, sample_file):
        tool = EditFileTool(str(tmp_workdir))
        result = tool.execute(
            path="hello.py",
            old_string='return f"Hello, {name}!"',
            new_string='return f"Hi, {name}!"',
        )
        assert "---" in result  # unified diff header
        assert "+++" in result
        assert "-" in result  # removed line
        assert "+" in result  # added line

    def test_edit_no_match(self, tmp_workdir, sample_file):
        tool = EditFileTool(str(tmp_workdir))
        result = tool.execute(
            path="hello.py",
            old_string="this does not exist",
            new_string="replacement",
        )
        assert "Error" in result
        assert "not found" in result

    def test_edit_multiple_matches(self, tmp_workdir):
        f = tmp_workdir / "dupe.py"
        f.write_text("x = 1\nx = 1\n")
        tool = EditFileTool(str(tmp_workdir))
        result = tool.execute(
            path="dupe.py",
            old_string="x = 1",
            new_string="x = 2",
        )
        assert "Error" in result
        assert "2 times" in result

    def test_edit_insert_at_beginning(self, tmp_workdir, sample_file):
        tool = EditFileTool(str(tmp_workdir), undo_stack=UndoStack())
        result = tool.execute(
            path="hello.py",
            old_string="",
            new_string="# header\n",
        )
        assert "Successfully inserted" in result
        assert sample_file.read_text().startswith("# header\n")

    def test_edit_with_undo(self, tmp_workdir, sample_file):
        stack = UndoStack()
        tool = EditFileTool(str(tmp_workdir), undo_stack=stack)
        original = sample_file.read_text()
        tool.execute(
            path="hello.py",
            old_string='return f"Hello, {name}!"',
            new_string='return f"Hi, {name}!"',
        )
        stack.undo_last()
        assert sample_file.read_text() == original
