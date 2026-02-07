"""Shared fixtures for claude1 tests."""

import os
import tempfile
from pathlib import Path

import pytest

from claude1.undo import UndoStack


@pytest.fixture
def tmp_workdir(tmp_path: Path) -> Path:
    """Create a temporary working directory for tool tests."""
    workdir = tmp_path / "project"
    workdir.mkdir()
    return workdir


@pytest.fixture
def undo_stack() -> UndoStack:
    """Fresh undo stack."""
    return UndoStack()


@pytest.fixture
def sample_file(tmp_workdir: Path) -> Path:
    """Create a sample source file in the temp workdir."""
    f = tmp_workdir / "hello.py"
    f.write_text('def greet(name):\n    return f"Hello, {name}!"\n')
    return f
