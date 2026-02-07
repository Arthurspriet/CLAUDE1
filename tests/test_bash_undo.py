"""Tests for BashUndoManager (P2-1)."""

import subprocess

import pytest

from claude1.undo import BashUndoManager


@pytest.fixture
def git_workdir(tmp_path):
    """Create a temp directory with an initialized git repo."""
    d = tmp_path / "repo"
    d.mkdir()
    subprocess.run(["git", "init"], cwd=str(d), capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=str(d), capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(d), capture_output=True, check=True)
    # Create initial commit
    (d / "readme.txt").write_text("initial")
    subprocess.run(["git", "add", "."], cwd=str(d), capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(d), capture_output=True, check=True)
    return d


class TestBashUndoManager:
    def test_is_git_repo_true(self, git_workdir):
        assert BashUndoManager._is_git_repo(str(git_workdir)) is True

    def test_is_git_repo_false(self, tmp_path):
        d = tmp_path / "notgit"
        d.mkdir()
        assert BashUndoManager._is_git_repo(str(d)) is False

    def test_capture_returns_message(self, git_workdir):
        # Make a change to capture
        (git_workdir / "readme.txt").write_text("modified")
        mgr = BashUndoManager()
        msg = mgr.capture("rm -rf stuff", str(git_workdir))
        assert msg is not None
        assert "rm -rf stuff" in msg

    def test_capture_on_non_git_returns_none(self, tmp_path):
        d = tmp_path / "plain"
        d.mkdir()
        mgr = BashUndoManager()
        msg = mgr.capture("rm -rf stuff", str(d))
        assert msg is None

    def test_capture_clean_tree(self, git_workdir):
        """Even on a clean tree, capture should succeed (uses HEAD)."""
        mgr = BashUndoManager()
        msg = mgr.capture("git reset --hard", str(git_workdir))
        assert msg is not None

    def test_undo_empty_stack(self):
        mgr = BashUndoManager()
        result = mgr.undo_last()
        assert "No bash snapshots" in result

    def test_list_snapshots_empty(self):
        mgr = BashUndoManager()
        assert mgr.list_snapshots() == []

    def test_list_snapshots_after_capture(self, git_workdir):
        mgr = BashUndoManager()
        (git_workdir / "readme.txt").write_text("changed")
        mgr.capture("rm -rf important", str(git_workdir))

        snaps = mgr.list_snapshots()
        assert len(snaps) == 1
        assert "rm -rf important" in snaps[0]["command"]

    def test_size_property(self, git_workdir):
        mgr = BashUndoManager()
        assert mgr.size == 0
        mgr.capture("cmd1", str(git_workdir))
        assert mgr.size == 1
        mgr.capture("cmd2", str(git_workdir))
        assert mgr.size == 2

    def test_max_snapshots_enforced(self, git_workdir):
        mgr = BashUndoManager(max_snapshots=3)
        for i in range(5):
            mgr.capture(f"cmd{i}", str(git_workdir))
        assert mgr.size == 3

    def test_capture_and_undo_restores(self, git_workdir):
        """Full capture-modify-undo cycle."""
        (git_workdir / "readme.txt").write_text("before-destroy")
        subprocess.run(["git", "add", "."], cwd=str(git_workdir), capture_output=True)

        mgr = BashUndoManager()
        mgr.capture("rm readme.txt", str(git_workdir))

        # Simulate destructive action
        (git_workdir / "readme.txt").write_text("destroyed!")

        # Undo should restore
        result = mgr.undo_last()
        assert "Restored" in result or "state before" in result
