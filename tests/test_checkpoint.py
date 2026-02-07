"""Tests for conversation checkpointing (P2-3)."""

import json
from pathlib import Path

import pytest

import claude1.session as session_mod


@pytest.fixture(autouse=True)
def tmp_checkpoints_dir(tmp_path, monkeypatch):
    """Redirect checkpoints to temp directory."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    checkpoints_dir = sessions_dir / "checkpoints"
    checkpoints_dir.mkdir()
    monkeypatch.setattr(session_mod, "SESSIONS_DIR", sessions_dir)
    monkeypatch.setattr(session_mod, "CHECKPOINTS_DIR", checkpoints_dir)
    return checkpoints_dir


SAMPLE_MESSAGES = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
]


class TestCheckpointSaveLoad:
    def test_save_and_load_roundtrip(self):
        path = session_mod.save_checkpoint(SAMPLE_MESSAGES, "test-cp")
        assert Path(path).exists()

        loaded = session_mod.load_checkpoint("test-cp")
        assert loaded is not None
        assert len(loaded) == len(SAMPLE_MESSAGES)
        assert loaded[0]["content"] == "You are helpful."

    def test_save_creates_json_file(self, tmp_checkpoints_dir):
        session_mod.save_checkpoint(SAMPLE_MESSAGES, "my-checkpoint")
        files = list(tmp_checkpoints_dir.glob("*.json"))
        assert len(files) == 1
        assert files[0].stem == "my-checkpoint"

    def test_load_nonexistent_returns_none(self):
        result = session_mod.load_checkpoint("does-not-exist")
        assert result is None

    def test_list_checkpoints_empty(self):
        result = session_mod.list_checkpoints()
        assert result == []

    def test_list_checkpoints_shows_saved(self):
        session_mod.save_checkpoint(SAMPLE_MESSAGES, "cp1")
        session_mod.save_checkpoint(SAMPLE_MESSAGES[:2], "cp2")

        cps = session_mod.list_checkpoints()
        assert len(cps) == 2
        names = {cp["name"] for cp in cps}
        assert "cp1" in names
        assert "cp2" in names

    def test_delete_checkpoint(self, tmp_checkpoints_dir):
        session_mod.save_checkpoint(SAMPLE_MESSAGES, "to-delete")
        assert (tmp_checkpoints_dir / "to-delete.json").exists()

        result = session_mod.delete_checkpoint("to-delete")
        assert result is True
        assert not (tmp_checkpoints_dir / "to-delete.json").exists()

    def test_delete_nonexistent_returns_false(self):
        result = session_mod.delete_checkpoint("nope")
        assert result is False

    def test_save_sanitizes_name(self):
        path = session_mod.save_checkpoint(SAMPLE_MESSAGES, "my check/point!")
        # Should strip invalid chars
        assert Path(path).exists()

    def test_checkpoint_preserves_message_content(self):
        msgs = [
            {"role": "user", "content": "Write code"},
            {"role": "assistant", "content": "```python\nprint('hello')\n```"},
            {"role": "tool", "content": "Executed successfully"},
        ]
        session_mod.save_checkpoint(msgs, "code-cp")
        loaded = session_mod.load_checkpoint("code-cp")
        assert loaded is not None
        assert loaded[1]["content"] == "```python\nprint('hello')\n```"


class TestCheckpointMetadata:
    def test_checkpoint_includes_timestamp(self, tmp_checkpoints_dir):
        session_mod.save_checkpoint(SAMPLE_MESSAGES, "meta-cp")
        data = json.loads((tmp_checkpoints_dir / "meta-cp.json").read_text())
        assert "checkpoint_at" in data
        assert "message_count" in data
        assert data["message_count"] == 3
