"""Tests for session save/load roundtrip."""

import json
from pathlib import Path

from claude1 import session as session_mod
from claude1.session import (
    auto_save_session,
    get_latest_session,
    list_sessions,
    load_session,
    save_session,
)


class TestSessionSaveLoad:
    """Test session persistence."""

    def test_save_and_load_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(session_mod, "SESSIONS_DIR", tmp_path)
        messages = [
            {"role": "system", "content": "You are a helper."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        path = save_session(messages, "test_session")
        assert Path(path).exists()

        loaded = load_session("test_session")
        assert loaded is not None
        assert len(loaded) == 3
        assert loaded[0]["role"] == "system"
        assert loaded[2]["content"] == "Hi there!"

    def test_load_nonexistent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(session_mod, "SESSIONS_DIR", tmp_path)
        result = load_session("does_not_exist")
        assert result is None

    def test_list_sessions(self, tmp_path, monkeypatch):
        monkeypatch.setattr(session_mod, "SESSIONS_DIR", tmp_path)
        save_session([{"role": "user", "content": "a"}], "session1")
        save_session([{"role": "user", "content": "b"}], "session2")
        sessions = list_sessions()
        names = [s["name"] for s in sessions]
        assert "session1" in names
        assert "session2" in names

    def test_autosave_and_resume(self, tmp_path, monkeypatch):
        monkeypatch.setattr(session_mod, "SESSIONS_DIR", tmp_path)
        messages = [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "response"},
        ]
        auto_save_session(messages)
        restored = get_latest_session()
        assert restored is not None
        assert len(restored) == 2

    def test_autosave_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(session_mod, "SESSIONS_DIR", tmp_path)
        auto_save_session([])  # Should not create file
        result = get_latest_session()
        assert result is None

    def test_save_sanitizes_name(self, tmp_path, monkeypatch):
        monkeypatch.setattr(session_mod, "SESSIONS_DIR", tmp_path)
        path = save_session([{"role": "user", "content": "x"}], "test/../../evil")
        # Should sanitize the name
        assert Path(path).exists()
        # The filename should not contain path separators
        assert "/" not in Path(path).name or "\\" not in Path(path).name
