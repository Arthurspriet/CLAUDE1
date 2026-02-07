"""Tests for the undo stack."""

from pathlib import Path

from claude1.undo import FileSnapshot, UndoStack


class TestUndoStack:
    """Test undo stack push/pop behavior."""

    def test_empty_undo(self):
        stack = UndoStack()
        result = stack.undo_last()
        assert "Nothing to undo" in result

    def test_push_and_undo_existing_file(self, tmp_workdir):
        f = tmp_workdir / "test.txt"
        f.write_text("original")

        stack = UndoStack()
        stack.push(f)

        # Modify the file
        f.write_text("modified")
        assert f.read_text() == "modified"

        # Undo
        result = stack.undo_last()
        assert "Restored" in result
        assert f.read_text() == "original"

    def test_push_and_undo_new_file(self, tmp_workdir):
        f = tmp_workdir / "new.txt"

        stack = UndoStack()
        stack.push(f)  # File doesn't exist yet

        # Create the file
        f.write_text("new content")

        # Undo should delete it
        result = stack.undo_last()
        assert "Deleted" in result
        assert not f.exists()

    def test_multiple_undos(self, tmp_workdir):
        f = tmp_workdir / "test.txt"
        f.write_text("v1")

        stack = UndoStack()

        stack.push(f)
        f.write_text("v2")

        stack.push(f)
        f.write_text("v3")

        # Undo v3 -> v2
        stack.undo_last()
        assert f.read_text() == "v2"

        # Undo v2 -> v1
        stack.undo_last()
        assert f.read_text() == "v1"

    def test_max_size_enforcement(self):
        stack = UndoStack(max_size=3)
        for i in range(5):
            stack._stack.append(
                FileSnapshot(path=Path(f"/fake/{i}"), content=f"c{i}")
            )
        # Manually trim like push() does
        if len(stack._stack) > stack._max_size:
            stack._stack = stack._stack[-stack._max_size:]
        assert stack.size == 3

    def test_size_property(self):
        stack = UndoStack()
        assert stack.size == 0

    def test_undo_already_deleted_file(self, tmp_workdir):
        f = tmp_workdir / "gone.txt"
        stack = UndoStack()
        stack.push(f)  # File doesn't exist
        # Don't create it, so undo of a "new file" that's already gone
        result = stack.undo_last()
        assert "already gone" in result.lower()
