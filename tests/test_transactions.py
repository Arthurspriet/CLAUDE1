"""Tests for multi-file transactions (P2-5)."""

import pytest

from claude1.undo import FileTransaction


@pytest.fixture
def workdir(tmp_path):
    """Create a temp working directory with sample files."""
    d = tmp_path / "project"
    d.mkdir()
    (d / "a.txt").write_text("original A")
    (d / "b.txt").write_text("original B")
    return d


class TestFileTransaction:
    def test_commit_preserves_changes(self, workdir):
        a = workdir / "a.txt"
        b = workdir / "b.txt"

        with FileTransaction() as txn:
            txn.track(a)
            a.write_text("modified A")
            txn.track(b)
            b.write_text("modified B")
            txn.commit()

        assert a.read_text() == "modified A"
        assert b.read_text() == "modified B"

    def test_rollback_on_exception(self, workdir):
        a = workdir / "a.txt"
        b = workdir / "b.txt"

        with pytest.raises(ValueError):
            with FileTransaction() as txn:
                txn.track(a)
                a.write_text("modified A")
                txn.track(b)
                b.write_text("modified B")
                raise ValueError("something went wrong")

        # Both files should be restored
        assert a.read_text() == "original A"
        assert b.read_text() == "original B"

    def test_manual_rollback(self, workdir):
        a = workdir / "a.txt"
        b = workdir / "b.txt"

        txn = FileTransaction()
        txn.track(a)
        a.write_text("modified A")
        txn.track(b)
        b.write_text("modified B")

        messages = txn.rollback()
        assert len(messages) == 2
        assert a.read_text() == "original A"
        assert b.read_text() == "original B"

    def test_rollback_deletes_new_files(self, workdir):
        new_file = workdir / "new.txt"
        assert not new_file.exists()

        with pytest.raises(RuntimeError):
            with FileTransaction() as txn:
                txn.track(new_file)
                new_file.write_text("new content")
                raise RuntimeError("abort")

        assert not new_file.exists()

    def test_tracked_files_property(self, workdir):
        a = workdir / "a.txt"
        b = workdir / "b.txt"

        txn = FileTransaction()
        txn.track(a)
        txn.track(b)

        paths = txn.tracked_files
        assert len(paths) == 2
        assert a.resolve() in paths
        assert b.resolve() in paths

    def test_committed_transaction_does_not_rollback(self, workdir):
        a = workdir / "a.txt"

        txn = FileTransaction()
        txn.track(a)
        a.write_text("modified A")
        txn.commit()

        # Even if we call __exit__ with an exception, committed won't roll back
        txn.__exit__(ValueError, None, None)
        assert a.read_text() == "modified A"

    def test_empty_transaction(self):
        with FileTransaction() as txn:
            txn.commit()
        # Should not raise

    def test_nested_directories(self, tmp_path):
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)
        f = deep / "file.txt"
        f.write_text("deep content")

        with pytest.raises(RuntimeError):
            with FileTransaction() as txn:
                txn.track(f)
                f.write_text("modified")
                raise RuntimeError("abort")

        assert f.read_text() == "deep content"
