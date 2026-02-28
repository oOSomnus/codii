"""Tests for the SnapshotManager class."""

import pytest
from pathlib import Path
import json

from codii.storage.snapshot import SnapshotManager, CodebaseStatus


class TestSnapshotCreation:
    """Tests for snapshot file creation."""

    def test_snapshot_creation(self, temp_snapshot_path):
        """Create snapshot file."""
        manager = SnapshotManager(temp_snapshot_path)

        assert temp_snapshot_path.exists()

        # Verify initial structure
        with open(temp_snapshot_path) as f:
            data = json.load(f)

        assert "codebases" in data
        assert data["codebases"] == {}

    def test_snapshot_creates_parent_directories(self, temp_storage_dir):
        """Verify parent directories are created."""
        snapshot_path = temp_storage_dir / "nested" / "snapshot.json"
        SnapshotManager(snapshot_path)

        assert snapshot_path.exists()
        assert snapshot_path.parent.exists()


class TestGetStatus:
    """Tests for getting codebase status."""

    def test_get_status_not_found(self, temp_snapshot_path):
        """Status for untracked codebase."""
        manager = SnapshotManager(temp_snapshot_path)

        status = manager.get_status("/nonexistent/path")

        assert status.path == "/nonexistent/path"
        assert status.status == "not_found"

    def test_get_status_existing(self, temp_snapshot_path):
        """Status for tracked codebase."""
        manager = SnapshotManager(temp_snapshot_path)

        # Set up a codebase
        manager.set_status(CodebaseStatus(
            path="/test/path",
            status="indexed",
            indexed_files=10,
            total_chunks=50,
        ))

        status = manager.get_status("/test/path")

        assert status.path == "/test/path"
        assert status.status == "indexed"
        assert status.indexed_files == 10
        assert status.total_chunks == 50


class TestMarkIndexing:
    """Tests for marking codebase as indexing."""

    def test_mark_indexing(self, temp_snapshot_path):
        """Mark codebase as indexing."""
        manager = SnapshotManager(temp_snapshot_path)

        manager.mark_indexing("/test/codebase")

        status = manager.get_status("/test/codebase")

        assert status.status == "indexing"
        assert status.progress == 0
        assert status.current_stage == "preparing"


class TestMarkIndexed:
    """Tests for marking codebase as indexed."""

    def test_mark_indexed(self, temp_snapshot_path):
        """Mark codebase as fully indexed."""
        manager = SnapshotManager(temp_snapshot_path)

        manager.mark_indexed(
            path="/test/codebase",
            merkle_root="abc123",
            indexed_files=15,
            total_chunks=75,
        )

        status = manager.get_status("/test/codebase")

        assert status.status == "indexed"
        assert status.progress == 100
        assert status.current_stage == "complete"
        assert status.merkle_root == "abc123"
        assert status.indexed_files == 15
        assert status.total_chunks == 75


class TestMarkFailed:
    """Tests for marking codebase as failed."""

    def test_mark_failed(self, temp_snapshot_path):
        """Mark codebase as failed."""
        manager = SnapshotManager(temp_snapshot_path)

        manager.mark_failed("/test/codebase", "Something went wrong", progress=50)

        status = manager.get_status("/test/codebase")

        assert status.status == "failed"
        assert status.progress == 50
        assert status.error_message == "Something went wrong"


class TestUpdateProgress:
    """Tests for updating indexing progress."""

    def test_update_progress(self, temp_snapshot_path):
        """Update progress percentage."""
        manager = SnapshotManager(temp_snapshot_path)

        # First, mark as indexing
        manager.mark_indexing("/test/codebase")

        # Update progress
        manager.update_progress(
            path="/test/codebase",
            progress=50,
            stage="embedding",
            indexed_files=5,
            total_chunks=25,
        )

        status = manager.get_status("/test/codebase")

        assert status.progress == 50
        assert status.current_stage == "embedding"
        assert status.indexed_files == 5
        assert status.total_chunks == 25

    def test_update_progress_creates_new_entry(self, temp_snapshot_path):
        """Update progress creates entry if not exists."""
        manager = SnapshotManager(temp_snapshot_path)

        # Update progress without marking indexing first
        manager.update_progress(
            path="/test/codebase",
            progress=25,
            stage="chunking",
            indexed_files=2,
            total_chunks=10,
        )

        status = manager.get_status("/test/codebase")

        assert status.status == "indexing"
        assert status.progress == 25


class TestRemoveCodebase:
    """Tests for removing codebase from tracking."""

    def test_remove_codebase(self, temp_snapshot_path):
        """Remove codebase from tracking."""
        manager = SnapshotManager(temp_snapshot_path)

        manager.mark_indexed(
            path="/test/codebase",
            merkle_root="hash",
            indexed_files=10,
            total_chunks=50,
        )

        result = manager.remove_codebase("/test/codebase")

        assert result is True
        status = manager.get_status("/test/codebase")
        assert status.status == "not_found"

    def test_remove_codebase_not_found(self, temp_snapshot_path):
        """Remove non-existent codebase."""
        manager = SnapshotManager(temp_snapshot_path)

        result = manager.remove_codebase("/nonexistent/path")

        assert result is False


class TestPathToHash:
    """Tests for path hashing."""

    def test_path_to_hash(self):
        """Consistent hash generation."""
        hash1 = SnapshotManager.path_to_hash("/test/path")
        hash2 = SnapshotManager.path_to_hash("/test/path")

        assert hash1 == hash2
        assert len(hash1) == 16  # First 16 chars of SHA-256

    def test_path_to_hash_different_paths(self):
        """Different paths produce different hashes."""
        hash1 = SnapshotManager.path_to_hash("/path/one")
        hash2 = SnapshotManager.path_to_hash("/path/two")

        assert hash1 != hash2


class TestIsIndexing:
    """Tests for checking indexing status."""

    def test_is_indexing_true(self, temp_snapshot_path):
        """Check if codebase is currently indexing."""
        manager = SnapshotManager(temp_snapshot_path)

        manager.mark_indexing("/test/codebase")

        assert manager.is_indexing("/test/codebase") is True

    def test_is_indexing_false(self, temp_snapshot_path):
        """Check if codebase is not indexing."""
        manager = SnapshotManager(temp_snapshot_path)

        manager.mark_indexed(
            path="/test/codebase",
            merkle_root="hash",
            indexed_files=10,
            total_chunks=50,
        )

        assert manager.is_indexing("/test/codebase") is False


class TestGetAllCodebases:
    """Tests for getting all codebases."""

    def test_get_all_codebases(self, temp_snapshot_path):
        """Get all tracked codebases."""
        manager = SnapshotManager(temp_snapshot_path)

        manager.mark_indexed("/path/one", "hash1", 10, 50)
        manager.mark_indexed("/path/two", "hash2", 20, 100)

        all_codebases = manager.get_all_codebases()

        assert len(all_codebases) == 2
        assert "/path/one" in all_codebases
        assert "/path/two" in all_codebases

    def test_get_all_codebases_empty(self, temp_snapshot_path):
        """Get all codebases when none tracked."""
        manager = SnapshotManager(temp_snapshot_path)

        all_codebases = manager.get_all_codebases()

        assert all_codebases == {}


class TestHasAnyCodebases:
    """Tests for checking if any codebases are tracked."""

    def test_has_any_codebases_true(self, temp_snapshot_path):
        """Check if any codebases are tracked."""
        manager = SnapshotManager(temp_snapshot_path)

        manager.mark_indexed("/test/codebase", "hash", 10, 50)

        assert manager.has_any_codebases() is True

    def test_has_any_codebases_false(self, temp_snapshot_path):
        """Check if no codebases are tracked."""
        manager = SnapshotManager(temp_snapshot_path)

        assert manager.has_any_codebases() is False


class TestCodebaseStatus:
    """Tests for CodebaseStatus dataclass."""

    def test_to_dict(self):
        """Convert status to dictionary."""
        status = CodebaseStatus(
            path="/test/path",
            status="indexed",
            progress=100,
            indexed_files=10,
            total_chunks=50,
        )

        data = status.to_dict()

        assert data["path"] == "/test/path"
        assert data["status"] == "indexed"
        assert data["progress"] == 100

    def test_from_dict(self):
        """Create status from dictionary."""
        data = {
            "path": "/test/path",
            "status": "indexed",
            "progress": 100,
            "current_stage": "complete",
            "merkle_root": "hash123",
            "indexed_files": 10,
            "total_chunks": 50,
            "last_updated": "2024-01-01T00:00:00",
            "error_message": None,
        }

        status = CodebaseStatus.from_dict(data)

        assert status.path == "/test/path"
        assert status.status == "indexed"
        assert status.merkle_root == "hash123"


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_updates(self, temp_snapshot_path):
        """Test concurrent status updates."""
        import threading

        manager = SnapshotManager(temp_snapshot_path)
        errors = []

        def update_status(path, progress):
            try:
                manager.update_progress(path, progress, "chunking", 1, 1)
            except Exception as e:
                errors.append(str(e))

        threads = [
            threading.Thread(target=update_status, args=(f"/path/{i}", i * 10))
            for i in range(10)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(manager.get_all_codebases()) == 10