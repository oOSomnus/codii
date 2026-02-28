"""Tests for the ClearIndexTool class."""

import pytest
from pathlib import Path

from codii.tools.clear_index import ClearIndexTool
from codii.utils.config import CodiiConfig, set_config
from codii.storage.snapshot import SnapshotManager


@pytest.fixture
def clear_tool(mock_config):
    """Create a ClearIndexTool for testing."""
    return ClearIndexTool()


@pytest.fixture
def indexed_codebase(temp_dir, mock_config):
    """Create an indexed codebase for clearing."""
    path_str = str(temp_dir)
    path_hash = SnapshotManager.path_to_hash(path_str)

    # Create index directory and files
    index_dir = mock_config.indexes_dir / path_hash
    index_dir.mkdir(parents=True, exist_ok=True)

    (index_dir / "chunks.db").touch()
    (index_dir / "vectors.bin").touch()
    (index_dir / "vectors.meta.json").touch()

    # Create merkle file
    merkle_dir = mock_config.merkle_dir
    merkle_dir.mkdir(parents=True, exist_ok=True)
    (merkle_dir / f"{path_hash}.json").touch()

    # Mark as indexed
    snapshot = SnapshotManager(mock_config.snapshot_file)
    snapshot.mark_indexed(path_str, "test_hash", 10, 50)

    return temp_dir


class TestClearNoCodebases:
    """Tests for clearing when no codebases exist."""

    def test_clear_no_codebases(self, clear_tool, temp_storage_dir):
        """Message when nothing indexed."""
        # Use fresh storage
        snapshot = SnapshotManager(temp_storage_dir / "snapshots" / "snapshot.json")
        tool = ClearIndexTool()
        tool.snapshot_manager = snapshot

        result = tool.run(path="/any/path")

        assert result["isError"] is False
        assert "No codebases" in result["content"][0]["text"]


class TestClearNotFound:
    """Tests for clearing non-existent codebase."""

    def test_clear_not_found(self, clear_tool, mock_config):
        """Error for unknown path."""
        # First, add a codebase so that the tool gets past the "no codebases" check
        snapshot = SnapshotManager(mock_config.snapshot_file)
        snapshot.mark_indexed("/some/other/path", "hash", 5, 10)

        result = clear_tool.run(path="/nonexistent/path/xyz")

        assert result["isError"] is True
        assert "not found" in result["content"][0]["text"].lower()


class TestClearCurrentlyIndexing:
    """Tests for clearing while indexing."""

    def test_clear_currently_indexing(self, clear_tool, temp_dir, mock_config):
        """Error when indexing."""
        path_str = str(temp_dir)

        # Mark as indexing
        snapshot = SnapshotManager(mock_config.snapshot_file)
        snapshot.mark_indexing(path_str)

        result = clear_tool.run(path=path_str)

        assert result["isError"] is True
        assert "currently being indexed" in result["content"][0]["text"]


class TestClearDeletesFiles:
    """Tests for file deletion."""

    def test_clear_deletes_files(self, clear_tool, indexed_codebase, mock_config):
        """Removes database and vectors."""
        path_str = str(indexed_codebase)
        path_hash = SnapshotManager.path_to_hash(path_str)

        # Verify files exist before
        index_dir = mock_config.indexes_dir / path_hash
        assert (index_dir / "chunks.db").exists()

        result = clear_tool.run(path=path_str)

        assert result["isError"] is False

        # Verify files deleted
        assert not (index_dir / "chunks.db").exists()
        assert not (index_dir / "vectors.bin").exists()

    def test_clear_deletes_merkle(self, clear_tool, indexed_codebase, mock_config):
        """Removes merkle tree file."""
        path_str = str(indexed_codebase)
        path_hash = SnapshotManager.path_to_hash(path_str)

        merkle_path = mock_config.merkle_dir / f"{path_hash}.json"

        # Verify file exists before
        assert merkle_path.exists()

        result = clear_tool.run(path=path_str)

        assert result["isError"] is False
        assert not merkle_path.exists()


class TestClearUpdatesSnapshot:
    """Tests for snapshot updates."""

    def test_clear_updates_snapshot(self, clear_tool, indexed_codebase, mock_config):
        """Removes from snapshot."""
        path_str = str(indexed_codebase)

        result = clear_tool.run(path=path_str)

        assert result["isError"] is False

        # Check snapshot
        snapshot = SnapshotManager(mock_config.snapshot_file)
        status = snapshot.get_status(path_str)

        assert status.status == "not_found"


class TestInputSchema:
    """Tests for input schema."""

    def test_get_input_schema(self, clear_tool):
        """Input schema is correct."""
        schema = clear_tool.get_input_schema()

        assert schema["type"] == "object"
        assert "path" in schema["properties"]
        assert "path" in schema["required"]


class TestClearOneOfMultiple:
    """Tests for clearing one of multiple codebases."""

    def test_clear_one_keeps_others(self, mock_config):
        """Clearing one codebase keeps others."""
        # Create two indexed codebases
        path1 = "/test/path1"
        path2 = "/test/path2"
        hash1 = SnapshotManager.path_to_hash(path1)
        hash2 = SnapshotManager.path_to_hash(path2)

        # Create index files for both
        index_dir1 = mock_config.indexes_dir / hash1
        index_dir1.mkdir(parents=True, exist_ok=True)
        (index_dir1 / "chunks.db").touch()

        index_dir2 = mock_config.indexes_dir / hash2
        index_dir2.mkdir(parents=True, exist_ok=True)
        (index_dir2 / "chunks.db").touch()

        snapshot = SnapshotManager(mock_config.snapshot_file)
        snapshot.mark_indexed(path1, "hash1", 5, 20)
        snapshot.mark_indexed(path2, "hash2", 10, 40)

        # Clear first
        tool = ClearIndexTool()
        result = tool.run(path=path1)

        assert result["isError"] is False

        # Second should still exist
        status2 = snapshot.get_status(path2)
        assert status2.status == "indexed"

        # First should be gone
        status1 = snapshot.get_status(path1)
        assert status1.status == "not_found"


class TestClearResultMessage:
    """Tests for result messages."""

    def test_clear_success_message(self, clear_tool, indexed_codebase):
        """Success message includes path."""
        path_str = str(indexed_codebase)

        result = clear_tool.run(path=path_str)

        assert result["isError"] is False
        assert "cleared" in result["content"][0]["text"].lower()