"""Tests for the GetIndexingStatusTool class."""

import pytest
from pathlib import Path

from codii.tools.status import GetIndexingStatusTool
from codii.utils.config import CodiiConfig, set_config
from codii.storage.snapshot import SnapshotManager


@pytest.fixture
def status_tool(mock_config):
    """Create a GetIndexingStatusTool for testing."""
    return GetIndexingStatusTool()


class TestStatusNotFound:
    """Tests for non-indexed codebase status."""

    def test_status_not_found(self, status_tool, temp_dir):
        """Codebase not indexed."""
        result = status_tool.run(path=str(temp_dir))

        assert result["isError"] is False
        assert "not indexed" in result["content"][0]["text"].lower()


class TestStatusIndexing:
    """Tests for indexing in progress status."""

    def test_status_indexing(self, status_tool, temp_dir, mock_config):
        """Shows progress and stage."""
        path_str = str(temp_dir)

        snapshot = SnapshotManager(mock_config.snapshot_file)
        snapshot.mark_indexing(path_str)
        snapshot.update_progress(path_str, 50, "embedding", 5, 25)

        result = status_tool.run(path=path_str)

        assert result["isError"] is False
        text = result["content"][0]["text"]

        assert "indexing" in text.lower()
        assert "50%" in text
        assert "embedding" in text.lower()


class TestStatusIndexed:
    """Tests for fully indexed status."""

    def test_status_indexed(self, status_tool, temp_dir, mock_config):
        """Shows completion stats."""
        path_str = str(temp_dir)

        snapshot = SnapshotManager(mock_config.snapshot_file)
        snapshot.mark_indexed(
            path_str,
            merkle_root="abc123",
            indexed_files=15,
            total_chunks=75,
        )

        result = status_tool.run(path=path_str)

        assert result["isError"] is False
        text = result["content"][0]["text"]

        assert "indexed" in text.lower()
        assert "15" in text  # indexed files
        assert "75" in text  # total chunks


class TestStatusFailed:
    """Tests for failed indexing status."""

    def test_status_failed(self, status_tool, temp_dir, mock_config):
        """Shows error message."""
        path_str = str(temp_dir)

        snapshot = SnapshotManager(mock_config.snapshot_file)
        snapshot.mark_failed(path_str, "Something went terribly wrong", progress=30)

        result = status_tool.run(path=path_str)

        assert result["isError"] is False
        text = result["content"][0]["text"]

        assert "failed" in text.lower()
        assert "Something went terribly wrong" in text
        assert "30%" in text


class TestInputSchema:
    """Tests for input schema."""

    def test_get_input_schema(self, status_tool):
        """Input schema is correct."""
        schema = status_tool.get_input_schema()

        assert schema["type"] == "object"
        assert "path" in schema["properties"]
        assert "path" in schema["required"]


class TestStatusProgressStages:
    """Tests for different progress stages."""

    def test_status_preparing_stage(self, status_tool, temp_dir, mock_config):
        """Shows preparing stage."""
        path_str = str(temp_dir)

        snapshot = SnapshotManager(mock_config.snapshot_file)
        snapshot.update_progress(path_str, 5, "preparing", 0, 0)

        result = status_tool.run(path=path_str)

        assert result["isError"] is False
        assert "preparing" in result["content"][0]["text"].lower()

    def test_status_chunking_stage(self, status_tool, temp_dir, mock_config):
        """Shows chunking stage."""
        path_str = str(temp_dir)

        snapshot = SnapshotManager(mock_config.snapshot_file)
        snapshot.update_progress(path_str, 20, "chunking", 3, 15)

        result = status_tool.run(path=path_str)

        assert result["isError"] is False
        assert "chunking" in result["content"][0]["text"].lower()

    def test_status_embedding_stage(self, status_tool, temp_dir, mock_config):
        """Shows embedding stage."""
        path_str = str(temp_dir)

        snapshot = SnapshotManager(mock_config.snapshot_file)
        snapshot.update_progress(path_str, 60, "embedding", 10, 50)

        result = status_tool.run(path=path_str)

        assert result["isError"] is False
        assert "embedding" in result["content"][0]["text"].lower()

    def test_status_indexing_stage(self, status_tool, temp_dir, mock_config):
        """Shows indexing stage."""
        path_str = str(temp_dir)

        snapshot = SnapshotManager(mock_config.snapshot_file)
        snapshot.update_progress(path_str, 90, "indexing", 10, 50)

        result = status_tool.run(path=path_str)

        assert result["isError"] is False
        # "indexing" appears in multiple contexts
        text = result["content"][0]["text"].lower()
        assert "indexing" in text


class TestStatusMultipleCodebases:
    """Tests for multiple codebases."""

    def test_status_independent_codebases(self, status_tool, temp_dir, mock_config):
        """Each codebase has independent status."""
        snapshot = SnapshotManager(mock_config.snapshot_file)

        path1 = str(temp_dir / "one")
        path2 = str(temp_dir / "two")

        snapshot.mark_indexed(path1, "hash1", 10, 50)
        snapshot.mark_indexing(path2)

        result1 = status_tool.run(path=path1)
        result2 = status_tool.run(path=path2)

        assert "indexed" in result1["content"][0]["text"].lower()
        assert "indexing" in result2["content"][0]["text"].lower()


class TestStatusPathHandling:
    """Tests for path handling."""

    def test_status_resolves_path(self, status_tool, temp_dir, mock_config):
        """Path is resolved to absolute."""
        # Mark with absolute path
        path_str = str(temp_dir.resolve())
        snapshot = SnapshotManager(mock_config.snapshot_file)
        snapshot.mark_indexed(path_str, "hash", 5, 20)

        # Query with relative or non-normalized path
        result = status_tool.run(path=str(temp_dir))

        # Should find the codebase
        assert result["isError"] is False
        assert "indexed" in result["content"][0]["text"].lower()