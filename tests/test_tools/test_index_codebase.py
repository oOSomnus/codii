"""Tests for the IndexCodebaseTool class."""

import pytest
from pathlib import Path
import time
import threading

from codii.tools.index_codebase import IndexCodebaseTool
from codii.utils.config import CodiiConfig, set_config
from codii.storage.snapshot import SnapshotManager


@pytest.fixture
def index_tool(mock_config):
    """Create an IndexCodebaseTool for testing."""
    return IndexCodebaseTool()


@pytest.fixture
def sample_repo(temp_dir):
    """Create a sample repository for indexing."""
    # Create Python files
    (temp_dir / "main.py").write_text('''
def main():
    """Main entry point."""
    print("Hello, World!")

if __name__ == "__main__":
    main()
''')

    (temp_dir / "utils.py").write_text('''
def helper():
    """A helper function."""
    return 42
''')

    # Create subdirectory
    subdir = temp_dir / "subdir"
    subdir.mkdir()
    (subdir / "module.py").write_text('''
class Module:
    pass
''')

    return temp_dir


class TestIndexNonexistentPath:
    """Tests for non-existent paths."""

    def test_index_nonexistent_path(self, index_tool):
        """Error for invalid path."""
        result = index_tool.run(path="/nonexistent/path/xyz")

        assert result["isError"] is True
        assert "does not exist" in result["content"][0]["text"]


class TestIndexFileNotDirectory:
    """Tests for file paths instead of directories."""

    def test_index_file_not_directory(self, index_tool, sample_python_file):
        """Error for file path."""
        result = index_tool.run(path=str(sample_python_file))

        assert result["isError"] is True
        assert "not a directory" in result["content"][0]["text"]


class TestIndexAlreadyIndexing:
    """Tests for already indexing state."""

    def test_index_already_indexing(self, index_tool, sample_repo, mock_config):
        """Error when already indexing."""
        # Mark as indexing
        snapshot = SnapshotManager(mock_config.snapshot_file)
        snapshot.mark_indexing(str(sample_repo))

        result = index_tool.run(path=str(sample_repo))

        assert result["isError"] is True
        assert "currently being indexed" in result["content"][0]["text"]


class TestIndexAlreadyIndexed:
    """Tests for already indexed state."""

    def test_index_already_indexed(self, index_tool, sample_repo, mock_config):
        """Info message when indexed."""
        # Mark as indexed
        snapshot = SnapshotManager(mock_config.snapshot_file)
        snapshot.mark_indexed(
            str(sample_repo),
            merkle_root="hash",
            indexed_files=5,
            total_chunks=10,
        )

        result = index_tool.run(path=str(sample_repo))

        assert result["isError"] is False
        assert "already indexed" in result["content"][0]["text"]


class TestIndexForceReindex:
    """Tests for force re-indexing."""

    def test_index_force_reindex(self, index_tool, sample_repo, mock_config):
        """Force re-index existing."""
        # First, mark as indexed
        snapshot = SnapshotManager(mock_config.snapshot_file)
        snapshot.mark_indexed(
            str(sample_repo),
            merkle_root="hash",
            indexed_files=5,
            total_chunks=10,
        )

        # Force re-index
        result = index_tool.run(path=str(sample_repo), force=True)

        assert result["isError"] is False
        assert "Indexing started" in result["content"][0]["text"]


class TestIndexStartsBackground:
    """Tests for background indexing."""

    def test_index_starts_background(self, index_tool, sample_repo):
        """Indexing runs in background."""
        result = index_tool.run(path=str(sample_repo))

        assert result["isError"] is False
        assert "Indexing started" in result["content"][0]["text"]

        # Wait a bit for indexing to start
        time.sleep(0.5)

    def test_index_returns_immediately(self, index_tool, sample_repo):
        """Index tool returns immediately."""
        start = time.time()
        result = index_tool.run(path=str(sample_repo))
        elapsed = time.time() - start

        # Should return quickly, not wait for indexing to complete
        assert elapsed < 2.0


class TestIndexWithCustomExtensions:
    """Tests for custom file extensions."""

    def test_index_with_custom_extensions(self, index_tool, temp_dir, mock_config):
        """Custom file extensions."""
        # Create files with custom extension
        (temp_dir / "file.custom").write_text("custom content")
        (temp_dir / "file.py").write_text("def hello(): pass")

        result = index_tool.run(
            path=str(temp_dir),
            customExtensions=[".custom"],
        )

        assert result["isError"] is False


class TestIndexWithIgnorePatterns:
    """Tests for custom ignore patterns."""

    def test_index_with_ignore_patterns(self, index_tool, temp_dir, mock_config):
        """Custom ignore patterns."""
        # Create files
        (temp_dir / "main.py").write_text("def main(): pass")
        (temp_dir / "test_main.py").write_text("def test_main(): pass")

        result = index_tool.run(
            path=str(temp_dir),
            ignorePatterns=["test_*.py"],
        )

        assert result["isError"] is False


class TestInputSchema:
    """Tests for input schema."""

    def test_get_input_schema(self, index_tool):
        """Input schema is correct."""
        schema = index_tool.get_input_schema()

        assert schema["type"] == "object"
        assert "path" in schema["properties"]
        assert "path" in schema["required"]


class TestInvalidPath:
    """Tests for invalid path handling."""

    def test_index_relative_path(self, index_tool):
        """Handle relative path."""
        result = index_tool.run(path="some/relative/path")

        # Should resolve to absolute path and fail
        assert result["isError"] is True


class TestEmptyDirectory:
    """Tests for empty directories."""

    def test_index_empty_directory(self, index_tool, temp_dir):
        """Handle empty directory."""
        result = index_tool.run(path=str(temp_dir))

        # Should start indexing but may fail during processing
        assert result["isError"] is False or "No files" in result["content"][0]["text"]


class TestSplitterOptions:
    """Tests for splitter options."""

    def test_index_with_ast_splitter(self, index_tool, sample_repo):
        """Use AST splitter."""
        result = index_tool.run(path=str(sample_repo), splitter="ast")

        assert result["isError"] is False

    def test_index_with_text_splitter(self, index_tool, sample_repo):
        """Use text splitter."""
        result = index_tool.run(path=str(sample_repo), splitter="langchain")

        assert result["isError"] is False


class TestConcurrentIndexing:
    """Tests for concurrent indexing prevention."""

    def test_concurrent_indexing_same_path(self, index_tool, sample_repo, mock_config):
        """Prevent concurrent indexing of same path."""
        # Start first indexing
        result1 = index_tool.run(path=str(sample_repo))
        assert result1["isError"] is False

        # Try to start second indexing of same path
        result2 = index_tool.run(path=str(sample_repo))
        assert result2["isError"] is True
        assert "currently being indexed" in result2["content"][0]["text"]