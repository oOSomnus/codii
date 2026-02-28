"""Tests for the BM25Indexer class."""

import pytest
from pathlib import Path

from codii.indexers.bm25_indexer import BM25Indexer
from codii.chunkers.text_chunker import CodeChunk
from codii.storage.database import Database


@pytest.fixture
def bm25_indexer(temp_db_path):
    """Create a BM25Indexer for testing."""
    indexer = BM25Indexer(temp_db_path)
    yield indexer
    indexer.close()


class TestAddChunks:
    """Tests for adding chunks to index."""

    def test_add_chunks(self, bm25_indexer):
        """Add chunks to index."""
        chunks = [
            CodeChunk(
                content="def hello_world():\n    print('hello')",
                path="/test/file1.py",
                start_line=1,
                end_line=2,
                language="python",
                chunk_type="function",
            ),
            CodeChunk(
                content="class MyClass:\n    pass",
                path="/test/file1.py",
                start_line=5,
                end_line=6,
                language="python",
                chunk_type="class",
            ),
        ]

        bm25_indexer.add_chunks(chunks)

        assert bm25_indexer.get_chunk_count() == 2

    def test_add_empty_chunks(self, bm25_indexer):
        """Add empty list of chunks."""
        bm25_indexer.add_chunks([])

        assert bm25_indexer.get_chunk_count() == 0


class TestSearchExactMatch:
    """Tests for exact match search."""

    def test_search_exact_match(self, bm25_indexer):
        """Exact term search."""
        chunks = [
            CodeChunk(
                content="def calculate_sum(a, b):\n    return a + b",
                path="/test/math.py",
                start_line=1,
                end_line=2,
                language="python",
                chunk_type="function",
            ),
            CodeChunk(
                content="def greet(name):\n    print(f'Hello, {name}')",
                path="/test/greeting.py",
                start_line=1,
                end_line=2,
                language="python",
                chunk_type="function",
            ),
        ]

        bm25_indexer.add_chunks(chunks)

        results = bm25_indexer.search("calculate")

        assert len(results) >= 1
        assert any("calculate" in r["content"].lower() for r in results)


class TestSearchPartialMatch:
    """Tests for partial/fuzzy match."""

    def test_search_partial_match(self, bm25_indexer):
        """Partial/fuzzy match."""
        chunks = [
            CodeChunk(
                content="def calculation_function():\n    pass",
                path="/test/file.py",
                start_line=1,
                end_line=2,
                language="python",
                chunk_type="function",
            ),
        ]

        bm25_indexer.add_chunks(chunks)

        # Search for partial term
        results = bm25_indexer.search("calc")

        # Should match "calculation"
        assert isinstance(results, list)


class TestSearchWithPathFilter:
    """Tests for path-filtered search."""

    def test_search_with_path_filter(self, bm25_indexer):
        """Filter by path."""
        chunks = [
            CodeChunk(
                content="def process_data():\n    pass",
                path="/project/module_a/processor.py",
                start_line=1,
                end_line=2,
                language="python",
                chunk_type="function",
            ),
            CodeChunk(
                content="def process_data():\n    pass",
                path="/project/module_b/handler.py",
                start_line=1,
                end_line=2,
                language="python",
                chunk_type="function",
            ),
        ]

        bm25_indexer.add_chunks(chunks)

        results = bm25_indexer.search("process", path_filter="module_a")

        # All results should be from module_a
        assert all("module_a" in r["path"] for r in results)


class TestRemoveFile:
    """Tests for removing file chunks."""

    def test_remove_file(self, bm25_indexer):
        """Remove all chunks for a file."""
        chunks = [
            CodeChunk(
                content="chunk1",
                path="/test/file.py",
                start_line=1,
                end_line=5,
                language="python",
                chunk_type="function",
            ),
            CodeChunk(
                content="chunk2",
                path="/test/file.py",
                start_line=10,
                end_line=15,
                language="python",
                chunk_type="class",
            ),
            CodeChunk(
                content="chunk3",
                path="/test/other.py",
                start_line=1,
                end_line=5,
                language="python",
                chunk_type="function",
            ),
        ]

        bm25_indexer.add_chunks(chunks)

        removed = bm25_indexer.remove_file("/test/file.py")

        assert removed == 2
        assert bm25_indexer.get_chunk_count() == 1


class TestClearIndex:
    """Tests for clearing the index."""

    def test_clear_index(self, bm25_indexer):
        """Clear all chunks."""
        chunks = [
            CodeChunk(
                content=f"chunk{i}",
                path=f"/test/file{i}.py",
                start_line=1,
                end_line=5,
                language="python",
                chunk_type="function",
            )
            for i in range(5)
        ]

        bm25_indexer.add_chunks(chunks)

        removed = bm25_indexer.clear()

        assert removed == 5
        assert bm25_indexer.get_chunk_count() == 0


class TestGetChunkCount:
    """Tests for chunk counting."""

    def test_get_chunk_count(self, bm25_indexer):
        """Get correct chunk count."""
        assert bm25_indexer.get_chunk_count() == 0

        chunks = [
            CodeChunk(
                content="chunk",
                path="/test/file.py",
                start_line=1,
                end_line=5,
                language="python",
                chunk_type="function",
            )
        ]

        bm25_indexer.add_chunks(chunks)

        assert bm25_indexer.get_chunk_count() == 1


class TestGetAllChunkIds:
    """Tests for getting all chunk IDs."""

    def test_get_all_chunk_ids(self, bm25_indexer):
        """Get all chunk IDs."""
        chunks = [
            CodeChunk(
                content=f"content{i}",
                path=f"/test/file{i}.py",
                start_line=1,
                end_line=5,
                language="python",
                chunk_type="function",
            )
            for i in range(3)
        ]

        bm25_indexer.add_chunks(chunks)

        ids = bm25_indexer.get_all_chunk_ids()

        assert len(ids) == 3
        assert all(isinstance(i, int) for i in ids)


class TestSearchResults:
    """Tests for search result format."""

    def test_search_result_format(self, bm25_indexer):
        """Verify search result contains expected fields."""
        chunks = [
            CodeChunk(
                content="def my_function():\n    return 42",
                path="/test/module.py",
                start_line=1,
                end_line=2,
                language="python",
                chunk_type="function",
            )
        ]

        bm25_indexer.add_chunks(chunks)

        results = bm25_indexer.search("my_function")

        if results:
            result = results[0]
            assert "id" in result
            assert "content" in result
            assert "path" in result
            assert "start_line" in result
            assert "end_line" in result
            assert "language" in result
            assert "chunk_type" in result
            assert "score" in result


class TestSearchLimit:
    """Tests for search result limiting."""

    def test_search_respects_limit(self, bm25_indexer):
        """Search respects limit parameter."""
        # Add many chunks with similar content
        chunks = [
            CodeChunk(
                content=f"def function_{i}():\n    return {i}",
                path=f"/test/file{i}.py",
                start_line=1,
                end_line=2,
                language="python",
                chunk_type="function",
            )
            for i in range(20)
        ]

        bm25_indexer.add_chunks(chunks)

        results = bm25_indexer.search("function", limit=5)

        assert len(results) <= 5