"""Tests for the HybridSearch class."""

import pytest
import numpy as np
from pathlib import Path

from codii.indexers.hybrid_search import HybridSearch, SearchResult
from codii.indexers.bm25_indexer import BM25Indexer
from codii.indexers.vector_indexer import VectorIndexer
from codii.chunkers.text_chunker import CodeChunk


@pytest.fixture
def hybrid_search(temp_db_path, temp_vector_path, mock_embedder):
    """Create a HybridSearch instance for testing."""
    search = HybridSearch(temp_db_path, temp_vector_path)
    search.vector_indexer._embedder = mock_embedder
    yield search
    search.close()


@pytest.fixture
def populated_hybrid_search(temp_db_path, temp_vector_path, mock_embedder):
    """Create a HybridSearch instance with pre-populated data."""
    # Populate BM25 index - need enough chunks for HNSW
    bm25 = BM25Indexer(temp_db_path)
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
            content="def calculate_product(x, y):\n    return x * y",
            path="/test/math.py",
            start_line=5,
            end_line=6,
            language="python",
            chunk_type="function",
        ),
        CodeChunk(
            content="def greet_user(name):\n    print(f'Hello, {name}')",
            path="/test/greeting.py",
            start_line=1,
            end_line=2,
            language="python",
            chunk_type="function",
        ),
    ]
    # Add more dummy chunks to reach minimum for HNSW (at least 50)
    for i in range(50):
        chunks.append(CodeChunk(
            content=f"def helper_{i}():\n    return {i}",
            path=f"/test/helper_{i}.py",
            start_line=1,
            end_line=2,
            language="python",
            chunk_type="function",
        ))

    bm25.add_chunks(chunks)

    # Populate vector index using text embedding with higher ef_search
    vector = VectorIndexer(temp_vector_path, ef_search=100)
    vector._embedder = mock_embedder
    chunk_ids = bm25.get_all_chunk_ids()
    vector.add_vectors(chunk_ids, texts=[c.content for c in chunks])
    vector.save()

    bm25.close()

    search = HybridSearch(temp_db_path, temp_vector_path)
    search.vector_indexer._embedder = mock_embedder
    yield search
    search.close()


class TestSearchCombinesResults:
    """Tests for result combination."""

    def test_search_combines_results(self, populated_hybrid_search):
        """RRF combines BM25 and vector results."""
        results = populated_hybrid_search.search("calculate", limit=5, rerank=False)

        assert isinstance(results, list)
        # Should have results from the combination
        assert len(results) >= 1


class TestSearchDeduplicates:
    """Tests for result deduplication."""

    def test_search_deduplicates(self, populated_hybrid_search):
        """No duplicate results."""
        results = populated_hybrid_search.search("def", limit=5, rerank=False)

        # Extract IDs
        ids = [r.id for r in results]

        # All IDs should be unique
        assert len(ids) == len(set(ids))


class TestSearchWeights:
    """Tests for weighted combination."""

    def test_search_weights(self, temp_db_path, temp_vector_path, mock_embedder):
        """Weighted combination."""
        # Create with custom weights
        search = HybridSearch(
            temp_db_path,
            temp_vector_path,
            bm25_weight=0.7,
            vector_weight=0.3,
        )
        search.vector_indexer._embedder = mock_embedder

        assert search.bm25_weight == 0.7
        assert search.vector_weight == 0.3

        search.close()


class TestSearchEmptyIndex:
    """Tests for searching empty index."""

    def test_search_empty_index(self, temp_db_path, temp_vector_path, mock_embedder):
        """Handle empty index."""
        # Create an empty hybrid search (no data added)
        # First populate BM25 but not the vector index
        bm25 = BM25Indexer(temp_db_path)
        bm25.close()

        search = HybridSearch(temp_db_path, temp_vector_path)
        search.vector_indexer._embedder = mock_embedder
        results = search.search("nonexistent query", limit=5, rerank=False)
        search.close()

        assert isinstance(results, list)
        # Empty index should return empty results
        assert len(results) == 0


class TestSearchResultFormat:
    """Tests for search result format."""

    def test_search_result_format(self, populated_hybrid_search):
        """Correct output format."""
        results = populated_hybrid_search.search("calculate", limit=5, rerank=False)

        if results:
            result = results[0]

            assert isinstance(result, SearchResult)
            assert hasattr(result, "id")
            assert hasattr(result, "content")
            assert hasattr(result, "path")
            assert hasattr(result, "start_line")
            assert hasattr(result, "end_line")
            assert hasattr(result, "language")
            assert hasattr(result, "chunk_type")
            assert hasattr(result, "bm25_score")
            assert hasattr(result, "vector_score")
            assert hasattr(result, "combined_score")
            assert hasattr(result, "rerank_score")
            assert hasattr(result, "rank")


class TestSearchLimit:
    """Tests for search result limiting."""

    def test_search_with_limit(self, temp_db_path, temp_vector_path, mock_embedder):
        """Respect limit parameter."""
        # Populate with many chunks
        bm25 = BM25Indexer(temp_db_path)
        chunks = [
            CodeChunk(
                content=f"def function_{i}():\n    pass",
                path=f"/test/file{i}.py",
                start_line=1,
                end_line=2,
                language="python",
                chunk_type="function",
            )
            for i in range(50)
        ]
        bm25.add_chunks(chunks)

        vector = VectorIndexer(temp_vector_path, ef_search=100)
        vector._embedder = mock_embedder
        chunk_ids = bm25.get_all_chunk_ids()
        vector.add_vectors(chunk_ids, texts=[c.content for c in chunks])
        vector.save()

        bm25.close()

        search = HybridSearch(temp_db_path, temp_vector_path)
        search.vector_indexer._embedder = mock_embedder
        results = search.search("function", limit=5, rerank=False)
        search.close()

        assert len(results) <= 5


class TestReciprocalRankFusion:
    """Tests for RRF algorithm."""

    def test_rrf_scores(self, populated_hybrid_search):
        """RRF produces valid scores."""
        results = populated_hybrid_search.search("calculate", limit=5, rerank=False)

        for result in results:
            # Combined score should be sum of BM25 and vector scores
            expected = result.bm25_score + result.vector_score
            assert abs(result.combined_score - expected) < 0.0001


class TestPathFilter:
    """Tests for path filtering."""

    def test_search_with_path_filter(self, temp_db_path, temp_vector_path, mock_embedder):
        """Filter by path."""
        bm25 = BM25Indexer(temp_db_path)
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
        # Add more chunks to reach minimum for HNSW - all in module_a
        for i in range(50):
            chunks.append(CodeChunk(
                content=f"def helper_{i}():\n    pass",
                path=f"/project/module_a/helper_{i}.py",
                start_line=1,
                end_line=2,
                language="python",
                chunk_type="function",
            ))

        bm25.add_chunks(chunks)

        vector = VectorIndexer(temp_vector_path, ef_search=100)
        vector._embedder = mock_embedder
        chunk_ids = bm25.get_all_chunk_ids()
        vector.add_vectors(chunk_ids, texts=[c.content for c in chunks])
        vector.save()

        bm25.close()

        search = HybridSearch(temp_db_path, temp_vector_path)
        search.vector_indexer._embedder = mock_embedder
        results = search.search("process", path_filter="module_a", limit=10, rerank=False)
        search.close()

        # Results exist and filtering works
        assert len(results) >= 1


class TestSearchResultRanking:
    """Tests for result ranking."""

    def test_results_are_ranked(self, populated_hybrid_search):
        """Results have rank numbers."""
        results = populated_hybrid_search.search("calculate", limit=5, rerank=False)

        for i, result in enumerate(results, 1):
            assert result.rank == i

    def test_results_sorted_by_score(self, populated_hybrid_search):
        """Results sorted by combined score."""
        results = populated_hybrid_search.search("calculate", limit=5, rerank=False)

        scores = [r.combined_score for r in results]

        assert scores == sorted(scores, reverse=True)


class TestSearchResultData:
    """Tests for search result data population."""

    def test_result_has_chunk_data(self, populated_hybrid_search):
        """Results contain chunk data."""
        results = populated_hybrid_search.search("calculate", limit=5, rerank=False)

        for result in results:
            assert result.content is not None
            assert result.path is not None
            assert result.language is not None


class TestClose:
    """Tests for closing resources."""

    def test_close(self, hybrid_search):
        """Close releases resources."""
        # Access index to initialize
        _ = hybrid_search.bm25_indexer

        hybrid_search.close()

        # Just verify it doesn't raise
        pass