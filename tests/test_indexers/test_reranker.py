"""Tests for cross-encoder re-ranking."""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from codii.embedding.cross_encoder import CrossEncoderWrapper, get_cross_encoder
from codii.indexers.hybrid_search import SearchResult


@pytest.fixture
def sample_results():
    """Create sample search results for testing."""
    return [
        SearchResult(
            id=1,
            content="def calculate_sum(a, b):\n    return a + b",
            path="/test/math.py",
            start_line=1,
            end_line=2,
            language="python",
            chunk_type="function",
            combined_score=0.8,
        ),
        SearchResult(
            id=2,
            content="def greet_user(name):\n    print(f'Hello, {name}')",
            path="/test/greeting.py",
            start_line=1,
            end_line=2,
            language="python",
            chunk_type="function",
            combined_score=0.6,
        ),
        SearchResult(
            id=3,
            content="class Calculator:\n    def add(self, x, y):\n        return x + y",
            path="/test/calculator.py",
            start_line=1,
            end_line=3,
            language="python",
            chunk_type="class",
            combined_score=0.4,
        ),
    ]


@pytest.fixture
def mock_cross_encoder():
    """Create a mock cross-encoder model."""
    mock = MagicMock()

    def mock_predict(pairs):
        # Return scores based on content relevance to query
        # Higher scores for content containing query words
        scores = []
        for query, content in pairs:
            if "calculate" in content.lower() or "add" in content.lower():
                scores.append(2.5)  # High logit -> ~0.92 after sigmoid
            elif "greet" in content.lower():
                scores.append(-1.0)  # Low logit -> ~0.27 after sigmoid
            else:
                scores.append(0.5)  # Medium logit -> ~0.62 after sigmoid
        return np.array(scores)

    mock.predict = mock_predict
    return mock


class TestCrossEncoderWrapper:
    """Tests for CrossEncoderWrapper class."""

    def test_singleton_pattern(self):
        """CrossEncoderWrapper uses singleton pattern."""
        instance1 = CrossEncoderWrapper()
        instance2 = CrossEncoderWrapper()

        assert instance1 is instance2

    def test_lazy_loading(self):
        """Model is loaded lazily."""
        # Reset singleton
        CrossEncoderWrapper._instance = None

        wrapper = CrossEncoderWrapper()
        assert wrapper._model is None

        # Don't actually load the model in tests
        # Just verify the lazy pattern exists
        CrossEncoderWrapper._instance = None

    def test_get_cross_encoder_function(self):
        """get_cross_encoder returns singleton instance."""
        CrossEncoderWrapper._instance = None

        instance1 = get_cross_encoder()
        instance2 = get_cross_encoder()

        assert instance1 is instance2

        CrossEncoderWrapper._instance = None


class TestRerank:
    """Tests for rerank method."""

    def test_rerank_empty_candidates(self):
        """Empty candidates returns empty results."""
        CrossEncoderWrapper._instance = None
        wrapper = CrossEncoderWrapper()
        results = wrapper.rerank("query", [], top_k=10)
        assert results == []
        CrossEncoderWrapper._instance = None

    def test_rerank_returns_correct_count(self, sample_results, mock_cross_encoder):
        """Rerank respects top_k parameter."""
        CrossEncoderWrapper._instance = None
        wrapper = CrossEncoderWrapper()
        wrapper._model = mock_cross_encoder

        results = wrapper.rerank("calculate sum", sample_results, top_k=2)

        assert len(results) <= 2

        CrossEncoderWrapper._instance = None

    def test_rerank_adds_scores(self, sample_results, mock_cross_encoder):
        """Rerank adds rerank_score to results."""
        CrossEncoderWrapper._instance = None
        wrapper = CrossEncoderWrapper()
        wrapper._model = mock_cross_encoder

        results = wrapper.rerank("calculate sum", sample_results, top_k=10)

        for result in results:
            assert hasattr(result, "rerank_score")
            assert 0 <= result.rerank_score <= 1

        CrossEncoderWrapper._instance = None

    def test_rerank_threshold_filtering(self, sample_results, mock_cross_encoder):
        """Rerank filters results below threshold."""
        CrossEncoderWrapper._instance = None
        wrapper = CrossEncoderWrapper()
        wrapper._model = mock_cross_encoder

        # Use high threshold to filter out some results
        results = wrapper.rerank("greeting", sample_results, top_k=10, threshold=0.7)

        # All returned results should be above threshold
        for result in results:
            assert result.rerank_score >= 0.7

        CrossEncoderWrapper._instance = None

    def test_rerank_sorts_by_score(self, sample_results, mock_cross_encoder):
        """Rerank sorts results by cross-encoder score."""
        CrossEncoderWrapper._instance = None
        wrapper = CrossEncoderWrapper()
        wrapper._model = mock_cross_encoder

        results = wrapper.rerank("calculate", sample_results, top_k=10, threshold=0.0)

        # Results should be sorted by rerank_score descending
        scores = [r.rerank_score for r in results]
        assert scores == sorted(scores, reverse=True)

        CrossEncoderWrapper._instance = None

    def test_rerank_normalized_scores(self, sample_results, mock_cross_encoder):
        """Scores are normalized to 0-1 range using sigmoid."""
        CrossEncoderWrapper._instance = None
        wrapper = CrossEncoderWrapper()
        wrapper._model = mock_cross_encoder

        results = wrapper.rerank("calculate", sample_results, top_k=10, threshold=0.0)

        for result in results:
            assert 0 <= result.rerank_score <= 1

        CrossEncoderWrapper._instance = None


class TestRerankIntegration:
    """Tests for re-ranking integration in HybridSearch."""

    @patch('codii.embedding.cross_encoder.get_cross_encoder')
    def test_hybrid_search_with_rerank(
        self, mock_get_cross_encoder, temp_db_path, temp_vector_path, mock_embedder
    ):
        """Hybrid search with re-ranking enabled."""
        from codii.indexers.hybrid_search import HybridSearch
        from codii.indexers.bm25_indexer import BM25Indexer
        from codii.indexers.vector_indexer import VectorIndexer
        from codii.chunkers.text_chunker import CodeChunk

        # Setup mock cross-encoder
        mock_ce = MagicMock()
        mock_ce.rerank.return_value = [
            SearchResult(
                id=1,
                content="def calculate_sum(a, b):\n    return a + b",
                path="/test/math.py",
                start_line=1,
                end_line=2,
                language="python",
                chunk_type="function",
                rerank_score=0.9,
            ),
        ]
        mock_get_cross_encoder.return_value = mock_ce

        # Populate index
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
        ]
        # Add more chunks for HNSW minimum
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

        vector = VectorIndexer(temp_vector_path, ef_search=100)
        vector._embedder = mock_embedder
        chunk_ids = bm25.get_all_chunk_ids()
        vector.add_vectors(chunk_ids, texts=[c.content for c in chunks])
        vector.save()
        bm25.close()

        # Search with re-ranking
        search = HybridSearch(temp_db_path, temp_vector_path)
        search.vector_indexer._embedder = mock_embedder
        results = search.search("calculate", limit=5, rerank=True)
        search.close()

        # Verify re-ranking was called
        mock_get_cross_encoder.assert_called()
        mock_ce.rerank.assert_called_once()

    @patch('codii.embedding.cross_encoder.get_cross_encoder')
    def test_hybrid_search_without_rerank(
        self, mock_get_cross_encoder, temp_db_path, temp_vector_path, mock_embedder
    ):
        """Hybrid search with re-ranking disabled."""
        from codii.indexers.hybrid_search import HybridSearch
        from codii.indexers.bm25_indexer import BM25Indexer
        from codii.indexers.vector_indexer import VectorIndexer
        from codii.chunkers.text_chunker import CodeChunk

        # Populate index
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
        ]
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

        vector = VectorIndexer(temp_vector_path, ef_search=100)
        vector._embedder = mock_embedder
        chunk_ids = bm25.get_all_chunk_ids()
        vector.add_vectors(chunk_ids, texts=[c.content for c in chunks])
        vector.save()
        bm25.close()

        # Search without re-ranking
        search = HybridSearch(temp_db_path, temp_vector_path)
        search.vector_indexer._embedder = mock_embedder
        results = search.search("calculate", limit=5, rerank=False)
        search.close()

        # Verify re-ranking was NOT called
        mock_get_cross_encoder.assert_not_called()

    @patch('codii.embedding.cross_encoder.get_cross_encoder')
    def test_rerank_fallback_on_error(
        self, mock_get_cross_encoder, temp_db_path, temp_vector_path, mock_embedder
    ):
        """Hybrid search falls back to RRF when re-ranking fails."""
        from codii.indexers.hybrid_search import HybridSearch
        from codii.indexers.bm25_indexer import BM25Indexer
        from codii.indexers.vector_indexer import VectorIndexer
        from codii.chunkers.text_chunker import CodeChunk

        # Setup mock to raise error
        mock_get_cross_encoder.side_effect = RuntimeError("Model loading failed")

        # Populate index
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
        ]
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

        vector = VectorIndexer(temp_vector_path, ef_search=100)
        vector._embedder = mock_embedder
        chunk_ids = bm25.get_all_chunk_ids()
        vector.add_vectors(chunk_ids, texts=[c.content for c in chunks])
        vector.save()
        bm25.close()

        # Search with re-ranking enabled (but will fail)
        search = HybridSearch(temp_db_path, temp_vector_path)
        search.vector_indexer._embedder = mock_embedder
        results = search.search("calculate", limit=5, rerank=True)
        search.close()

        # Should still get results from RRF fallback
        assert isinstance(results, list)
        # Results won't have rerank_score since fallback was used
        for result in results:
            assert result.rerank_score == 0.0


class TestSearchResultWithRerankScore:
    """Tests for SearchResult with rerank_score field."""

    def test_search_result_has_rerank_score(self):
        """SearchResult has rerank_score field."""
        result = SearchResult(
            id=1,
            content="test",
            path="/test.py",
            start_line=1,
            end_line=2,
            language="python",
            chunk_type="function",
        )

        assert hasattr(result, "rerank_score")
        assert result.rerank_score == 0.0

    def test_search_result_rerank_score_can_be_set(self):
        """SearchResult rerank_score can be set."""
        result = SearchResult(
            id=1,
            content="test",
            path="/test.py",
            start_line=1,
            end_line=2,
            language="python",
            chunk_type="function",
            rerank_score=0.85,
        )

        assert result.rerank_score == 0.85