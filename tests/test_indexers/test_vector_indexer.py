"""Tests for the VectorIndexer class."""

import pytest
import numpy as np
from pathlib import Path

from codii.indexers.vector_indexer import VectorIndexer


@pytest.fixture
def vector_indexer(temp_vector_path, mock_embedder):
    """Create a VectorIndexer for testing with mocked embedder."""
    indexer = VectorIndexer(temp_vector_path)
    indexer._embedder = mock_embedder
    yield indexer


class TestAddVectors:
    """Tests for adding vectors to index."""

    def test_add_vectors(self, vector_indexer):
        """Add vectors to HNSW index."""
        # Create sample vectors - need at least 5 for HNSW
        vectors = np.random.rand(15, 384).astype(np.float32)
        chunk_ids = list(range(1, 16))

        vector_indexer.add_vectors(chunk_ids, vectors=vectors)

        assert vector_indexer.get_vector_count() == 15

    def test_add_vectors_with_texts(self, vector_indexer):
        """Add vectors by embedding texts."""
        texts = [f"hello world {i}" for i in range(15)]
        chunk_ids = list(range(1, 16))

        vector_indexer.add_vectors(chunk_ids, texts=texts)

        assert vector_indexer.get_vector_count() == 15

    def test_add_empty_vectors(self, vector_indexer):
        """Add empty list of vectors."""
        vector_indexer.add_vectors([], vectors=np.array([]))

        assert vector_indexer.get_vector_count() == 0


class TestSearchSimilar:
    """Tests for searching similar vectors."""

    def test_search_similar(self, vector_indexer):
        """Find similar vectors."""
        # Add some vectors - need at least 10 for HNSW reliability
        vectors = np.random.rand(20, 384).astype(np.float32)
        chunk_ids = list(range(1, 21))

        vector_indexer.add_vectors(chunk_ids, vectors=vectors)

        # Search with one of the vectors
        query_vector = vectors[0]
        results = vector_indexer.search("test", query_vector=query_vector, k=5)

        assert len(results) >= 1
        # First result should be the query vector itself (most similar)
        assert results[0][0] == chunk_ids[0]

    def test_search_with_text_query(self, vector_indexer):
        """Search using text query."""
        texts = [f"def function_{i}():" for i in range(15)]
        chunk_ids = list(range(1, 16))

        vector_indexer.add_vectors(chunk_ids, texts=texts)

        results = vector_indexer.search("function", k=5)

        assert isinstance(results, list)

    def test_search_k_parameter(self, vector_indexer):
        """Search respects k parameter."""
        vectors = np.random.rand(25, 384).astype(np.float32)
        chunk_ids = list(range(1, 26))

        vector_indexer.add_vectors(chunk_ids, vectors=vectors)

        results = vector_indexer.search("test", k=5, query_vector=vectors[0])

        assert len(results) <= 5


class TestSaveLoad:
    """Tests for persisting and restoring index."""

    def test_save_creates_files(self, temp_vector_path, mock_embedder):
        """Save creates index files."""
        indexer = VectorIndexer(temp_vector_path, ef_search=100)
        indexer._embedder = mock_embedder
        vectors = np.random.rand(50, 384).astype(np.float32)
        chunk_ids = list(range(1, 51))

        indexer.add_vectors(chunk_ids, vectors=vectors)
        indexer.save()

        # Verify files exist
        assert temp_vector_path.with_suffix(".bin").exists()
        assert temp_vector_path.with_suffix(".meta.json").exists()
        assert indexer.get_vector_count() == 50


class TestClearIndex:
    """Tests for clearing the index."""

    def test_clear_index(self, vector_indexer):
        """Clear vector index."""
        vectors = np.random.rand(15, 384).astype(np.float32)
        chunk_ids = list(range(1, 16))

        vector_indexer.add_vectors(chunk_ids, vectors=vectors)

        vector_indexer.clear()

        assert vector_indexer.get_vector_count() == 0


class TestVectorCount:
    """Tests for vector counting."""

    def test_vector_count(self, vector_indexer):
        """Correct count of vectors."""
        assert vector_indexer.get_vector_count() == 0

        vectors = np.random.rand(15, 384).astype(np.float32)
        chunk_ids = list(range(1, 16))

        vector_indexer.add_vectors(chunk_ids, vectors=vectors)

        assert vector_indexer.get_vector_count() == 15


class TestRemoveByChunkId:
    """Tests for removing vectors by chunk ID."""

    def test_remove_by_chunk_id(self, vector_indexer):
        """Mark a vector as deleted."""
        vectors = np.random.rand(15, 384).astype(np.float32)
        chunk_ids = list(range(1, 16))

        vector_indexer.add_vectors(chunk_ids, vectors=vectors)

        result = vector_indexer.remove_by_chunk_id(2)

        assert result is True
        assert vector_indexer.get_vector_count() == 14

    def test_remove_nonexistent_chunk_id(self, vector_indexer):
        """Remove non-existent chunk ID."""
        result = vector_indexer.remove_by_chunk_id(999)

        assert result is False


class TestEmbedder:
    """Tests for embedder integration."""

    def test_embedder_set(self, vector_indexer):
        """Embedder is set from fixture."""
        assert vector_indexer._embedder is not None

    def test_embedding_dimension(self, vector_indexer):
        """Correct embedding dimension."""
        dim = vector_indexer._embedder.embedding_dim

        assert dim > 0
        assert isinstance(dim, int)


class TestHNSWParameters:
    """Tests for HNSW parameters."""

    def test_custom_hnsw_parameters(self, temp_vector_path, mock_embedder):
        """Custom HNSW parameters are used."""
        indexer = VectorIndexer(
            temp_vector_path,
            m=32,
            ef_construction=400,
            ef_search=100,
        )
        indexer._embedder = mock_embedder

        assert indexer.m == 32
        assert indexer.ef_construction == 400
        assert indexer.ef_search == 100


class TestIndexProperties:
    """Tests for index properties."""

    def test_index_lazy_initialization(self, vector_indexer):
        """Index is initialized lazily."""
        # Index should not exist yet
        assert vector_indexer._index is None

        # Access index property
        index = vector_indexer.index

        assert index is not None

    def test_index_creates_parent_directories(self, temp_storage_dir, mock_embedder):
        """Index creates parent directories."""
        vector_path = temp_storage_dir / "nested" / "deep" / "vectors"

        indexer = VectorIndexer(vector_path)
        indexer._embedder = mock_embedder

        # Accessing index should create directories
        _ = indexer.index

        assert vector_path.parent.exists()