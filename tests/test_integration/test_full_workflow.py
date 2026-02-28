"""Integration tests for full workflow."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

from codii.utils.config import CodiiConfig, set_config
from codii.storage.snapshot import SnapshotManager
from codii.storage.database import Database
from codii.indexers.bm25_indexer import BM25Indexer
from codii.indexers.vector_indexer import VectorIndexer
from codii.indexers.hybrid_search import HybridSearch
from codii.chunkers.ast_chunker import ASTChunker
from codii.merkle.tree import MerkleTree


@pytest.fixture
def mock_embedder():
    """Create a mock embedder that returns deterministic vectors."""
    mock = MagicMock()
    mock._embedding_dim = 384
    mock._initialized = True

    def mock_embed(texts):
        if not texts:
            return np.array([])
        np.random.seed(42)
        return np.random.rand(len(texts), 384).astype(np.float32)

    def mock_embed_single(text):
        np.random.seed(42)
        return np.random.rand(384).astype(np.float32)

    mock.embed = mock_embed
    mock.embed_single = mock_embed_single
    mock.embedding_dim = 384

    return mock


class TestFullWorkflow:
    """End-to-end workflow tests."""

    def test_full_indexing_workflow(self, temp_dir, mock_config, mock_embedder):
        """Complete indexing and search workflow."""
        with patch('codii.indexers.vector_indexer.get_embedder', return_value=mock_embedder):
            # Create sample codebase
            (temp_dir / "main.py").write_text('''
def main():
    """Main entry point."""
    print("Hello, World!")
    helper()

def helper():
    """Helper function."""
    return 42

if __name__ == "__main__":
    main()
''')

            (temp_dir / "utils.py").write_text('''
def format_string(s: str) -> str:
    """Format a string."""
    return s.strip().lower()

def calculate_sum(a: int, b: int) -> int:
    """Calculate sum of two numbers."""
    return a + b
''')

            path_str = str(temp_dir)
            path_hash = SnapshotManager.path_to_hash(path_str)

            # Step 1: Initialize snapshot
            snapshot = SnapshotManager(mock_config.snapshot_file)
            snapshot.mark_indexing(path_str)

            # Step 2: Create index directory
            index_dir = mock_config.indexes_dir / path_hash
            index_dir.mkdir(parents=True, exist_ok=True)

            db_path = index_dir / "chunks.db"
            vector_path = index_dir / "vectors"

            # Step 3: Chunk files
            chunker = ASTChunker()
            all_chunks = []

            for py_file in temp_dir.glob("*.py"):
                content = py_file.read_text()
                chunks = chunker.chunk_file(content, str(py_file), "python")
                all_chunks.extend(chunks)

            # Add more chunks to reach HNSW minimum
            for i in range(10):
                from codii.chunkers.text_chunker import CodeChunk
                all_chunks.append(CodeChunk(
                    content=f"def dummy_{i}(): pass",
                    path=str(temp_dir / f"dummy_{i}.py"),
                    start_line=1,
                    end_line=1,
                    language="python",
                    chunk_type="function",
                ))

            assert len(all_chunks) >= 2  # Should have at least 2 chunks

            # Step 4: Add to BM25 index
            bm25 = BM25Indexer(db_path)
            bm25.add_chunks(all_chunks)

            assert bm25.get_chunk_count() >= 2

            # Step 5: Add to vector index
            vector = VectorIndexer(vector_path)
            vector._embedder = mock_embedder
            chunk_ids = bm25.get_all_chunk_ids()

            # Embed and add vectors
            vector.add_vectors(chunk_ids, texts=[c.content for c in all_chunks])
            vector.save()

            assert vector.get_vector_count() >= 2

            # Step 6: Build Merkle tree
            merkle = MerkleTree()
            for py_file in temp_dir.glob("*.py"):
                from codii.utils.file_utils import compute_file_hash
                file_hash = compute_file_hash(py_file)
                merkle.add_file(str(py_file), file_hash)

            merkle_path = mock_config.merkle_dir / f"{path_hash}.json"
            merkle.compute_root()
            merkle.save(merkle_path)

            # Step 7: Mark as complete
            snapshot.mark_indexed(
                path_str,
                merkle.root_hash,
                indexed_files=2,
                total_chunks=len(all_chunks),
            )

            # Step 8: Verify status
            status = snapshot.get_status(path_str)
            assert status.status == "indexed"

            # Step 9: Search
            search = HybridSearch(db_path, vector_path)
            search.vector_indexer._embedder = mock_embedder
            results = search.search("calculate", limit=5)
            search.close()

            assert len(results) >= 1

    def test_incremental_update(self, temp_dir, mock_config):
        """Detect and handle file changes."""
        # Initial indexing
        (temp_dir / "original.py").write_text('''
def original_function():
    return 1
''')

        path_str = str(temp_dir)
        path_hash = SnapshotManager.path_to_hash(path_str)

        # Build initial Merkle tree
        merkle1 = MerkleTree()
        from codii.utils.file_utils import compute_file_hash

        for py_file in temp_dir.glob("*.py"):
            file_hash = compute_file_hash(py_file)
            merkle1.add_file(str(py_file), file_hash)

        merkle1.compute_root()
        original_root = merkle1.root_hash

        # Save merkle tree
        merkle_path = mock_config.merkle_dir / f"{path_hash}.json"
        merkle1.save(merkle_path)

        # Modify file
        (temp_dir / "original.py").write_text('''
def modified_function():
    return 2

def new_function():
    return 3
''')

        # Add new file
        (temp_dir / "new_file.py").write_text('''
def additional_function():
    return 4
''')

        # Build new Merkle tree
        merkle2 = MerkleTree()
        for py_file in temp_dir.glob("*.py"):
            file_hash = compute_file_hash(py_file)
            merkle2.add_file(str(py_file), file_hash)

        merkle2.compute_root()

        # Load old tree and diff
        old_merkle = MerkleTree.load(merkle_path)
        added, removed, modified = merkle2.diff(old_merkle)

        assert str(temp_dir / "new_file.py") in added
        assert str(temp_dir / "original.py") in modified


class TestMultiCodebase:
    """Tests for multiple codebases."""

    def test_multiple_codebases(self, temp_dir, mock_config, mock_embedder):
        """Index and search multiple independent codebases."""
        with patch('codii.indexers.vector_indexer.get_embedder', return_value=mock_embedder):
            # Create two codebases
            codebase1 = temp_dir / "project1"
            codebase2 = temp_dir / "project2"
            codebase1.mkdir()
            codebase2.mkdir()

            (codebase1 / "math.py").write_text('def add(a, b): return a + b')
            (codebase2 / "math.py").write_text('def multiply(a, b): return a * b')

            snapshot = SnapshotManager(mock_config.snapshot_file)

            # Index both
            for codebase in [codebase1, codebase2]:
                path_str = str(codebase)
                path_hash = SnapshotManager.path_to_hash(path_str)

                index_dir = mock_config.indexes_dir / path_hash
                index_dir.mkdir(parents=True, exist_ok=True)

                db_path = index_dir / "chunks.db"
                vector_path = index_dir / "vectors"

                chunker = ASTChunker()
                chunks = []
                for py_file in codebase.glob("*.py"):
                    content = py_file.read_text()
                    file_chunks = chunker.chunk_file(content, str(py_file), "python")
                    chunks.extend(file_chunks)

                # Add dummy chunks for HNSW
                from codii.chunkers.text_chunker import CodeChunk
                for i in range(10):
                    chunks.append(CodeChunk(
                        content=f"def dummy_{i}(): pass",
                        path=str(codebase / f"dummy_{i}.py"),
                        start_line=1,
                        end_line=1,
                        language="python",
                        chunk_type="function",
                    ))

                bm25 = BM25Indexer(db_path)
                bm25.add_chunks(chunks)

                vector = VectorIndexer(vector_path)
                vector._embedder = mock_embedder
                chunk_ids = bm25.get_all_chunk_ids()
                vector.add_vectors(chunk_ids, texts=[c.content for c in chunks])
                vector.save()

                snapshot.mark_indexed(path_str, "hash", 1, len(chunks))

            # Verify both are indexed
            status1 = snapshot.get_status(str(codebase1))
            status2 = snapshot.get_status(str(codebase2))

            assert status1.status == "indexed"
            assert status2.status == "indexed"

            # Search each independently
            hash1 = SnapshotManager.path_to_hash(str(codebase1))
            hash2 = SnapshotManager.path_to_hash(str(codebase2))

            db1 = mock_config.indexes_dir / hash1 / "chunks.db"
            db2 = mock_config.indexes_dir / hash2 / "chunks.db"

            search1 = BM25Indexer(db1)
            search2 = BM25Indexer(db2)

            results1 = search1.search("add")
            results2 = search2.search("multiply")

            assert len(results1) >= 1
            assert len(results2) >= 1

            search1.close()
            search2.close()

    def test_clear_one_preserves_others(self, temp_dir, mock_config):
        """Clearing one codebase doesn't affect others."""
        # Create two codebases
        codebase1 = temp_dir / "project1"
        codebase2 = temp_dir / "project2"
        codebase1.mkdir()
        codebase2.mkdir()

        (codebase1 / "file.py").write_text('def func1(): pass')
        (codebase2 / "file.py").write_text('def func2(): pass')

        snapshot = SnapshotManager(mock_config.snapshot_file)

        # Index both
        for codebase in [codebase1, codebase2]:
            path_str = str(codebase)
            path_hash = SnapshotManager.path_to_hash(path_str)

            index_dir = mock_config.indexes_dir / path_hash
            index_dir.mkdir(parents=True, exist_ok=True)

            db_path = index_dir / "chunks.db"
            (db_path).touch()  # Create placeholder

            snapshot.mark_indexed(path_str, "hash", 1, 1)

        # Clear first
        hash1 = SnapshotManager.path_to_hash(str(codebase1))
        index_dir1 = mock_config.indexes_dir / hash1
        (index_dir1 / "chunks.db").unlink()

        snapshot.remove_codebase(str(codebase1))

        # Verify second still exists
        status2 = snapshot.get_status(str(codebase2))
        assert status2.status == "indexed"

        # Verify first is gone
        status1 = snapshot.get_status(str(codebase1))
        assert status1.status == "not_found"


class TestErrorRecovery:
    """Tests for error handling and recovery."""

    def test_indexing_failure_recovery(self, temp_dir, mock_config):
        """Recover from indexing failure."""
        snapshot = SnapshotManager(mock_config.snapshot_file)
        path_str = str(temp_dir)

        # Simulate failed indexing
        snapshot.mark_indexing(path_str)
        snapshot.mark_failed(path_str, "Test error", progress=50)

        # Verify failure status
        status = snapshot.get_status(path_str)
        assert status.status == "failed"

        # Retry indexing
        snapshot.mark_indexing(path_str)
        snapshot.mark_indexed(path_str, "hash", 5, 20)

        # Verify success
        status = snapshot.get_status(path_str)
        assert status.status == "indexed"

    def test_empty_codebase_handling(self, temp_dir, mock_config):
        """Handle empty codebase."""
        snapshot = SnapshotManager(mock_config.snapshot_file)
        path_str = str(temp_dir)

        # Index empty directory
        snapshot.mark_indexing(path_str)

        # No files to index
        snapshot.mark_failed(path_str, "No files found to index")

        status = snapshot.get_status(path_str)
        assert status.status == "failed"


class TestSearchQuality:
    """Tests for search result quality."""

    def test_relevant_results_first(self, temp_dir, mock_config, mock_embedder):
        """Most relevant results appear first."""
        with patch('codii.indexers.vector_indexer.get_embedder', return_value=mock_embedder):
            # Create files with varying relevance
            (temp_dir / "highly_relevant.py").write_text('''
def calculate_sum(a, b):
    """Calculate the sum of two numbers."""
    return a + b

def calculate_difference(a, b):
    """Calculate the difference."""
    return a - b
''')

            (temp_dir / "less_relevant.py").write_text('''
def process_data():
    """Process some data."""
    pass
''')

            path_str = str(temp_dir)
            path_hash = SnapshotManager.path_to_hash(path_str)

            index_dir = mock_config.indexes_dir / path_hash
            index_dir.mkdir(parents=True, exist_ok=True)

            db_path = index_dir / "chunks.db"
            vector_path = index_dir / "vectors"

            # Index files
            chunker = ASTChunker()
            all_chunks = []

            for py_file in temp_dir.glob("*.py"):
                content = py_file.read_text()
                chunks = chunker.chunk_file(content, str(py_file), "python")
                all_chunks.extend(chunks)

            # Add dummy chunks for HNSW
            from codii.chunkers.text_chunker import CodeChunk
            for i in range(10):
                all_chunks.append(CodeChunk(
                    content=f"def dummy_{i}(): pass",
                    path=str(temp_dir / f"dummy_{i}.py"),
                    start_line=1,
                    end_line=1,
                    language="python",
                    chunk_type="function",
                ))

            bm25 = BM25Indexer(db_path)
            bm25.add_chunks(all_chunks)

            vector = VectorIndexer(vector_path)
            vector._embedder = mock_embedder
            chunk_ids = bm25.get_all_chunk_ids()
            vector.add_vectors(chunk_ids, texts=[c.content for c in all_chunks])
            vector.save()

            # Search
            search = HybridSearch(db_path, vector_path)
            search.vector_indexer._embedder = mock_embedder
            results = search.search("calculate", limit=5)
            search.close()

            # First result should be from highly_relevant.py
            if results:
                assert "calculate" in results[0].content.lower()


class TestPersistence:
    """Tests for data persistence."""

    def test_index_persists_across_sessions(self, temp_dir, mock_config, mock_embedder):
        """Index data persists between sessions."""
        with patch('codii.indexers.vector_indexer.get_embedder', return_value=mock_embedder):
            (temp_dir / "file.py").write_text('def test_function(): pass')

            path_str = str(temp_dir)
            path_hash = SnapshotManager.path_to_hash(path_str)

            index_dir = mock_config.indexes_dir / path_hash
            index_dir.mkdir(parents=True, exist_ok=True)

            db_path = index_dir / "chunks.db"
            vector_path = index_dir / "vectors"

            # Add dummy chunks for HNSW
            from codii.chunkers.text_chunker import CodeChunk

            # First session: index
            chunker = ASTChunker()
            chunks = chunker.chunk_file(
                (temp_dir / "file.py").read_text(),
                str(temp_dir / "file.py"),
                "python"
            )

            # Add dummy chunks
            for i in range(10):
                chunks.append(CodeChunk(
                    content=f"def dummy_{i}(): pass",
                    path=str(temp_dir / f"dummy_{i}.py"),
                    start_line=1,
                    end_line=1,
                    language="python",
                    chunk_type="function",
                ))

            bm25 = BM25Indexer(db_path)
            bm25.add_chunks(chunks)
            chunk_ids = bm25.get_all_chunk_ids()

            vector = VectorIndexer(vector_path)
            vector._embedder = mock_embedder
            vector.add_vectors(chunk_ids, texts=[c.content for c in chunks])
            vector.save()

            initial_count = bm25.get_chunk_count()
            bm25.close()

            # Second session: reload and verify
            bm25_new = BM25Indexer(db_path)
            assert bm25_new.get_chunk_count() == initial_count

            results = bm25_new.search("test_function")
            assert len(results) >= 1

            bm25_new.close()