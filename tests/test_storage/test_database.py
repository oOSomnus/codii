"""Tests for the Database class."""

import pytest
from pathlib import Path

from codii.storage.database import Database


class TestDatabaseCreation:
    """Tests for database initialization."""

    def test_database_creation(self, temp_storage_dir):
        """Verify database and tables created correctly."""
        db_path = temp_storage_dir / "test.db"
        db = Database(db_path)

        assert db_path.exists()

        # Check that tables were created
        cursor = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row["name"] for row in cursor.fetchall()}

        assert "chunks" in tables
        assert "chunks_fts" in tables
        assert "files" in tables

        db.close()

    def test_database_creates_parent_directories(self, temp_storage_dir):
        """Verify parent directories are created if they don't exist."""
        db_path = temp_storage_dir / "nested" / "deep" / "test.db"
        db = Database(db_path)

        assert db_path.exists()
        assert db_path.parent.exists()

        db.close()


class TestInsertChunk:
    """Tests for inserting chunks."""

    def test_insert_chunk(self, temp_db_path):
        """Insert single chunk and verify."""
        db = Database(temp_db_path)

        chunk_id = db.insert_chunk(
            content="def hello():\n    print('hello')",
            path="/test/file.py",
            start_line=1,
            end_line=2,
            language="python",
            chunk_type="function",
        )

        assert chunk_id == 1

        # Verify chunk was inserted
        row = db.get_chunk_by_id(chunk_id)
        assert row is not None
        assert row["content"] == "def hello():\n    print('hello')"
        assert row["path"] == "/test/file.py"
        assert row["start_line"] == 1
        assert row["end_line"] == 2
        assert row["language"] == "python"
        assert row["chunk_type"] == "function"

        db.close()

    def test_insert_chunks_batch(self, temp_db_path):
        """Batch insert multiple chunks."""
        db = Database(temp_db_path)

        chunks = [
            ("content1", "/test/file1.py", 1, 5, "python", "function"),
            ("content2", "/test/file1.py", 10, 15, "python", "class"),
            ("content3", "/test/file2.py", 1, 3, "javascript", "function"),
        ]

        db.insert_chunks_batch(chunks)

        assert db.get_chunk_count() == 3

        db.close()


class TestBM25Search:
    """Tests for BM25 search functionality."""

    def test_search_bm25_returns_results(self, temp_db_path):
        """BM25 search returns matching chunks."""
        db = Database(temp_db_path)

        # Insert test chunks
        db.insert_chunk(
            content="def calculate_sum(a, b):\n    return a + b",
            path="/test/math.py",
            language="python",
            chunk_type="function",
        )
        db.insert_chunk(
            content="def greet(name):\n    print(f'Hello, {name}')",
            path="/test/greeting.py",
            language="python",
            chunk_type="function",
        )
        db.insert_chunk(
            content="class Calculator:\n    pass",
            path="/test/calculator.py",
            language="python",
            chunk_type="class",
        )

        # Search for "calculate"
        results = db.search_bm25("calculate")
        assert len(results) >= 1
        assert any("calculate" in r["content"].lower() for r in results)

        db.close()

    def test_search_bm25_no_results(self, temp_db_path):
        """BM25 search with no matches."""
        db = Database(temp_db_path)

        db.insert_chunk(
            content="def hello():\n    pass",
            path="/test/file.py",
            language="python",
            chunk_type="function",
        )

        results = db.search_bm25("xyznonexistent123")
        assert len(results) == 0

        db.close()

    def test_search_bm25_with_path_filter(self, temp_db_path):
        """Search with path filter."""
        db = Database(temp_db_path)

        db.insert_chunk(
            content="def calculate_sum():\n    pass",
            path="/project/module_a/math.py",
            language="python",
            chunk_type="function",
        )
        db.insert_chunk(
            content="def calculate_product():\n    pass",
            path="/project/module_b/math.py",
            language="python",
            chunk_type="function",
        )

        results = db.search_bm25("calculate", path_filter="module_a")

        # Should only get results from module_a
        assert all("module_a" in r["path"] for r in results)

        db.close()


class TestDeleteChunks:
    """Tests for deleting chunks."""

    def test_delete_chunks_by_path(self, temp_db_path):
        """Delete all chunks for a file."""
        db = Database(temp_db_path)

        db.insert_chunk(
            content="chunk1",
            path="/test/file.py",
            language="python",
            chunk_type="function",
        )
        db.insert_chunk(
            content="chunk2",
            path="/test/file.py",
            language="python",
            chunk_type="class",
        )
        db.insert_chunk(
            content="chunk3",
            path="/test/other.py",
            language="python",
            chunk_type="function",
        )

        deleted_count = db.delete_chunks_by_path("/test/file.py")

        assert deleted_count == 2
        assert db.get_chunk_count() == 1

        db.close()


class TestFileHash:
    """Tests for file hash storage."""

    def test_file_hash_storage(self, temp_db_path):
        """Store and retrieve file hashes."""
        db = Database(temp_db_path)

        # Store hash
        db.upsert_file_hash("/test/file.py", "abc123hash")

        # Retrieve hash
        stored_hash = db.get_file_hash("/test/file.py")
        assert stored_hash == "abc123hash"

        # Non-existent file
        assert db.get_file_hash("/test/nonexistent.py") is None

        db.close()

    def test_file_hash_update(self, temp_db_path):
        """Update existing file hash."""
        db = Database(temp_db_path)

        db.upsert_file_hash("/test/file.py", "old_hash")
        db.upsert_file_hash("/test/file.py", "new_hash")

        assert db.get_file_hash("/test/file.py") == "new_hash"

        db.close()

    def test_get_all_file_hashes(self, temp_db_path):
        """Get all file hashes."""
        db = Database(temp_db_path)

        db.upsert_file_hash("/test/file1.py", "hash1")
        db.upsert_file_hash("/test/file2.py", "hash2")

        all_hashes = db.get_all_file_hashes()

        assert all_hashes == {
            "/test/file1.py": "hash1",
            "/test/file2.py": "hash2",
        }

        db.close()


class TestFTS5Triggers:
    """Tests for FTS5 synchronization triggers."""

    def test_fts5_triggers_on_insert(self, temp_db_path):
        """Verify FTS5 sync on insert."""
        db = Database(temp_db_path)

        # Insert chunk
        chunk_id = db.insert_chunk(
            content="def my_function():\n    pass",
            path="/test/file.py",
            language="python",
            chunk_type="function",
        )

        # Check FTS5 table
        cursor = db.conn.execute(
            "SELECT * FROM chunks_fts WHERE chunks_fts MATCH 'my_function'"
        )
        results = cursor.fetchall()

        assert len(results) >= 1

        db.close()

    def test_fts5_triggers_on_delete(self, temp_db_path):
        """Verify FTS5 sync on delete."""
        db = Database(temp_db_path)

        db.insert_chunk(
            content="unique_function_name_xyz",
            path="/test/file.py",
            language="python",
            chunk_type="function",
        )

        # Delete the chunk
        db.delete_chunks_by_path("/test/file.py")

        # Check FTS5 table
        cursor = db.conn.execute(
            "SELECT * FROM chunks_fts WHERE chunks_fts MATCH 'unique_function_name_xyz'"
        )
        results = cursor.fetchall()

        assert len(results) == 0

        db.close()


class TestChunkCount:
    """Tests for chunk counting."""

    def test_get_chunk_count_empty(self, temp_db_path):
        """Get count of empty database."""
        db = Database(temp_db_path)

        assert db.get_chunk_count() == 0

        db.close()

    def test_get_chunk_count(self, temp_db_path):
        """Get count of chunks."""
        db = Database(temp_db_path)

        db.insert_chunk("content1", "/test/file.py", language="python", chunk_type="function")
        db.insert_chunk("content2", "/test/file.py", language="python", chunk_type="class")

        assert db.get_chunk_count() == 2

        db.close()


class TestGetChunkById:
    """Tests for retrieving chunks by ID."""

    def test_get_chunk_by_id(self, temp_db_path):
        """Get a chunk by its ID."""
        db = Database(temp_db_path)

        chunk_id = db.insert_chunk(
            content="test content",
            path="/test/file.py",
            start_line=10,
            end_line=20,
            language="python",
            chunk_type="function",
        )

        result = db.get_chunk_by_id(chunk_id)

        assert result is not None
        assert result["id"] == chunk_id
        assert result["content"] == "test content"

        db.close()

    def test_get_chunk_by_id_not_found(self, temp_db_path):
        """Get non-existent chunk."""
        db = Database(temp_db_path)

        result = db.get_chunk_by_id(999)
        assert result is None

        db.close()


class TestGetAllChunkIds:
    """Tests for getting all chunk IDs."""

    def test_get_all_chunk_ids(self, temp_db_path):
        """Get all chunk IDs."""
        db = Database(temp_db_path)

        id1 = db.insert_chunk("content1", "/test/file.py", language="python", chunk_type="function")
        id2 = db.insert_chunk("content2", "/test/file.py", language="python", chunk_type="class")

        all_ids = db.get_all_chunk_ids()

        assert set(all_ids) == {id1, id2}

        db.close()

    def test_get_all_chunk_ids_empty(self, temp_db_path):
        """Get all chunk IDs from empty database."""
        db = Database(temp_db_path)

        all_ids = db.get_all_chunk_ids()
        assert all_ids == []

        db.close()


class TestClearAllChunks:
    """Tests for clearing all chunks."""

    def test_clear_all_chunks(self, temp_db_path):
        """Clear all chunks from database."""
        db = Database(temp_db_path)

        db.insert_chunk("content1", "/test/file.py", language="python", chunk_type="function")
        db.insert_chunk("content2", "/test/file.py", language="python", chunk_type="class")

        deleted = db.clear_all_chunks()

        assert deleted == 2
        assert db.get_chunk_count() == 0

        db.close()


class TestThreadSafety:
    """Tests for thread safety."""

    def test_thread_local_connections(self, temp_db_path):
        """Verify thread-local connections work."""
        import threading

        db = Database(temp_db_path)
        results = []
        errors = []

        def insert_chunk(content, path):
            try:
                chunk_id = db.insert_chunk(
                    content=content,
                    path=path,
                    language="python",
                    chunk_type="function",
                )
                results.append(chunk_id)
            except Exception as e:
                errors.append(str(e))

        threads = [
            threading.Thread(target=insert_chunk, args=(f"content{i}", f"/test/file{i}.py"))
            for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 5

        db.close()