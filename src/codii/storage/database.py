"""SQLite database management for codii."""

import sqlite3
from pathlib import Path
from typing import Optional
import threading


class Database:
    """Thread-safe SQLite database manager."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    @property
    def conn(self) -> sqlite3.Connection:
        """Get thread-local connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self.db_path))
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self.conn

        # Create chunks table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                path TEXT NOT NULL,
                start_line INTEGER,
                end_line INTEGER,
                language TEXT,
                chunk_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create FTS5 virtual table for BM25 search
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                content,
                path,
                language,
                content='chunks',
                content_rowid='id'
            )
        """)

        # Create triggers to keep FTS in sync
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, content, path, language)
                VALUES (new.id, new.content, new.path, new.language);
            END
        """)

        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, content, path, language)
                VALUES('delete', old.id, old.content, old.path, old.language);
            END
        """)

        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, content, path, language)
                VALUES('Delete', old.id, old.content, old.path, old.language);
                INSERT INTO chunks_fts(rowid, content, path, language)
                VALUES (new.id, new.content, new.path, new.language);
            END
        """)

        # Create files table for Merkle tree tracking
        conn.execute("""
            CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY,
                hash TEXT NOT NULL,
                last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create index on path for faster lookups
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path)
        """)

        conn.commit()

    def insert_chunk(
        self,
        content: str,
        path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        language: Optional[str] = None,
        chunk_type: Optional[str] = None,
    ) -> int:
        """Insert a chunk and return its ID."""
        cursor = self.conn.execute(
            """
            INSERT INTO chunks (content, path, start_line, end_line, language, chunk_type)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (content, path, start_line, end_line, language, chunk_type),
        )
        self.conn.commit()
        return cursor.lastrowid

    def insert_chunks_batch(self, chunks: list) -> None:
        """Insert multiple chunks in a batch."""
        self.conn.executemany(
            """
            INSERT INTO chunks (content, path, start_line, end_line, language, chunk_type)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            chunks,
        )
        self.conn.commit()

    def search_bm25(self, query: str, limit: int = 10, path_filter: Optional[str] = None) -> list:
        """Search using BM25 via FTS5."""
        # Escape special characters for FTS5
        escaped_query = query.replace("'", "''")

        if path_filter:
            sql = f"""
                SELECT
                    c.id, c.content, c.path, c.start_line, c.end_line, c.language, c.chunk_type,
                    bm25(chunks_fts) as score
                FROM chunks_fts
                JOIN chunks c ON chunks_fts.rowid = c.id
                WHERE chunks_fts MATCH ?
                AND c.path LIKE ?
                ORDER BY score
                LIMIT ?
            """
            cursor = self.conn.execute(sql, (escaped_query, f"%{path_filter}%", limit))
        else:
            sql = f"""
                SELECT
                    c.id, c.content, c.path, c.start_line, c.end_line, c.language, c.chunk_type,
                    bm25(chunks_fts) as score
                FROM chunks_fts
                JOIN chunks c ON chunks_fts.rowid = c.id
                WHERE chunks_fts MATCH ?
                ORDER BY score
                LIMIT ?
            """
            cursor = self.conn.execute(sql, (escaped_query, limit))

        return cursor.fetchall()

    def get_chunk_by_id(self, chunk_id: int) -> Optional[dict]:
        """Get a chunk by ID."""
        cursor = self.conn.execute(
            "SELECT * FROM chunks WHERE id = ?",
            (chunk_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_all_chunk_ids(self) -> list:
        """Get all chunk IDs."""
        cursor = self.conn.execute("SELECT id FROM chunks")
        return [row["id"] for row in cursor.fetchall()]

    def delete_chunks_by_path(self, path: str) -> int:
        """Delete all chunks for a given path."""
        cursor = self.conn.execute(
            "DELETE FROM chunks WHERE path = ?",
            (path,),
        )
        self.conn.commit()
        return cursor.rowcount

    def clear_all_chunks(self) -> int:
        """Delete all chunks."""
        cursor = self.conn.execute("DELETE FROM chunks")
        self.conn.commit()
        return cursor.rowcount

    def upsert_file_hash(self, path: str, file_hash: str) -> None:
        """Insert or update file hash."""
        self.conn.execute(
            """
            INSERT OR REPLACE INTO files (path, hash, last_modified)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            """,
            (path, file_hash),
        )
        self.conn.commit()

    def get_file_hash(self, path: str) -> Optional[str]:
        """Get stored hash for a file."""
        cursor = self.conn.execute(
            "SELECT hash FROM files WHERE path = ?",
            (path,),
        )
        row = cursor.fetchone()
        return row["hash"] if row else None

    def get_all_file_hashes(self) -> dict:
        """Get all file hashes."""
        cursor = self.conn.execute("SELECT path, hash FROM files")
        return {row["path"]: row["hash"] for row in cursor.fetchall()}

    def delete_file_hash(self, path: str) -> None:
        """Delete file hash entry."""
        self.conn.execute("DELETE FROM files WHERE path = ?", (path,))
        self.conn.commit()

    def clear_all_file_hashes(self) -> None:
        """Delete all file hashes."""
        self.conn.execute("DELETE FROM files")
        self.conn.commit()

    def get_chunk_count(self) -> int:
        """Get total number of chunks."""
        cursor = self.conn.execute("SELECT COUNT(*) as count FROM chunks")
        return cursor.fetchone()["count"]

    def get_file_count(self) -> int:
        """Get number of indexed files."""
        cursor = self.conn.execute("SELECT COUNT(*) as count FROM files")
        return cursor.fetchone()["count"]

    def close(self) -> None:
        """Close the connection."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None