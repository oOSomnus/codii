"""BM25 indexer using SQLite FTS5."""

from pathlib import Path
from typing import List, Optional
import sqlite3

from ..storage.database import Database
from ..chunkers.ast_chunker import CodeChunk


class BM25Indexer:
    """BM25 indexer using SQLite FTS5."""

    def __init__(self, db_path: Path):
        """
        Initialize the BM25 indexer.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db = Database(db_path)

    def add_chunks(self, chunks: List[CodeChunk]) -> None:
        """
        Add chunks to the index.

        Args:
            chunks: List of CodeChunk objects
        """
        chunk_tuples = [chunk.to_tuple() for chunk in chunks]
        self.db.insert_chunks_batch(chunk_tuples)

    def search(
        self,
        query: str,
        limit: int = 10,
        path_filter: Optional[str] = None,
    ) -> List[dict]:
        """
        Search for chunks using BM25.

        Args:
            query: Search query
            limit: Maximum number of results
            path_filter: Optional path filter (matches if path contains this string)

        Returns:
            List of result dictionaries with chunk info and scores
        """
        rows = self.db.search_bm25(query, limit, path_filter)
        results = []
        for row in rows:
            results.append({
                "id": row["id"],
                "content": row["content"],
                "path": row["path"],
                "start_line": row["start_line"],
                "end_line": row["end_line"],
                "language": row["language"],
                "chunk_type": row["chunk_type"],
                "score": row["score"],
            })
        return results

    def remove_file(self, path: str) -> int:
        """
        Remove all chunks for a file.

        Args:
            path: File path

        Returns:
            Number of chunks removed
        """
        return self.db.delete_chunks_by_path(path)

    def clear(self) -> int:
        """
        Clear all chunks from the index.

        Returns:
            Number of chunks removed
        """
        return self.db.clear_all_chunks()

    def get_chunk_count(self) -> int:
        """Get total number of indexed chunks."""
        return self.db.get_chunk_count()

    def get_all_chunk_ids(self) -> List[int]:
        """Get all chunk IDs."""
        return self.db.get_all_chunk_ids()

    def close(self) -> None:
        """Close the database connection."""
        self.db.close()