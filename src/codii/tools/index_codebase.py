"""Index codebase tool for codii."""

import asyncio
import threading
from pathlib import Path
from typing import Optional, List

from ..utils.config import get_config
from ..utils.file_utils import scan_directory, get_file_content, detect_language
from ..storage.snapshot import SnapshotManager
from ..storage.database import Database
from ..chunkers.ast_chunker import ASTChunker
from ..chunkers.text_chunker import TextChunker
from ..indexers.bm25_indexer import BM25Indexer
from ..indexers.vector_indexer import VectorIndexer
from ..merkle.tree import MerkleTree


class IndexCodebaseTool:
    """Tool to index a codebase with incremental update support.

    Behavior:
    - New codebase: Performs full index
    - Already indexed + no changes: Returns early with "no changes detected"
    - Already indexed + changes detected: Performs incremental update (only processes changed files)
    - force=true: Clears existing index and performs full re-index
    """

    def __init__(self):
        self.config = get_config()
        self.snapshot_manager = SnapshotManager(self.config.snapshot_file)
        self._indexing_threads: dict[str, threading.Thread] = {}

    def get_input_schema(self) -> dict:
        """Get the input schema for the tool."""
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the codebase to index",
                },
                "force": {
                    "type": "boolean",
                    "default": False,
                    "description": "Force full re-index by clearing existing index. Use only for recovery from corrupted indexes or when you want to reset completely. Normal indexing automatically detects changes and performs incremental updates.",
                },
                "splitter": {
                    "type": "string",
                    "enum": ["ast", "langchain"],
                    "default": "ast",
                    "description": "Code splitting method (ast or langchain/text)",
                },
                "customExtensions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                    "description": "Additional file extensions to index",
                },
                "ignorePatterns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                    "description": "Additional patterns to ignore",
                },
            },
            "required": ["path"],
        }

    def run(self, path: str, force: bool = False, splitter: str = "ast",
            customExtensions: Optional[List[str]] = None,
            ignorePatterns: Optional[List[str]] = None) -> dict:
        """
        Index a codebase with automatic incremental update support.

        Behavior:
        - New codebase: Performs full index
        - Already indexed + no changes: Returns early with "no changes detected"
        - Already indexed + changes detected: Performs incremental update
          (only processes added, modified, or removed files)
        - force=True: Clears existing index and performs full re-index

        Args:
            path: Absolute path to the codebase
            force: Force full re-index by clearing existing index. Use only for
                   recovery or reset. Normal calls auto-detect changes.
            splitter: Code splitting method ("ast" or "langchain")
            customExtensions: Additional file extensions to index
            ignorePatterns: Additional patterns to ignore

        Returns:
            Dict with result message or error
        """
        # Validate path
        repo_path = Path(path).resolve()
        if not repo_path.exists():
            return {
                "content": [{"type": "text", "text": f"Error: Path does not exist: {path}"}],
                "isError": True,
            }
        if not repo_path.is_dir():
            return {
                "content": [{"type": "text", "text": f"Error: Path is not a directory: {path}"}],
                "isError": True,
            }

        path_str = str(repo_path)

        # Check if already indexing
        if self.snapshot_manager.is_indexing(path_str):
            return {
                "content": [{"type": "text", "text": "Codebase is currently being indexed. Please wait."}],
                "isError": True,
            }

        # Check if force or needs re-indexing
        status = self.snapshot_manager.get_status(path_str)

        # For incremental indexing, we need to check if there are changes
        # even when already indexed (unless force=True which does full re-index)
        if status.status == "indexed" and not force:
            # Check if there are any file changes for incremental update
            path_hash = self.snapshot_manager.path_to_hash(path_str)
            merkle_path = self.config.merkle_dir / f"{path_hash}.json"
            old_merkle = MerkleTree.load(merkle_path)

            if old_merkle:
                # Quick check: scan files and compare with old merkle tree
                normalized_extensions = []
                for ext in (customExtensions or []):
                    if ext and not ext.startswith("."):
                        normalized_extensions.append("." + ext)
                    else:
                        normalized_extensions.append(ext)

                extensions = set(self.config.default_extensions)
                files = scan_directory(
                    repo_path,
                    extensions,
                    self.config.default_ignore_patterns,
                    normalized_extensions,
                    ignorePatterns or [],
                    use_gitignore=True,
                )

                # Build new Merkle tree for comparison
                new_merkle = MerkleTree()
                for file_path, file_hash in files:
                    new_merkle.add_file(str(file_path), file_hash)
                new_merkle.compute_root()

                # If no changes, return early
                if new_merkle.root_hash == old_merkle.root_hash:
                    return {
                        "content": [{
                            "type": "text",
                            "text": f"Codebase already indexed at {path}. No changes detected. Use force=true to re-index."
                        }],
                        "isError": False,
                    }
                # Otherwise, proceed with incremental indexing (don't return early)
            else:
                # No merkle tree, need to re-index
                return {
                    "content": [{
                        "type": "text",
                        "text": f"Codebase already indexed at {path}. Use force=true to re-index."
                    }],
                    "isError": False,
                }

        # Clear existing index if force (full re-index)
        if force and status.status == "indexed":
            self._clear_index(path_str)

        # Start indexing in background
        thread = threading.Thread(
            target=self._index_codebase,
            args=(path_str, splitter, customExtensions or [], ignorePatterns or [], force),
            daemon=True,
        )
        thread.start()
        self._indexing_threads[path_str] = thread

        return {
            "content": [{
                "type": "text",
                "text": f"Indexing started for {path}. Use get_indexing_status to check progress."
            }],
            "isError": False,
        }

    def _index_codebase(
        self,
        path: str,
        splitter: str,
        custom_extensions: List[str],
        ignore_patterns: List[str],
        force: bool = False,
    ) -> None:
        """Perform the actual indexing (runs in background thread).

        Supports incremental updates: only processes files that have been
        added, removed, or modified since the last index.
        """
        try:
            # Mark as indexing
            self.snapshot_manager.mark_indexing(path)

            repo_path = Path(path)
            path_hash = self.snapshot_manager.path_to_hash(path)

            # Prepare directories
            index_dir = self.config.indexes_dir / path_hash
            index_dir.mkdir(parents=True, exist_ok=True)

            db_path = index_dir / "chunks.db"
            vector_path = index_dir / "vectors"

            # Initialize components
            db = Database(db_path)
            bm25_indexer = BM25Indexer(db_path)
            vector_indexer = VectorIndexer(vector_path)

            # Prepare extensions (normalize to include .)
            normalized_extensions = []
            for ext in custom_extensions:
                if ext and not ext.startswith("."):
                    normalized_extensions.append("." + ext)
                else:
                    normalized_extensions.append(ext)

            # Stage 1: Preparing (0-10%)
            self.snapshot_manager.update_progress(path, 5, "preparing")

            extensions = set(self.config.default_extensions)
            files = scan_directory(
                repo_path,
                extensions,
                self.config.default_ignore_patterns,
                normalized_extensions,
                ignore_patterns,
                use_gitignore=True,
            )

            total_files = len(files)
            self.snapshot_manager.update_progress(
                path, 10, "preparing", 0, 0, total_files, 0
            )

            if not files:
                self.snapshot_manager.mark_failed(path, "No files found to index", 10)
                return

            # Load previous Merkle tree for incremental updates
            merkle_path = self.config.merkle_dir / f"{path_hash}.json"
            old_merkle = MerkleTree.load(merkle_path)

            # Build new Merkle tree
            new_merkle = MerkleTree()
            for file_path, file_hash in files:
                new_merkle.add_file(str(file_path), file_hash)

            new_merkle.compute_root()

            # Compute diff for incremental updates
            if old_merkle and not force:
                added, removed, modified = new_merkle.diff(old_merkle)
            else:
                # Force re-index or new codebase: treat all files as added
                added = set(str(f) for f, _ in files)
                removed = set()
                modified = set()

            files_to_delete = removed | modified
            files_to_add = added | modified
            files_to_process = len(files_to_add)

            # Early exit if no changes
            if not files_to_delete and not files_to_add:
                self.snapshot_manager.mark_indexed(
                    path,
                    new_merkle.root_hash,
                    len(files),
                    db.get_chunk_count(),
                )
                return

            # Stage 2: Deletion (10-20%) - Remove stale chunks and vectors
            if files_to_delete:
                self.snapshot_manager.update_progress(
                    path, 15, "deleting", 0, 0, total_files, files_to_process
                )

                for file_path in files_to_delete:
                    # Get chunk IDs before deleting from database
                    chunk_ids = db.get_chunk_ids_by_path(file_path)
                    # Remove vectors from HNSW index (soft delete)
                    if chunk_ids:
                        vector_indexer.remove_by_chunk_ids(chunk_ids)
                    # Delete chunks from SQLite (FTS cleanup via trigger)
                    db.delete_chunks_by_path(file_path)

                self.snapshot_manager.update_progress(
                    path, 20, "deleting", 0, 0, total_files, files_to_process
                )

            # Stage 3: Chunking (20-40%) - Only process files to add
            self.snapshot_manager.update_progress(
                path, 20, "chunking", 0, 0, total_files, files_to_process
            )

            chunker = ASTChunker() if splitter == "ast" else TextChunker(
                max_chunk_size=self.config.max_chunk_size,
                min_chunk_size=self.config.min_chunk_size,
            )

            all_chunks = []
            files_to_add_list = list(files_to_add)
            total_files_to_add = len(files_to_add_list)

            for i, file_path_str in enumerate(files_to_add_list):
                file_path = Path(file_path_str)
                content = get_file_content(file_path)
                if content is None:
                    continue

                language = detect_language(file_path)
                chunks = chunker.chunk_file(
                    content,
                    str(file_path),
                    language,
                    self.config.max_chunk_size,
                    self.config.min_chunk_size,
                )
                all_chunks.extend(chunks)

                # Update progress
                if total_files_to_add > 0:
                    progress = 20 + int((i / total_files_to_add) * 20)
                    self.snapshot_manager.update_progress(
                        path, progress, "chunking", i + 1, len(all_chunks), total_files, files_to_process
                    )

            # Stage 4: Embedding (40-80%)
            self.snapshot_manager.update_progress(
                path, 40, "embedding", len(files_to_add_list), len(all_chunks), total_files, files_to_process
            )

            if all_chunks:
                # Batch embed
                texts = [chunk.content for chunk in all_chunks]
                batch_size = self.config.embedding_batch_size
                total_batches = (len(texts) + batch_size - 1) // batch_size

                all_vectors = []
                for batch_idx in range(total_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, len(texts))
                    batch_texts = texts[start_idx:end_idx]

                    vectors = vector_indexer.embedder.embed(batch_texts)
                    all_vectors.append(vectors)

                    # Update progress
                    progress = 40 + int((batch_idx / total_batches) * 40)
                    self.snapshot_manager.update_progress(
                        path, progress, "embedding", len(files_to_add_list), len(all_chunks), total_files, files_to_process
                    )

                # Combine all vectors
                import numpy as np
                all_vectors = np.vstack(all_vectors)

                # Stage 5: Indexing (80-100%)
                self.snapshot_manager.update_progress(
                    path, 80, "indexing", len(files_to_add_list), len(all_chunks), total_files, files_to_process
                )

                # Add chunks to BM25
                bm25_indexer.add_chunks(all_chunks)

                # Update progress
                self.snapshot_manager.update_progress(
                    path, 90, "indexing", len(files_to_add_list), len(all_chunks), total_files, files_to_process
                )

                # Get the newly inserted chunk IDs (last N IDs)
                all_chunk_ids = bm25_indexer.get_all_chunk_ids()
                new_chunk_ids = all_chunk_ids[-len(all_chunks):]

                if len(new_chunk_ids) == len(all_vectors):
                    vector_indexer.add_vectors(new_chunk_ids, vectors=all_vectors)

                # Save vector index
                vector_indexer.save()

            # Save Merkle tree
            new_merkle.save(merkle_path)

            # Mark as complete
            self.snapshot_manager.mark_indexed(
                path,
                new_merkle.root_hash,
                len(files),
                db.get_chunk_count(),
            )

        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.snapshot_manager.mark_failed(path, error_msg)
            import sys
            print(f"Indexing failed: {error_msg}", file=sys.stderr)

    def _clear_index(self, path: str) -> None:
        """Clear the index for a codebase."""
        path_hash = self.snapshot_manager.path_to_hash(path)
        index_dir = self.config.indexes_dir / path_hash

        # Delete database
        db_path = index_dir / "chunks.db"
        if db_path.exists():
            db_path.unlink()

        # Delete vector index
        vector_path = index_dir / "vectors.bin"
        if vector_path.exists():
            vector_path.unlink()

        vector_meta = index_dir / "vectors.meta.json"
        if vector_meta.exists():
            vector_meta.unlink()

        # Delete merkle tree
        merkle_path = self.config.merkle_dir / f"{path_hash}.json"
        if merkle_path.exists():
            merkle_path.unlink()

        # Remove from snapshot
        self.snapshot_manager.remove_codebase(path)