"""Clear index tool for codii."""

from pathlib import Path

from ..utils.config import get_config
from ..storage.snapshot import SnapshotManager


class ClearIndexTool:
    """Tool to clear an indexed codebase."""

    def __init__(self):
        self.config = get_config()
        self.snapshot_manager = SnapshotManager(self.config.snapshot_file)

    def get_input_schema(self) -> dict:
        """Get the input schema for the tool."""
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the codebase to clear",
                },
            },
            "required": ["path"],
        }

    def run(self, path: str) -> dict:
        """
        Clear the index for a codebase.

        Args:
            path: Absolute path to the codebase

        Returns:
            Dict with result message or error
        """
        # Check if any codebases are indexed
        if not self.snapshot_manager.has_any_codebases():
            return {
                "content": [{
                    "type": "text",
                    "text": "No codebases are currently indexed or being indexed."
                }],
                "isError": False,
            }

        # Validate path
        repo_path = Path(path).resolve()
        path_str = str(repo_path)

        # Check if this path is indexed or indexing
        status = self.snapshot_manager.get_status(path_str)

        if status.status == "not_found":
            return {
                "content": [{
                    "type": "text",
                    "text": f"Codebase not found in index: {path}"
                }],
                "isError": True,
            }

        # Check if currently indexing
        if status.status == "indexing":
            return {
                "content": [{
                    "type": "text",
                    "text": f"Cannot clear index: codebase is currently being indexed. Wait for indexing to complete."
                }],
                "isError": True,
            }

        # Get index paths
        path_hash = self.snapshot_manager.path_to_hash(path_str)
        index_dir = self.config.indexes_dir / path_hash

        # Delete database
        db_path = index_dir / "chunks.db"
        if db_path.exists():
            db_path.unlink()

        # Delete vector index files
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

        # Remove index directory if empty
        try:
            if index_dir.exists() and not any(index_dir.iterdir()):
                index_dir.rmdir()
        except Exception:
            pass

        # Remove from snapshot
        self.snapshot_manager.remove_codebase(path_str)

        return {
            "content": [{
                "type": "text",
                "text": f"Index cleared for {path}"
            }],
            "isError": False,
        }