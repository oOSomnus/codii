"""Get indexing status tool for codii."""

from pathlib import Path

from ..utils.config import get_config
from ..storage.snapshot import SnapshotManager


class GetIndexingStatusTool:
    """Tool to get the indexing status of a codebase."""

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
                    "description": "Absolute path to the codebase",
                },
            },
            "required": ["path"],
        }

    def run(self, path: str) -> dict:
        """
        Get the indexing status for a codebase.

        Args:
            path: Absolute path to the codebase

        Returns:
            Dict with status message or error
        """
        # Validate path
        repo_path = Path(path).resolve()
        path_str = str(repo_path)

        # Get status
        status = self.snapshot_manager.get_status(path_str)

        # Build response based on status
        if status.status == "indexed":
            message = (
                f"Codebase is fully indexed and ready for search.\n"
                f"  indexedFiles: {status.indexed_files}\n"
                f"  totalChunks: {status.total_chunks}\n"
                f"  lastUpdated: {status.last_updated}"
            )
        elif status.status == "indexing":
            stage_display = status.current_stage or "unknown"
            # Build files processed message with context
            if status.files_to_process > 0 and status.files_to_process != status.total_files:
                # Incremental update context
                files_msg = f"{status.indexed_files} of {status.files_to_process} changed ({status.total_files} total)"
            elif status.total_files > 0:
                # Full index context
                files_msg = f"{status.indexed_files} of {status.total_files}"
            else:
                # Fallback without context
                files_msg = str(status.indexed_files)

            message = (
                f"Indexing in progress.\n"
                f"  Progress: {status.progress}%\n"
                f"  Stage: {stage_display}\n"
                f"  Files processed: {files_msg}\n"
                f"  Chunks created: {status.total_chunks}"
            )
        elif status.status == "failed":
            message = (
                f"Indexing failed at {status.progress}%.\n"
                f"  Error: {status.error_message}\n"
                f"You can retry with index_codebase."
            )
        else:  # not_found
            message = (
                f"Codebase not indexed.\n"
                f"Use index_codebase to start indexing."
            )

        return {
            "content": [{
                "type": "text",
                "text": message
            }],
            "isError": False,
        }