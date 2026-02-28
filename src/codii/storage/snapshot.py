"""Snapshot management for codii."""

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict
import hashlib


@dataclass
class CodebaseStatus:
    """Status information for a single codebase."""

    path: str
    status: str  # "indexed", "indexing", "failed", "not_found"
    progress: int = 0
    current_stage: str = ""  # "preparing", "chunking", "embedding", "indexing"
    merkle_root: Optional[str] = None
    indexed_files: int = 0
    total_chunks: int = 0
    last_updated: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CodebaseStatus":
        return cls(**data)


class SnapshotManager:
    """Manages index state persistence."""

    def __init__(self, snapshot_file: Path):
        self.snapshot_file = snapshot_file
        self.snapshot_file.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._ensure_snapshot_exists()

    def _ensure_snapshot_exists(self) -> None:
        """Create snapshot file if it doesn't exist."""
        if not self.snapshot_file.exists():
            self._write_snapshot({"codebases": {}})

    def _read_snapshot(self) -> dict:
        """Read snapshot from disk."""
        try:
            with open(self.snapshot_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {"codebases": {}}

    def _write_snapshot(self, data: dict) -> None:
        """Write snapshot to disk."""
        with open(self.snapshot_file, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def path_to_hash(path: str) -> str:
        """Convert a path to a unique hash for storage."""
        return hashlib.sha256(path.encode()).hexdigest()[:16]

    def get_status(self, path: str) -> CodebaseStatus:
        """Get status for a codebase."""
        with self._lock:
            snapshot = self._read_snapshot()
            codebases = snapshot.get("codebases", {})

            if path in codebases:
                return CodebaseStatus.from_dict(codebases[path])
            else:
                return CodebaseStatus(
                    path=path,
                    status="not_found",
                )

    def set_status(self, status: CodebaseStatus) -> None:
        """Set status for a codebase."""
        with self._lock:
            snapshot = self._read_snapshot()
            snapshot.setdefault("codebases", {})

            status.last_updated = datetime.now().isoformat()
            snapshot["codebases"][status.path] = status.to_dict()

            self._write_snapshot(snapshot)

    def update_progress(
        self,
        path: str,
        progress: int,
        stage: str,
        indexed_files: int = 0,
        total_chunks: int = 0,
    ) -> None:
        """Update indexing progress."""
        with self._lock:
            snapshot = self._read_snapshot()
            codebases = snapshot.setdefault("codebases", {})

            if path in codebases:
                codebases[path]["progress"] = progress
                codebases[path]["current_stage"] = stage
                codebases[path]["indexed_files"] = indexed_files
                codebases[path]["total_chunks"] = total_chunks
                codebases[path]["last_updated"] = datetime.now().isoformat()
            else:
                codebases[path] = CodebaseStatus(
                    path=path,
                    status="indexing",
                    progress=progress,
                    current_stage=stage,
                    indexed_files=indexed_files,
                    total_chunks=total_chunks,
                ).to_dict()

            self._write_snapshot(snapshot)

    def mark_indexing(self, path: str) -> None:
        """Mark a codebase as currently indexing."""
        self.set_status(CodebaseStatus(
            path=path,
            status="indexing",
            progress=0,
            current_stage="preparing",
        ))

    def mark_indexed(
        self,
        path: str,
        merkle_root: str,
        indexed_files: int,
        total_chunks: int,
    ) -> None:
        """Mark a codebase as fully indexed."""
        self.set_status(CodebaseStatus(
            path=path,
            status="indexed",
            progress=100,
            current_stage="complete",
            merkle_root=merkle_root,
            indexed_files=indexed_files,
            total_chunks=total_chunks,
        ))

    def mark_failed(self, path: str, error_message: str, progress: int = 0) -> None:
        """Mark a codebase as failed."""
        self.set_status(CodebaseStatus(
            path=path,
            status="failed",
            progress=progress,
            error_message=error_message,
        ))

    def remove_codebase(self, path: str) -> bool:
        """Remove a codebase from tracking."""
        with self._lock:
            snapshot = self._read_snapshot()
            codebases = snapshot.get("codebases", {})

            if path in codebases:
                del codebases[path]
                self._write_snapshot(snapshot)
                return True
            return False

    def is_indexing(self, path: str) -> bool:
        """Check if a codebase is currently being indexed."""
        status = self.get_status(path)
        return status.status == "indexing"

    def get_all_codebases(self) -> dict[str, CodebaseStatus]:
        """Get all tracked codebases."""
        with self._lock:
            snapshot = self._read_snapshot()
            return {
                path: CodebaseStatus.from_dict(data)
                for path, data in snapshot.get("codebases", {}).items()
            }

    def has_any_codebases(self) -> bool:
        """Check if any codebases are tracked."""
        with self._lock:
            snapshot = self._read_snapshot()
            return len(snapshot.get("codebases", {})) > 0