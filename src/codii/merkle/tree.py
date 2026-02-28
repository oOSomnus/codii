"""Merkle tree implementation for change detection."""

import json
from pathlib import Path
from typing import Optional
import hashlib


class MerkleTree:
    """
    Simple Merkle tree for file change detection.
    Stores hashes of files and computes a root hash.
    """

    def __init__(self):
        self.file_hashes: dict[str, str] = {}
        self.root_hash: Optional[str] = None

    def add_file(self, path: str, file_hash: str) -> None:
        """Add a file hash to the tree."""
        self.file_hashes[path] = file_hash

    def compute_root(self) -> str:
        """Compute the Merkle root hash."""
        if not self.file_hashes:
            self.root_hash = hashlib.sha256(b"empty").hexdigest()
            return self.root_hash

        # Sort paths for consistent ordering
        sorted_paths = sorted(self.file_hashes.keys())

        # Build a simple Merkle tree by combining hashes
        hashes = [self.file_hashes[p] for p in sorted_paths]

        while len(hashes) > 1:
            new_level = []
            for i in range(0, len(hashes), 2):
                if i + 1 < len(hashes):
                    combined = hashes[i] + hashes[i + 1]
                    new_hash = hashlib.sha256(combined.encode()).hexdigest()
                else:
                    new_hash = hashes[i]
                new_level.append(new_hash)
            hashes = new_level

        self.root_hash = hashes[0]
        return self.root_hash

    def save(self, path: Path) -> None:
        """Save the Merkle tree state to a file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "root_hash": self.root_hash,
            "file_hashes": self.file_hashes,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> Optional["MerkleTree"]:
        """Load a Merkle tree from a file."""
        if not path.exists():
            return None

        try:
            with open(path, "r") as f:
                data = json.load(f)

            tree = cls()
            tree.root_hash = data.get("root_hash")
            tree.file_hashes = data.get("file_hashes", {})
            return tree
        except (json.JSONDecodeError, KeyError):
            return None

    def diff(self, other: "MerkleTree") -> tuple[set[str], set[str], set[str]]:
        """
        Compare with another Merkle tree.

        Returns:
            Tuple of (added_files, removed_files, modified_files)
        """
        self_paths = set(self.file_hashes.keys())
        other_paths = set(other.file_hashes.keys())

        added = self_paths - other_paths
        removed = other_paths - self_paths

        # Check for modified files (same path, different hash)
        common = self_paths & other_paths
        modified = {
            path for path in common
            if self.file_hashes[path] != other.file_hashes.get(path)
        }

        return added, removed, modified