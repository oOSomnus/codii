"""Tests for the MerkleTree class."""

import pytest
from pathlib import Path
import hashlib

from codii.merkle.tree import MerkleTree


class TestEmptyTree:
    """Tests for empty Merkle tree."""

    def test_empty_tree(self):
        """Empty tree has default hash."""
        tree = MerkleTree()

        assert len(tree.file_hashes) == 0
        assert tree.root_hash is None

        # Computing root on empty tree
        root = tree.compute_root()
        assert root == hashlib.sha256(b"empty").hexdigest()


class TestSingleFile:
    """Tests for tree with single file."""

    def test_single_file(self):
        """Tree with one file."""
        tree = MerkleTree()

        tree.add_file("/test/file.py", "hash123")
        root = tree.compute_root()

        assert tree.root_hash == "hash123"
        assert root == "hash123"


class TestMultipleFiles:
    """Tests for tree with multiple files."""

    def test_multiple_files(self):
        """Tree with multiple files."""
        tree = MerkleTree()

        tree.add_file("/test/a.py", "hash_a")
        tree.add_file("/test/b.py", "hash_b")
        tree.add_file("/test/c.py", "hash_c")

        root = tree.compute_root()

        assert root is not None
        assert len(tree.file_hashes) == 3

    def test_consistent_root_for_same_files(self):
        """Same files produce same root hash."""
        tree1 = MerkleTree()
        tree2 = MerkleTree()

        files = [
            ("/test/a.py", "hash_a"),
            ("/test/b.py", "hash_b"),
            ("/test/c.py", "hash_c"),
        ]

        for path, hash_val in files:
            tree1.add_file(path, hash_val)
            tree2.add_file(path, hash_val)

        assert tree1.compute_root() == tree2.compute_root()

    def test_different_ordering_same_root(self):
        """Different insertion order produces same root."""
        tree1 = MerkleTree()
        tree2 = MerkleTree()

        # Add in different orders
        tree1.add_file("/test/a.py", "hash_a")
        tree1.add_file("/test/b.py", "hash_b")

        tree2.add_file("/test/b.py", "hash_b")
        tree2.add_file("/test/a.py", "hash_a")

        assert tree1.compute_root() == tree2.compute_root()


class TestComputeRoot:
    """Tests for root hash computation."""

    def test_compute_root(self):
        """Root hash computed correctly."""
        tree = MerkleTree()

        # Add files
        tree.add_file("/test/file1.py", "a" * 64)
        tree.add_file("/test/file2.py", "b" * 64)

        root = tree.compute_root()

        # Should be a valid SHA-256 hash
        assert len(root) == 64
        assert all(c in "0123456789abcdef" for c in root)


class TestSaveLoad:
    """Tests for saving and loading tree."""

    def test_save_load(self, temp_merkle_path):
        """Save and load tree preserves state."""
        tree = MerkleTree()

        tree.add_file("/test/file1.py", "hash1")
        tree.add_file("/test/file2.py", "hash2")
        original_root = tree.compute_root()

        # Save
        tree.save(temp_merkle_path)

        assert temp_merkle_path.exists()

        # Load
        loaded_tree = MerkleTree.load(temp_merkle_path)

        assert loaded_tree is not None
        assert loaded_tree.root_hash == original_root
        assert loaded_tree.file_hashes == tree.file_hashes

    def test_load_nonexistent_file(self, temp_storage_dir):
        """Load from non-existent file returns None."""
        nonexistent = temp_storage_dir / "nonexistent.json"
        result = MerkleTree.load(nonexistent)

        assert result is None

    def test_load_corrupted_file(self, temp_merkle_path):
        """Load from corrupted file returns None."""
        # Write invalid JSON
        temp_merkle_path.parent.mkdir(parents=True, exist_ok=True)
        temp_merkle_path.write_text("not valid json {{{")

        result = MerkleTree.load(temp_merkle_path)

        assert result is None


class TestDiff:
    """Tests for comparing trees."""

    def test_diff_added_files(self):
        """Detect added files."""
        old_tree = MerkleTree()
        old_tree.add_file("/test/file1.py", "hash1")
        old_tree.compute_root()

        new_tree = MerkleTree()
        new_tree.add_file("/test/file1.py", "hash1")
        new_tree.add_file("/test/file2.py", "hash2")  # New file
        new_tree.compute_root()

        added, removed, modified = new_tree.diff(old_tree)

        assert added == {"/test/file2.py"}
        assert removed == set()
        assert modified == set()

    def test_diff_removed_files(self):
        """Detect removed files."""
        old_tree = MerkleTree()
        old_tree.add_file("/test/file1.py", "hash1")
        old_tree.add_file("/test/file2.py", "hash2")
        old_tree.compute_root()

        new_tree = MerkleTree()
        new_tree.add_file("/test/file1.py", "hash1")
        new_tree.compute_root()

        added, removed, modified = new_tree.diff(old_tree)

        assert added == set()
        assert removed == {"/test/file2.py"}
        assert modified == set()

    def test_diff_modified_files(self):
        """Detect modified files (same path, different hash)."""
        old_tree = MerkleTree()
        old_tree.add_file("/test/file1.py", "old_hash")
        old_tree.compute_root()

        new_tree = MerkleTree()
        new_tree.add_file("/test/file1.py", "new_hash")
        new_tree.compute_root()

        added, removed, modified = new_tree.diff(old_tree)

        assert added == set()
        assert removed == set()
        assert modified == {"/test/file1.py"}

    def test_diff_no_changes(self):
        """Identical trees show no diff."""
        tree1 = MerkleTree()
        tree1.add_file("/test/file1.py", "hash1")
        tree1.add_file("/test/file2.py", "hash2")
        tree1.compute_root()

        tree2 = MerkleTree()
        tree2.add_file("/test/file1.py", "hash1")
        tree2.add_file("/test/file2.py", "hash2")
        tree2.compute_root()

        added, removed, modified = tree1.diff(tree2)

        assert added == set()
        assert removed == set()
        assert modified == set()

    def test_diff_multiple_changes(self):
        """Detect multiple types of changes."""
        old_tree = MerkleTree()
        old_tree.add_file("/test/file1.py", "hash1")
        old_tree.add_file("/test/file2.py", "hash2_old")
        old_tree.add_file("/test/file3.py", "hash3")
        old_tree.compute_root()

        new_tree = MerkleTree()
        new_tree.add_file("/test/file1.py", "hash1")  # Unchanged
        new_tree.add_file("/test/file2.py", "hash2_new")  # Modified
        # file3.py removed
        new_tree.add_file("/test/file4.py", "hash4")  # Added
        new_tree.compute_root()

        added, removed, modified = new_tree.diff(old_tree)

        assert added == {"/test/file4.py"}
        assert removed == {"/test/file3.py"}
        assert modified == {"/test/file2.py"}


class TestEdgeCases:
    """Tests for edge cases."""

    def test_add_same_file_twice(self):
        """Adding same file twice overwrites."""
        tree = MerkleTree()

        tree.add_file("/test/file.py", "hash1")
        tree.add_file("/test/file.py", "hash2")

        assert len(tree.file_hashes) == 1
        assert tree.file_hashes["/test/file.py"] == "hash2"

    def test_very_long_path(self):
        """Handle very long file paths."""
        tree = MerkleTree()

        long_path = "/test/" + "a" * 1000 + "/file.py"
        tree.add_file(long_path, "hash")

        assert long_path in tree.file_hashes

    def test_unicode_path(self):
        """Handle unicode characters in path."""
        tree = MerkleTree()

        unicode_path = "/test/ファイル/питон.py"
        tree.add_file(unicode_path, "hash")

        assert unicode_path in tree.file_hashes