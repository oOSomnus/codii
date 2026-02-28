"""File scanning utilities for codii."""

import hashlib
import os
from pathlib import Path
from typing import List, Optional, Set, Tuple
import pathspec


def read_gitignore(root_path: Path) -> List[str]:
    """
    Read .gitignore patterns from the specified directory.

    Args:
        root_path: Path to the directory containing .gitignore

    Returns:
        List of gitignore patterns, empty if file doesn't exist
    """
    gitignore_path = root_path / ".gitignore"
    if not gitignore_path.exists():
        return []

    try:
        with open(gitignore_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        patterns = []
        for line in lines:
            # Strip whitespace and skip empty lines and comments
            line = line.strip()
            if line and not line.startswith("#"):
                patterns.append(line)

        return patterns
    except (PermissionError, OSError):
        return []


def compute_file_hash(file_path: Path, chunk_size: int = 8192) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()


def should_index_file(
    file_path: Path,
    extensions: Set[str],
    ignore_spec: Optional[pathspec.PathSpec] = None,
    custom_extensions: Optional[List[str]] = None,
) -> bool:
    """Check if a file should be indexed based on extension and ignore patterns."""
    # Check ignore patterns first
    if ignore_spec:
        # Get relative path for matching
        try:
            rel_path = str(file_path)
            if ignore_spec.match_file(rel_path):
                return False
        except Exception:
            pass

    # Normalize extension (ensure it starts with .)
    ext = file_path.suffix.lower()
    if not ext.startswith("."):
        ext = "." + ext

    # Check if extension is in allowed list
    if ext in extensions:
        return True

    # Check custom extensions
    if custom_extensions:
        for custom_ext in custom_extensions:
            normalized = custom_ext if custom_ext.startswith(".") else "." + custom_ext
            if ext == normalized.lower():
                return True

    return False


def scan_directory(
    root_path: Path,
    extensions: Set[str],
    ignore_patterns: List[str],
    custom_extensions: Optional[List[str]] = None,
    custom_ignore: Optional[List[str]] = None,
    use_gitignore: bool = True,
) -> List[Tuple[Path, str]]:
    """
    Scan a directory for files to index.

    Args:
        root_path: Root directory to scan
        extensions: Set of file extensions to include
        ignore_patterns: Default ignore patterns
        custom_extensions: Additional extensions to include
        custom_ignore: Additional patterns to ignore
        use_gitignore: Whether to read and apply .gitignore patterns

    Returns:
        List of (file_path, file_hash) tuples
    """
    files = []

    # Combine default and custom ignore patterns
    all_ignore = list(ignore_patterns)
    if custom_ignore:
        all_ignore.extend(custom_ignore)

    # Add .gitignore patterns if enabled
    if use_gitignore:
        gitignore_patterns = read_gitignore(root_path)
        all_ignore.extend(gitignore_patterns)

    # Create pathspec for ignore patterns
    ignore_spec = pathspec.PathSpec.from_lines(
        pathspec.patterns.GitWildMatchPattern,
        all_ignore
    )

    # Walk the directory
    for root, dirs, filenames in os.walk(root_path):
        root_dir = Path(root)

        # Filter out ignored directories (modify dirs in-place to prevent walking them)
        dirs[:] = [
            d for d in dirs
            if not ignore_spec.match_file(str(root_dir / d))
        ]

        for filename in filenames:
            file_path = root_dir / filename

            if should_index_file(file_path, extensions, ignore_spec, custom_extensions):
                try:
                    file_hash = compute_file_hash(file_path)
                    files.append((file_path, file_hash))
                except (PermissionError, OSError) as e:
                    # Skip files we can't read
                    import sys
                    print(f"Warning: Cannot read {file_path}: {e}", file=sys.stderr)
                    continue

    return files


def get_file_content(file_path: Path) -> Optional[str]:
    """Read file content, returning None if not readable."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except (PermissionError, OSError) as e:
        import sys
        print(f"Warning: Cannot read {file_path}: {e}", file=sys.stderr)
        return None


def detect_language(file_path: Path) -> str:
    """Detect programming language from file extension."""
    ext_to_lang = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".go": "go",
        ".rs": "rust",
        ".java": "java",
        ".c": "c",
        ".h": "c",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".hpp": "cpp",
        ".hxx": "cpp",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".md": "markdown",
        ".rst": "rst",
        ".txt": "text",
        ".sh": "shell",
        ".bash": "shell",
        ".zsh": "shell",
        ".sql": "sql",
        ".proto": "protobuf",
        ".html": "html",
        ".css": "css",
        ".scss": "scss",
        ".less": "less",
    }
    ext = file_path.suffix.lower()
    return ext_to_lang.get(ext, "text")