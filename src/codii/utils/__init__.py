"""Utilities package for codii."""

from .config import CodiiConfig, get_config, set_config
from .file_utils import (
    compute_file_hash,
    should_index_file,
    scan_directory,
    get_file_content,
    detect_language,
)

__all__ = [
    "CodiiConfig",
    "get_config",
    "set_config",
    "compute_file_hash",
    "should_index_file",
    "scan_directory",
    "get_file_content",
    "detect_language",
]