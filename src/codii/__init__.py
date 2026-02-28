"""
Codii - Local code repository indexing MCP server.

Provides hybrid BM25 + vector search for code repositories.
"""

__version__ = "0.1.0"

from .server import mcp, main

__all__ = ["mcp", "main"]