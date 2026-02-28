"""MCP server entry point for codii."""

import sys
from mcp.server.fastmcp import FastMCP

from .tools.index_codebase import IndexCodebaseTool
from .tools.search_code import SearchCodeTool
from .tools.clear_index import ClearIndexTool
from .tools.status import GetIndexingStatusTool
from .utils.config import get_config

# Create FastMCP server
mcp = FastMCP("codii")

# Initialize tools
index_tool = IndexCodebaseTool()
search_tool = SearchCodeTool()
clear_tool = ClearIndexTool()
status_tool = GetIndexingStatusTool()


@mcp.tool(
    name="index_codebase",
    description="Index a codebase for semantic search. Supports Python, JavaScript, TypeScript, Go, Rust, Java, C/C++. Returns immediately; use get_indexing_status to check progress."
)
def index_codebase(
    path: str,
    force: bool = False,
    splitter: str = "ast",
    customExtensions: list = None,
    ignorePatterns: list = None,
) -> str:
    """
    Index a codebase for semantic search.

    Args:
        path: Absolute path to the codebase to index
        force: Force re-indexing even if already indexed
        splitter: Code splitting method (ast or langchain)
        customExtensions: Additional file extensions to index
        ignorePatterns: Additional patterns to ignore

    Returns:
        Status message
    """
    result = index_tool.run(
        path=path,
        force=force,
        splitter=splitter,
        customExtensions=customExtensions or [],
        ignorePatterns=ignorePatterns or [],
    )
    return result["content"][0]["text"]


@mcp.tool(
    name="search_code",
    description="Search indexed code using hybrid BM25 + vector search. Returns code snippets with file locations and context."
)
def search_code(
    path: str,
    query: str,
    limit: int = 10,
    extensionFilter: list = None,
) -> str:
    """
    Search indexed code using hybrid BM25 + vector search.

    Args:
        path: Absolute path to the indexed codebase
        query: Search query
        limit: Maximum number of results (default 10, max 50)
        extensionFilter: Filter by file extensions (e.g., ['.py', '.js'])

    Returns:
        Search results with code snippets
    """
    result = search_tool.run(
        path=path,
        query=query,
        limit=limit,
        extensionFilter=extensionFilter or [],
    )
    return result["content"][0]["text"]


@mcp.tool(
    name="clear_index",
    description="Clear the index for a codebase, removing all indexed data."
)
def clear_index(path: str) -> str:
    """
    Clear the index for a codebase.

    Args:
        path: Absolute path to the codebase to clear

    Returns:
        Status message
    """
    result = clear_tool.run(path=path)
    return result["content"][0]["text"]


@mcp.tool(
    name="get_indexing_status",
    description="Get the current indexing status for a codebase. Shows progress percentage, current stage, and file/chunk counts."
)
def get_indexing_status(path: str) -> str:
    """
    Get the indexing status for a codebase.

    Args:
        path: Absolute path to the codebase

    Returns:
        Status information
    """
    result = status_tool.run(path=path)
    return result["content"][0]["text"]


def main():
    """Main entry point."""
    # Initialize config
    config = get_config()

    print(f"Starting codii MCP server...", file=sys.stderr)
    print(f"Index storage: {config.base_dir}", file=sys.stderr)

    # Run the server
    mcp.run()


if __name__ == "__main__":
    main()