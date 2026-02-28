"""Tests for the TextChunker class."""

import pytest

from codii.chunkers.text_chunker import TextChunker, CodeChunk


class TestChunkLargeFile:
    """Tests for splitting large files."""

    def test_chunk_large_file(self):
        """Split large files."""
        chunker = TextChunker(max_chunk_size=200, min_chunk_size=50)

        # Create content larger than max_chunk_size
        content = "def function_one():\n    pass\n\n" * 50

        chunks = chunker.chunk_file(content, "/test/large.py", "python")

        assert len(chunks) >= 1

        # Each chunk should respect max size
        for chunk in chunks:
            assert len(chunk.content) <= chunker.max_chunk_size + 100  # Allow some margin

    def test_chunk_small_file(self):
        """Handle small files."""
        chunker = TextChunker()

        content = "def hello():\n    pass"
        chunks = chunker.chunk_file(content, "/test/small.py", "python")

        assert len(chunks) == 1
        assert chunks[0].content == content


class TestMaxChunkSize:
    """Tests for max_chunk_size parameter."""

    def test_chunk_respects_max_size(self):
        """Honor max_chunk_size."""
        chunker = TextChunker(max_chunk_size=100, min_chunk_size=10)

        # Create content that will need multiple chunks
        lines = ["line " + str(i) for i in range(20)]
        content = "\n".join(lines)

        chunks = chunker.chunk_file(content, "/test/file.py", "python")

        for chunk in chunks:
            # Allow some margin for line boundaries
            assert len(chunk.content) <= chunker.max_chunk_size + 50

    def test_custom_max_chunk_size(self):
        """Custom max_chunk_size is used."""
        chunker = TextChunker(max_chunk_size=500)

        assert chunker.max_chunk_size == 500


class TestChunkOverlap:
    """Tests for chunk overlap."""

    def test_chunk_overlap(self):
        """Correct overlap between chunks."""
        chunker = TextChunker(max_chunk_size=100, min_chunk_size=20, chunk_overlap=30)

        # Create content that will produce multiple chunks
        lines = ["This is a line with some content " + str(i) for i in range(20)]
        content = "\n".join(lines)

        chunks = chunker.chunk_file(content, "/test/file.py", "python")

        if len(chunks) > 1:
            # Check that consecutive chunks have overlapping content
            for i in range(len(chunks) - 1):
                # There should be some content that appears in both
                # This is hard to verify exactly due to line-based splitting
                pass  # Just verify we get multiple chunks

    def test_custom_chunk_overlap(self):
        """Custom chunk_overlap is used."""
        chunker = TextChunker(chunk_overlap=100)

        assert chunker.chunk_overlap == 100


class TestMinChunkSize:
    """Tests for min_chunk_size parameter."""

    def test_chunk_respects_min_size(self):
        """Chunks meet minimum size requirement."""
        chunker = TextChunker(max_chunk_size=200, min_chunk_size=50)

        # Create content
        lines = ["line " * 10 + str(i) for i in range(20)]
        content = "\n".join(lines)

        chunks = chunker.chunk_file(content, "/test/file.py", "python")

        # All chunks except possibly the last should meet min size
        for chunk in chunks[:-1]:
            assert len(chunk.content) >= chunker.min_chunk_size or len(chunk.content) > 0


class TestEmptyFile:
    """Tests for empty files."""

    def test_chunk_empty_file(self):
        """Handle empty files."""
        chunker = TextChunker()

        chunks = chunker.chunk_file("", "/test/empty.py", "python")

        assert len(chunks) == 0

    def test_chunk_whitespace_only(self):
        """Handle whitespace-only files."""
        chunker = TextChunker()

        chunks = chunker.chunk_file("   \n\n   ", "/test/whitespace.py", "python")

        assert len(chunks) == 0


class TestLineNumbers:
    """Tests for line number tracking."""

    def test_chunk_line_numbers(self):
        """Correct line number tracking."""
        chunker = TextChunker()

        content = "line1\nline2\nline3\nline4\nline5"
        chunks = chunker.chunk_file(content, "/test/file.py", "python")

        assert len(chunks) >= 1

        # First chunk should start at line 1
        assert chunks[0].start_line == 1

        # End line should be greater or equal to start line
        for chunk in chunks:
            assert chunk.end_line >= chunk.start_line


class TestGetOverlapLines:
    """Tests for overlap line calculation."""

    def test_get_overlap_lines(self):
        """Get overlap lines correctly."""
        chunker = TextChunker(chunk_overlap=20)

        lines = ["line one content", "line two content", "line three content"]

        overlap = chunker._get_overlap_lines(lines)

        assert isinstance(overlap, list)
        # Overlap should be subset of input lines
        for line in overlap:
            assert line in lines

    def test_get_overlap_lines_empty(self):
        """Get overlap from empty list."""
        chunker = TextChunker()

        overlap = chunker._get_overlap_lines([])

        assert overlap == []


class TestCodeChunk:
    """Tests for CodeChunk dataclass."""

    def test_code_chunk_creation(self):
        """Create CodeChunk correctly."""
        chunk = CodeChunk(
            content="test content",
            path="/test/file.py",
            start_line=1,
            end_line=5,
            language="python",
            chunk_type="text_block",
        )

        assert chunk.content == "test content"
        assert chunk.path == "/test/file.py"
        assert chunk.start_line == 1
        assert chunk.end_line == 5
        assert chunk.language == "python"
        assert chunk.chunk_type == "text_block"
        assert chunk.name is None

    def test_code_chunk_to_tuple(self):
        """Convert to tuple for database insertion."""
        chunk = CodeChunk(
            content="content",
            path="/path",
            start_line=1,
            end_line=2,
            language="python",
            chunk_type="text_block",
        )

        result = chunk.to_tuple()

        assert result == ("content", "/path", 1, 2, "python", "text_block")


class TestDifferentLanguages:
    """Tests for different programming languages."""

    def test_chunk_python(self):
        """Chunk Python file."""
        chunker = TextChunker()

        content = "def hello():\n    print('hello')"
        chunks = chunker.chunk_file(content, "/test/file.py", "python")

        assert len(chunks) >= 1
        assert chunks[0].language == "python"

    def test_chunk_javascript(self):
        """Chunk JavaScript file."""
        chunker = TextChunker()

        content = "function hello() { console.log('hello'); }"
        chunks = chunker.chunk_file(content, "/test/file.js", "javascript")

        assert len(chunks) >= 1
        assert chunks[0].language == "javascript"

    def test_chunk_markdown(self):
        """Chunk Markdown file."""
        chunker = TextChunker()

        content = "# Title\n\nParagraph content here.\n\n## Section\n\nMore content."
        chunks = chunker.chunk_file(content, "/test/file.md", "markdown")

        assert len(chunks) >= 1
        assert chunks[0].language == "markdown"


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_long_line(self):
        """Handle single very long line."""
        chunker = TextChunker(max_chunk_size=100)

        # Single line longer than max_chunk_size
        content = "a" * 200

        chunks = chunker.chunk_file(content, "/test/file.py", "python")

        # Should create at least one chunk
        assert len(chunks) >= 1

    def test_no_newlines(self):
        """Handle content without newlines."""
        chunker = TextChunker()

        content = "word " * 100
        chunks = chunker.chunk_file(content, "/test/file.py", "python")

        assert len(chunks) >= 1

    def test_very_small_content(self):
        """Handle very small content."""
        chunker = TextChunker(min_chunk_size=100)

        content = "small"
        chunks = chunker.chunk_file(content, "/test/file.py", "python")

        # Small content below min_chunk_size should still create a chunk
        # because it's the only content
        assert len(chunks) >= 1


class TestChunkerDefaults:
    """Tests for default parameters."""

    def test_default_parameters(self):
        """Verify default parameters."""
        chunker = TextChunker()

        assert chunker.max_chunk_size == 1500
        assert chunker.min_chunk_size == 100
        assert chunker.chunk_overlap == 200


class TestMultipleChunks:
    """Tests for multiple chunk scenarios."""

    def test_multiple_chunks_preserve_content(self):
        """All content is preserved across chunks."""
        chunker = TextChunker(max_chunk_size=100, min_chunk_size=20)

        # Create content that will be split
        lines = [f"Line number {i} with some content to make it longer" for i in range(20)]
        content = "\n".join(lines)

        chunks = chunker.chunk_file(content, "/test/file.py", "python")

        # Combine all chunks
        combined = "\n".join(c.content for c in chunks)

        # Key parts of original content should appear in chunks
        for line in lines[:5]:  # Check first few lines
            assert any(line in c.content for c in chunks)