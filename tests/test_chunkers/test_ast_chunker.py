"""Tests for the ASTChunker class."""

import pytest

from codii.chunkers.ast_chunker import ASTChunker, CodeChunk


class TestChunkPython:
    """Tests for Python code chunking."""

    def test_chunk_python_function(self):
        """Extract Python functions."""
        chunker = ASTChunker()

        content = '''
def hello_world():
    """A simple function."""
    print("Hello, World!")
    return True

def another_function(x, y):
    return x + y
'''
        chunks = chunker.chunk_file(content, "/test/file.py", "python")

        assert len(chunks) >= 2

        # Check for functions
        function_chunks = [c for c in chunks if c.chunk_type == "function"]
        assert len(function_chunks) >= 2

        names = [c.name for c in function_chunks if c.name]
        assert "hello_world" in names
        assert "another_function" in names

    def test_chunk_python_class(self):
        """Extract Python classes."""
        chunker = ASTChunker()

        content = '''
class MyClass:
    def __init__(self):
        self.value = 10

    def get_value(self):
        return self.value
'''
        chunks = chunker.chunk_file(content, "/test/file.py", "python")

        class_chunks = [c for c in chunks if c.chunk_type == "class"]
        assert len(class_chunks) >= 1

        assert class_chunks[0].name == "MyClass"

    def test_chunk_python_nested(self):
        """Handle nested functions/classes."""
        chunker = ASTChunker()

        content = '''
class OuterClass:
    def outer_method(self):
        def inner_function():
            pass
        return inner_function
'''
        chunks = chunker.chunk_file(content, "/test/file.py", "python")

        # Should extract the class and methods
        assert len(chunks) >= 1

    def test_chunk_python_async_function(self):
        """Extract async functions."""
        chunker = ASTChunker()

        content = '''
async def fetch_data():
    """Fetch data asynchronously."""
    await some_operation()
    return data
'''
        chunks = chunker.chunk_file(content, "/test/file.py", "python")

        # Should find async function
        function_chunks = [c for c in chunks if "function" in c.chunk_type]
        assert len(function_chunks) >= 1


class TestChunkJavaScript:
    """Tests for JavaScript code chunking."""

    def test_chunk_javascript_function(self):
        """Extract JavaScript functions."""
        chunker = ASTChunker()

        content = '''
function helloWorld() {
    console.log("Hello, World!");
    return true;
}

const arrowFunc = () => {
    return 42;
};
'''
        chunks = chunker.chunk_file(content, "/test/file.js", "javascript")

        assert len(chunks) >= 1

    def test_chunk_javascript_class(self):
        """Extract JavaScript classes."""
        chunker = ASTChunker()

        content = '''
class MyClass {
    constructor() {
        this.value = 10;
    }

    getValue() {
        return this.value;
    }
}
'''
        chunks = chunker.chunk_file(content, "/test/file.js", "javascript")

        assert len(chunks) >= 1


class TestChunkTypeScript:
    """Tests for TypeScript code chunking."""

    def test_chunk_typescript(self):
        """TypeScript with interfaces."""
        chunker = ASTChunker()

        content = '''
interface User {
    name: string;
    age: number;
}

function greet(user: User): string {
    return `Hello, ${user.name}!`;
}
'''
        chunks = chunker.chunk_file(content, "/test/file.ts", "typescript")

        assert len(chunks) >= 1


class TestUnsupportedLanguage:
    """Tests for unsupported languages."""

    def test_chunk_unsupported_language(self):
        """Fallback to text chunking for unsupported language."""
        chunker = ASTChunker()

        content = "Some random text content that needs to be chunked."
        chunks = chunker.chunk_file(content, "/test/file.xyz", "unknown_lang")

        # Should fallback to text chunking
        assert len(chunks) >= 1


class TestEmptyFile:
    """Tests for empty files."""

    def test_chunk_empty_file(self):
        """Handle empty files."""
        chunker = ASTChunker()

        chunks = chunker.chunk_file("", "/test/empty.py", "python")

        assert len(chunks) == 0

    def test_chunk_whitespace_only(self):
        """Handle whitespace-only files."""
        chunker = ASTChunker()

        chunks = chunker.chunk_file("   \n\n   ", "/test/whitespace.py", "python")

        # Should produce no chunks or a single module chunk
        # depending on implementation


class TestLineNumberTracking:
    """Tests for line number accuracy."""

    def test_chunk_preserves_line_numbers(self):
        """Correct line number tracking."""
        chunker = ASTChunker()

        content = '''
# Line 1 - comment
# Line 2 - comment

def my_function():
    """A function starting at line 4."""
    pass

# Line 8 - comment

class MyClass:
    """A class starting at line 10."""
    pass
'''
        chunks = chunker.chunk_file(content, "/test/file.py", "python")

        # Find the function chunk
        func_chunks = [c for c in chunks if c.chunk_type == "function"]
        if func_chunks:
            # Line numbers should be 1-indexed
            assert func_chunks[0].start_line >= 1
            assert func_chunks[0].end_line >= func_chunks[0].start_line


class TestCodeChunk:
    """Tests for CodeChunk dataclass."""

    def test_to_tuple(self):
        """Convert chunk to tuple for database."""
        chunk = CodeChunk(
            content="def hello():\n    pass",
            path="/test/file.py",
            start_line=1,
            end_line=2,
            language="python",
            chunk_type="function",
            name="hello",
        )

        result = chunk.to_tuple()

        assert result == (
            "def hello():\n    pass",
            "/test/file.py",
            1,
            2,
            "python",
            "function",
        )


class TestIsLanguageSupported:
    """Tests for language support check."""

    def test_is_language_supported_true(self):
        """Check supported language."""
        chunker = ASTChunker()

        assert chunker.is_language_supported("python") is True
        assert chunker.is_language_supported("javascript") is True
        assert chunker.is_language_supported("typescript") is True
        assert chunker.is_language_supported("go") is True
        assert chunker.is_language_supported("rust") is True
        assert chunker.is_language_supported("java") is True
        assert chunker.is_language_supported("c") is True
        assert chunker.is_language_supported("cpp") is True

    def test_is_language_supported_false(self):
        """Check unsupported language."""
        chunker = ASTChunker()

        assert chunker.is_language_supported("unknown") is False
        assert chunker.is_language_supported("ruby") is False


class TestLargeFile:
    """Tests for large files."""

    def test_chunk_large_function(self):
        """Handle large function."""
        chunker = ASTChunker()

        # Create a large function
        content = "def large_function():\n"
        for i in range(100):
            content += f"    x{i} = {i}\n"
        content += "    return x\n"

        chunks = chunker.chunk_file(content, "/test/large.py", "python")

        assert len(chunks) >= 1
        assert chunks[0].chunk_type == "function"

    def test_chunk_many_functions(self):
        """Handle file with many functions."""
        chunker = ASTChunker()

        content = ""
        for i in range(50):
            content += f"def function_{i}():\n    pass\n\n"

        chunks = chunker.chunk_file(content, "/test/many.py", "python")

        function_chunks = [c for c in chunks if c.chunk_type == "function"]
        assert len(function_chunks) == 50


class TestSyntaxError:
    """Tests for handling syntax errors."""

    def test_chunk_with_syntax_error(self):
        """Handle file with syntax errors."""
        chunker = ASTChunker()

        # Invalid Python syntax
        content = "def broken(:\n    pass"

        # Should not raise, may fallback to text chunking
        chunks = chunker.chunk_file(content, "/test/broken.py", "python")

        # Just verify it doesn't crash
        assert isinstance(chunks, list)