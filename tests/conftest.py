"""Shared fixtures for codii tests."""

import pytest
import tempfile
from pathlib import Path
import os
import shutil
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock

from codii.utils.config import CodiiConfig, set_config


@pytest.fixture
def temp_dir():
    """Create a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def temp_storage_dir(tmp_path):
    """Create a temporary storage directory for tests."""
    storage = tmp_path / ".codii"
    storage.mkdir(parents=True, exist_ok=True)
    return storage


@pytest.fixture
def mock_config(temp_storage_dir):
    """Mock configuration with temp storage."""
    config = CodiiConfig()
    config.base_dir = temp_storage_dir
    set_config(config)
    yield config
    # Reset global config after test
    import codii.utils.config as config_module
    config_module._config = None


@pytest.fixture
def mock_embedder():
    """Create a mock embedder that returns deterministic vectors."""
    mock = MagicMock()
    mock._model = MagicMock()
    mock._embedding_dim = 384
    mock._initialized = True
    mock.model_name = "mock-model"

    def mock_embed(texts):
        # Return deterministic vectors based on text hash
        if not texts:
            return np.array([])
        np.random.seed(42)  # Deterministic
        return np.random.rand(len(texts), 384).astype(np.float32)

    def mock_embed_single(text):
        np.random.seed(42)
        return np.random.rand(384).astype(np.float32)

    mock.embed = mock_embed
    mock.embed_single = mock_embed_single
    mock.embedding_dim = 384

    return mock


@pytest.fixture
def mock_cross_encoder():
    """Create a mock cross-encoder for testing."""
    from codii.indexers.hybrid_search import SearchResult

    mock = MagicMock()
    mock._model = MagicMock()
    mock._initialized = True
    mock.model_name = "mock-cross-encoder"

    def mock_rerank(query, candidates, top_k=10, threshold=0.5):
        # Return candidates with mock rerank scores
        import math
        results = []
        for i, candidate in enumerate(candidates[:top_k]):
            # Give each result a score based on position (higher = better)
            score = 0.9 - (i * 0.1)
            if score >= threshold:
                candidate.rerank_score = score
                results.append(candidate)
        return results

    mock.rerank = mock_rerank
    return mock


@pytest.fixture(autouse=True)
def auto_mock_embedder(mock_embedder, mock_cross_encoder):
    """Automatically mock the embedder and cross-encoder for all tests."""
    # Patch get_embedder function everywhere it might be imported
    with patch('codii.embedding.embedder.get_embedder', return_value=mock_embedder):
        with patch('codii.indexers.vector_indexer.get_embedder', return_value=mock_embedder):
            # Patch get_cross_encoder for all tests
            with patch('codii.embedding.cross_encoder.get_cross_encoder', return_value=mock_cross_encoder):
                # Also reset the singleton instances
                import codii.embedding.embedder as embedder_module
                embedder_module.Embedder._instance = None
                import codii.embedding.cross_encoder as cross_encoder_module
                cross_encoder_module.CrossEncoderWrapper._instance = None
                yield mock_embedder


@pytest.fixture
def sample_python_file(temp_dir):
    """Create a sample Python file."""
    file_path = temp_dir / "sample.py"
    file_path.write_text('''
def hello_world():
    """A simple function."""
    print("Hello, World!")
    return True

class MyClass:
    def __init__(self):
        self.value = 10

    def get_value(self):
        return self.value
''')
    return file_path


@pytest.fixture
def sample_javascript_file(temp_dir):
    """Create a sample JavaScript file."""
    file_path = temp_dir / "sample.js"
    file_path.write_text('''
function helloWorld() {
    console.log("Hello, World!");
    return true;
}

class MyClass {
    constructor() {
        this.value = 10;
    }

    getValue() {
        return this.value;
    }
}
''')
    return file_path


@pytest.fixture
def sample_typescript_file(temp_dir):
    """Create a sample TypeScript file."""
    file_path = temp_dir / "sample.ts"
    file_path.write_text('''
interface User {
    name: string;
    age: number;
}

function greet(user: User): string {
    return `Hello, ${user.name}!`;
}

class UserService {
    private users: User[] = [];

    addUser(user: User): void {
        this.users.push(user);
    }
}
''')
    return file_path


@pytest.fixture
def sample_codebase(temp_dir):
    """Create a sample codebase with multiple files."""
    # Create main files
    (temp_dir / "main.py").write_text('''
def main():
    """Main entry point."""
    print("Starting application")
    helper()

def helper():
    """Helper function."""
    return 42

if __name__ == "__main__":
    main()
''')

    (temp_dir / "utils.py").write_text('''
def format_string(s: str) -> str:
    """Format a string."""
    return s.strip().lower()

def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two numbers."""
    return a + b
''')

    # Create subdirectory with files
    subdir = temp_dir / "subdir"
    subdir.mkdir()
    (subdir / "module.py").write_text('''
class Module:
    """A sample module class."""

    def __init__(self, name: str):
        self.name = name

    def process(self) -> str:
        return f"Processing {self.name}"
''')

    return temp_dir


@pytest.fixture
def large_python_file(temp_dir):
    """Create a larger Python file for testing chunking."""
    content = '''
"""A larger module for testing."""

def function_one():
    """First function."""
    x = 1
    y = 2
    return x + y


def function_two(a: int, b: int) -> int:
    """Second function with parameters."""
    result = a * b
    for i in range(10):
        result += i
    return result


class FirstClass:
    """First test class."""

    def __init__(self):
        self.value = 100

    def method_one(self):
        """Method one."""
        return self.value

    def method_two(self, multiplier: int):
        """Method two."""
        return self.value * multiplier


class SecondClass:
    """Second test class."""

    def __init__(self, name: str):
        self.name = name
        self.data = []

    def add_data(self, item):
        """Add item to data."""
        self.data.append(item)

    def get_data(self):
        """Get all data."""
        return self.data.copy()


def standalone_function():
    """A standalone function at the end."""
    items = [1, 2, 3, 4, 5]
    return sum(items)
'''
    file_path = temp_dir / "large_module.py"
    file_path.write_text(content)
    return file_path


@pytest.fixture
def empty_file(temp_dir):
    """Create an empty file."""
    file_path = temp_dir / "empty.py"
    file_path.write_text("")
    return file_path


@pytest.fixture
def whitespace_only_file(temp_dir):
    """Create a file with only whitespace."""
    file_path = temp_dir / "whitespace.py"
    file_path.write_text("   \n\n   \t\n   ")
    return file_path


@pytest.fixture
def temp_db_path(temp_storage_dir):
    """Get a temporary database path."""
    return temp_storage_dir / "indexes" / "test_hash" / "chunks.db"


@pytest.fixture
def temp_vector_path(temp_storage_dir):
    """Get a temporary vector index path."""
    return temp_storage_dir / "indexes" / "test_hash" / "vectors"


@pytest.fixture
def temp_snapshot_path(temp_storage_dir):
    """Get a temporary snapshot file path."""
    return temp_storage_dir / "snapshots" / "snapshot.json"


@pytest.fixture
def temp_merkle_path(temp_storage_dir):
    """Get a temporary merkle tree file path."""
    return temp_storage_dir / "merkle" / "test_hash.json"


@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset global state before and after each test."""
    # Reset config before test
    import codii.utils.config as config_module
    config_module._config = None

    # Reset embedder singleton
    import codii.embedding.embedder as embedder_module
    embedder_module.Embedder._instance = None

    # Reset cross-encoder singleton
    import codii.embedding.cross_encoder as cross_encoder_module
    cross_encoder_module.CrossEncoderWrapper._instance = None

    yield

    # Reset after test
    config_module._config = None
    embedder_module.Embedder._instance = None
    cross_encoder_module.CrossEncoderWrapper._instance = None


@pytest.fixture
def sample_chunks():
    """Create sample CodeChunk objects for testing."""
    from codii.chunkers.text_chunker import CodeChunk

    return [
        CodeChunk(
            content="def hello():\n    print('hello')",
            path="/test/file1.py",
            start_line=1,
            end_line=2,
            language="python",
            chunk_type="function",
            name="hello",
        ),
        CodeChunk(
            content="class World:\n    pass",
            path="/test/file1.py",
            start_line=5,
            end_line=6,
            language="python",
            chunk_type="class",
            name="World",
        ),
        CodeChunk(
            content="const x = 1;",
            path="/test/file2.js",
            start_line=1,
            end_line=1,
            language="javascript",
            chunk_type="text_block",
        ),
    ]