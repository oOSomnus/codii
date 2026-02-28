"""AST-based code chunker using tree-sitter."""

import hashlib
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class CodeChunk:
    """A chunk of code with metadata."""
    content: str
    path: str
    start_line: int
    end_line: int
    language: str
    chunk_type: str  # function, class, method, module, etc.
    name: Optional[str] = None  # function/class name

    def to_tuple(self) -> tuple:
        """Convert to tuple for database insertion."""
        return (
            self.content,
            self.path,
            self.start_line,
            self.end_line,
            self.language,
            self.chunk_type,
        )


# Language to tree-sitter language mapping
LANGUAGE_MAP = {
    "python": "python",
    "javascript": "javascript",
    "typescript": "typescript",
    "go": "go",
    "rust": "rust",
    "java": "java",
    "c": "c",
    "cpp": "cpp",
}

# Node types that represent semantic units for each language
SEMANTIC_NODES = {
    "python": {"function_definition", "class_definition", "async_function_definition"},
    "javascript": {"function_declaration", "class_declaration", "method_definition", "arrow_function", "function_expression"},
    "typescript": {"function_declaration", "class_declaration", "method_definition", "arrow_function", "function_expression", "interface_declaration", "type_alias_declaration"},
    "go": {"function_declaration", "method_declaration", "type_declaration"},
    "rust": {"function_definition", "struct_item", "enum_item", "impl_item", "trait_item"},
    "java": {"method_declaration", "class_declaration", "interface_declaration", "enum_declaration"},
    "c": {"function_definition", "struct_specifier", "enum_specifier"},
    "cpp": {"function_definition", "class_specifier", "struct_specifier", "namespace_definition"},
}


class ASTChunker:
    """AST-based code chunker using tree-sitter."""

    def __init__(self):
        self._parsers = {}

    def _get_parser(self, language: str):
        """Get or create a parser for the language."""
        if language not in self._parsers:
            lang_module = None

            # Try to load the language module
            if language == "python":
                import tree_sitter_python
                lang_module = tree_sitter_python
            elif language == "javascript":
                import tree_sitter_javascript
                lang_module = tree_sitter_javascript
            elif language == "typescript":
                import tree_sitter_typescript
                lang_module = tree_sitter_typescript
            elif language == "go":
                import tree_sitter_go
                lang_module = tree_sitter_go
            elif language == "rust":
                import tree_sitter_rust
                lang_module = tree_sitter_rust
            elif language == "java":
                import tree_sitter_java
                lang_module = tree_sitter_java
            elif language == "c":
                import tree_sitter_c
                lang_module = tree_sitter_c
            elif language == "cpp":
                import tree_sitter_cpp
                lang_module = tree_sitter_cpp
            else:
                raise ValueError(f"Unsupported language: {language}")

            # Create Language and Parser
            from tree_sitter import Language, Parser

            # For typescript, use language_typescript()
            if language == "typescript":
                lang = Language(lang_module.language_typescript())
            else:
                lang = Language(lang_module.language())

            parser = Parser(lang)
            self._parsers[language] = parser

        return self._parsers[language]

    def is_language_supported(self, language: str) -> bool:
        """Check if a language is supported for AST parsing."""
        return language in LANGUAGE_MAP

    def chunk_file(
        self,
        content: str,
        path: str,
        language: str,
        max_chunk_size: int = 1500,
        min_chunk_size: int = 100,
    ) -> List[CodeChunk]:
        """
        Chunk a file using AST parsing.

        Args:
            content: File content
            path: File path
            language: Programming language
            max_chunk_size: Maximum chunk size in characters
            min_chunk_size: Minimum chunk size in characters

        Returns:
            List of CodeChunks
        """
        if language not in LANGUAGE_MAP:
            # Fallback to text chunking for unsupported languages
            from .text_chunker import TextChunker
            text_chunker = TextChunker(
                max_chunk_size=max_chunk_size,
                min_chunk_size=min_chunk_size,
            )
            return text_chunker.chunk_file(content, path, language)

        try:
            parser = self._get_parser(language)
            tree = parser.parse(bytes(content, "utf-8"))
            return self._extract_chunks(
                tree.root_node,
                content,
                path,
                language,
                max_chunk_size,
                min_chunk_size,
            )
        except Exception as e:
            import sys
            print(f"Warning: AST parsing failed for {path}: {e}", file=sys.stderr)
            # Fallback to text chunking
            from .text_chunker import TextChunker
            text_chunker = TextChunker(
                max_chunk_size=max_chunk_size,
                min_chunk_size=min_chunk_size,
            )
            return text_chunker.chunk_file(content, path, language)

    def _extract_chunks(
        self,
        node,
        content: str,
        path: str,
        language: str,
        max_chunk_size: int,
        min_chunk_size: int,
    ) -> List[CodeChunk]:
        """Extract semantic chunks from AST nodes."""
        chunks = []
        semantic_types = SEMANTIC_NODES.get(language, set())

        def visit_node(current_node, parent_type=None):
            node_type = current_node.type

            if node_type in semantic_types:
                # Extract this semantic unit
                chunk_content = self._get_node_text(current_node, content)
                start_line = current_node.start_point[0] + 1
                end_line = current_node.end_point[0] + 1

                # Get name if available (for functions, classes, etc.)
                name = self._get_node_name(current_node, language)

                # Use smaller min_chunk_size for semantic units (functions, classes)
                effective_min = max(20, min_chunk_size // 5)

                # If chunk is too large, try to split it
                if len(chunk_content) > max_chunk_size:
                    # For large functions/classes, we might want to split
                    # For now, just include it as is
                    pass

                if len(chunk_content) >= effective_min:
                    chunks.append(CodeChunk(
                        content=chunk_content,
                        path=path,
                        start_line=start_line,
                        end_line=end_line,
                        language=language,
                        chunk_type=node_type.replace("_definition", "").replace("_declaration", "").replace("_item", ""),
                        name=name,
                    ))
            else:
                # Recursively visit children
                for child in current_node.children:
                    visit_node(child, node_type)

        visit_node(node)

        # If no semantic chunks found, fall back to the whole file
        if not chunks and content.strip():
            chunks.append(CodeChunk(
                content=content,
                path=path,
                start_line=1,
                end_line=content.count("\n") + 1,
                language=language,
                chunk_type="module",
            ))

        return chunks

    def _get_node_text(self, node, content: str) -> str:
        """Get the text content of a node."""
        start_byte = node.start_byte
        end_byte = node.end_byte
        return content[start_byte:end_byte]

    def _get_node_name(self, node, language: str) -> Optional[str]:
        """Extract the name of a function/class node."""
        # Look for identifier or name child
        for child in node.children:
            if child.type in ("identifier", "name", "property_identifier", "type_identifier"):
                return child.text.decode("utf-8")
            # For Python, the name is in a different child
            if child.type == "identifier":
                return child.text.decode("utf-8")

        # For some languages, name is in a specific position
        if language == "python":
            for child in node.children:
                if child.type == "identifier":
                    return child.text.decode("utf-8")

        return None