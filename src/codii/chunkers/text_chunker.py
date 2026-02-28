"""Text-based fallback chunker."""

from typing import List
from dataclasses import dataclass
import re


@dataclass
class CodeChunk:
    """A chunk of code with metadata."""
    content: str
    path: str
    start_line: int
    end_line: int
    language: str
    chunk_type: str
    name: str = None

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


class TextChunker:
    """Text-based code chunker that splits by lines and paragraphs."""

    def __init__(
        self,
        max_chunk_size: int = 1500,
        min_chunk_size: int = 100,
        chunk_overlap: int = 200,
    ):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_file(
        self,
        content: str,
        path: str,
        language: str,
    ) -> List[CodeChunk]:
        """
        Chunk a file using text-based splitting.

        Tries to split at natural boundaries (blank lines) when possible.
        """
        chunks = []

        if not content.strip():
            return chunks

        lines = content.split("\n")
        current_chunk_lines = []
        current_start_line = 1
        current_size = 0

        for i, line in enumerate(lines, start=1):
            line_size = len(line) + 1  # +1 for newline

            # Check if adding this line would exceed max size
            if current_size + line_size > self.max_chunk_size and current_chunk_lines:
                # Save current chunk
                chunk_content = "\n".join(current_chunk_lines)
                if len(chunk_content) >= self.min_chunk_size:
                    chunks.append(CodeChunk(
                        content=chunk_content,
                        path=path,
                        start_line=current_start_line,
                        end_line=i - 1,
                        language=language,
                        chunk_type="text_block",
                    ))

                # Start new chunk with overlap
                overlap_lines = self._get_overlap_lines(current_chunk_lines)
                current_chunk_lines = overlap_lines
                current_start_line = i - len(overlap_lines)
                current_size = sum(len(l) + 1 for l in overlap_lines)

            current_chunk_lines.append(line)
            current_size += line_size

        # Don't forget the last chunk
        if current_chunk_lines:
            chunk_content = "\n".join(current_chunk_lines)
            if len(chunk_content) >= self.min_chunk_size:
                chunks.append(CodeChunk(
                    content=chunk_content,
                    path=path,
                    start_line=current_start_line,
                    end_line=len(lines),
                    language=language,
                    chunk_type="text_block",
                ))

        # If no chunks were created but content exists, create one big chunk
        if not chunks and content.strip():
            chunks.append(CodeChunk(
                content=content,
                path=path,
                start_line=1,
                end_line=len(lines),
                language=language,
                chunk_type="module",
            ))

        return chunks

    def _get_overlap_lines(self, lines: List[str]) -> List[str]:
        """Get overlap lines for the next chunk."""
        if not lines:
            return []

        overlap_size = 0
        overlap_lines = []

        # Take lines from the end until we reach overlap size
        for line in reversed(lines):
            if overlap_size + len(line) > self.chunk_overlap:
                break
            overlap_lines.insert(0, line)
            overlap_size += len(line) + 1

        return overlap_lines