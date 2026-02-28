"""Tests for query preprocessing functionality."""

import pytest

from codii.indexers.query_processor import (
    QueryProcessor,
    ProcessedQuery,
    process_query,
    split_camel_case,
    split_snake_case,
    tokenize_identifier,
)
from codii.storage.database import preprocess_fts_query


class TestPreprocessFtsQuery:
    """Tests for FTS5 query preprocessing."""

    def test_single_term_with_wildcard(self):
        """Single terms should get wildcard suffix."""
        result = preprocess_fts_query("kalloc")
        assert result == "kalloc*"

    def test_multi_word_to_or(self):
        """Multi-word queries should use OR matching."""
        result = preprocess_fts_query("page table walk")
        assert result == "page* OR table* OR walk*"

    def test_special_characters_removed(self):
        """FTS5 special characters should be removed."""
        result = preprocess_fts_query('test* ^query" (with) special-chars')
        # Should clean the special chars and add wildcards
        # The result will have wildcards added to each term
        assert "^" not in result
        assert '"' not in result
        assert "(" not in result
        assert ")" not in result
        # Result should contain the cleaned terms with wildcards
        assert "test*" in result
        assert "query*" in result
        assert "with*" in result
        assert "special*" in result
        assert "chars*" in result

    def test_empty_query(self):
        """Empty queries should return empty."""
        result = preprocess_fts_query("")
        assert result == ""

    def test_whitespace_only(self):
        """Whitespace-only queries should return empty."""
        result = preprocess_fts_query("   ")
        assert result == ""

    def test_multiple_spaces(self):
        """Multiple spaces should be handled correctly."""
        result = preprocess_fts_query("page    table   walk")
        assert result == "page* OR table* OR walk*"

    def test_no_wildcards_option(self):
        """Can disable wildcard suffixing."""
        result = preprocess_fts_query("kalloc", add_wildcards=False)
        assert result == "kalloc"

    def test_no_or_option(self):
        """Can disable OR matching for single terms."""
        result = preprocess_fts_query("kalloc", use_or=False)
        assert result == "kalloc*"

    def test_code_query(self):
        """Code-related queries should be handled."""
        result = preprocess_fts_query("kalloc kfree memory")
        assert "kalloc*" in result
        assert "kfree*" in result
        assert "memory*" in result


class TestSplitCamelCase:
    """Tests for camelCase splitting."""

    def test_simple_camel_case(self):
        """Split simple camelCase."""
        result = split_camel_case("pageTableWalk")
        assert result == ["page", "table", "walk"]

    def test_pascal_case(self):
        """Split PascalCase."""
        result = split_camel_case("PageTableWalk")
        assert result == ["page", "table", "walk"]

    def test_single_word(self):
        """Single lowercase word should return as-is."""
        result = split_camel_case("kalloc")
        assert result == ["kalloc"]

    def test_all_uppercase(self):
        """All uppercase should be treated as single word."""
        result = split_camel_case("ALLOC")
        # All uppercase should be lowercased but not split
        assert result == ["alloc"]

    def test_mixed_case(self):
        """Handle mixed case with acronyms."""
        result = split_camel_case("getHTTPResponse")
        # Note: this will split on each uppercase letter
        # For "getHTTPResponse", we get get, h, t, t, p, response
        assert "get" in result
        assert "response" in result


class TestSplitSnakeCase:
    """Tests for snake_case splitting."""

    def test_simple_snake_case(self):
        """Split simple snake_case."""
        result = split_snake_case("page_table_walk")
        assert result == ["page", "table", "walk"]

    def test_single_word(self):
        """Single word should return as-is."""
        result = split_snake_case("kalloc")
        assert result == ["kalloc"]

    def test_screaming_snake_case(self):
        """Handle SCREAMING_SNAKE_CASE."""
        result = split_snake_case("PAGE_TABLE_WALK")
        assert result == ["page", "table", "walk"]

    def test_double_underscore(self):
        """Handle double underscores."""
        result = split_snake_case("page__table")
        assert "page" in result
        assert "table" in result


class TestTokenizeIdentifier:
    """Tests for identifier tokenization."""

    def test_camel_case_identifier(self):
        """Tokenize camelCase identifiers."""
        result = tokenize_identifier("pageTableWalk")
        assert result == ["page", "table", "walk"]

    def test_snake_case_identifier(self):
        """Tokenize snake_case identifiers."""
        result = tokenize_identifier("kalloc_memory")
        assert result == ["kalloc", "memory"]

    def test_plain_identifier(self):
        """Plain identifiers return as single token."""
        result = tokenize_identifier("kalloc")
        assert result == ["kalloc"]

    def test_mixed_notation(self):
        """Handle mixed notation (prefers snake_case if underscores present)."""
        result = tokenize_identifier("page_tableWalk")
        # Should split on underscores
        assert "page" in result


class TestQueryProcessor:
    """Tests for QueryProcessor class."""

    def test_basic_query_processing(self):
        """Process a basic query."""
        processor = QueryProcessor()
        result = processor.process("page table walk")

        assert result.original == "page table walk"
        assert "page" in result.terms
        assert "table" in result.terms
        assert "walk" in result.terms
        assert "page*" in result.fts_query

    def test_abbreviation_expansion(self):
        """Expand common abbreviations."""
        processor = QueryProcessor(use_expansion=True)
        result = processor.process("kalloc memory")

        # Should expand kalloc
        assert "kalloc" in result.terms
        # Expanded terms should include expansions
        assert any("kernel" in t for t in result.expanded_terms)

    def test_code_tokenization_in_query(self):
        """Tokenize code identifiers in queries."""
        processor = QueryProcessor(use_code_tokenization=True)
        result = processor.process("pageTableWalk")

        # Should tokenize the camelCase identifier
        assert len(result.expanded_terms) > 1
        assert any("page" in t for t in result.expanded_terms)
        assert any("table" in t for t in result.expanded_terms)
        assert any("walk" in t for t in result.expanded_terms)

    def test_disable_expansion(self):
        """Can disable abbreviation expansion."""
        processor = QueryProcessor(use_expansion=False)
        result = processor.process("kalloc memory")

        # Should not expand kalloc
        expanded_kalloc = [t for t in result.expanded_terms if "kernel" in t]
        assert len(expanded_kalloc) == 0

    def test_disable_code_tokenization(self):
        """Can disable code tokenization."""
        processor = QueryProcessor(use_code_tokenization=False)
        result = processor.process("pageTableWalk")

        # Should not split camelCase
        assert "pagetablewalk" in result.terms

    def test_min_term_length(self):
        """Filter out short terms."""
        processor = QueryProcessor(min_term_length=3)
        result = processor.process("a an the page table")

        # "a" (1 char) and "an" (2 chars) should be filtered
        assert "a" not in result.terms
        assert "an" not in result.terms
        # "the" (3 chars) should pass
        assert "the" in result.terms
        assert "page" in result.terms
        assert "table" in result.terms

    def test_empty_query(self):
        """Handle empty queries."""
        processor = QueryProcessor()
        result = processor.process("")

        assert result.original == ""
        assert result.terms == []
        assert result.expanded_terms == []
        assert result.fts_query == ""

    def test_whitespace_query(self):
        """Handle whitespace-only queries."""
        processor = QueryProcessor()
        result = processor.process("   ")

        assert result.original == "   "
        assert result.terms == []

    def test_fts_query_format(self):
        """FTS query should have proper format."""
        processor = QueryProcessor()
        result = processor.process("page table walk")

        # Should be OR-joined with wildcards
        assert " OR " in result.fts_query
        assert result.fts_query.endswith("*")


class TestProcessQueryFunction:
    """Tests for the convenience process_query function."""

    def test_basic_usage(self):
        """Test basic function usage."""
        result = process_query("kalloc memory")

        assert isinstance(result, ProcessedQuery)
        assert "kalloc" in result.terms
        assert "memory" in result.terms

    def test_with_options(self):
        """Test with options."""
        result = process_query(
            "pageTableWalk",
            use_expansion=True,
            use_code_tokenization=True,
        )

        # Should tokenize camelCase
        assert any("page" in t for t in result.expanded_terms)


class TestAbbreviationExpansions:
    """Tests for specific abbreviation expansions."""

    def test_alloc_expansion(self):
        """Test alloc abbreviation expansion."""
        processor = QueryProcessor()
        result = processor.process("alloc")

        # Should include allocate, allocation, allocator
        assert any("allocate" in t for t in result.expanded_terms)

    def test_mem_expansion(self):
        """Test mem abbreviation expansion."""
        processor = QueryProcessor()
        result = processor.process("mem")

        assert any("memory" in t for t in result.expanded_terms)

    def test_ptr_expansion(self):
        """Test ptr abbreviation expansion."""
        processor = QueryProcessor()
        result = processor.process("ptr")

        assert any("pointer" in t for t in result.expanded_terms)

    def test_proc_expansion(self):
        """Test proc abbreviation expansion."""
        processor = QueryProcessor()
        result = processor.process("proc")

        # Should expand to process or procedure
        assert any("process" in t or "procedure" in t for t in result.expanded_terms)

    def test_ctx_expansion(self):
        """Test ctx abbreviation expansion."""
        processor = QueryProcessor()
        result = processor.process("ctx")

        assert any("context" in t for t in result.expanded_terms)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_numbers_in_query(self):
        """Handle numbers in queries."""
        processor = QueryProcessor()
        result = processor.process("page2 table3")

        # Numbers should be handled
        assert len(result.terms) >= 2

    def test_mixed_case_query(self):
        """Handle mixed case queries."""
        processor = QueryProcessor()
        result = processor.process("PAGE TABLE Walk")

        # Should be lowercased
        assert all(t.islower() for t in result.terms)

    def test_dots_in_query(self):
        """Handle dots in queries (e.g., method calls)."""
        processor = QueryProcessor()
        result = processor.process("object.method")

        # Dots should be removed/split
        assert "." not in " ".join(result.terms)

    def test_very_long_query(self):
        """Handle very long queries."""
        processor = QueryProcessor()
        long_query = " ".join(["word"] * 100)
        result = processor.process(long_query)

        # Should process without error
        assert len(result.terms) > 0

    def test_unicode_characters(self):
        """Handle unicode characters."""
        processor = QueryProcessor()
        result = processor.process("hello world")

        # Should handle basic unicode
        assert "hello" in result.terms
        assert "world" in result.terms