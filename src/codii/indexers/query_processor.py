"""Query preprocessing for better search recall."""

import re
from typing import List, Set, Optional
from dataclasses import dataclass


@dataclass
class ProcessedQuery:
    """A processed query with multiple variations."""
    original: str
    terms: List[str]
    expanded_terms: List[str]
    fts_query: str


def split_camel_case(text: str) -> List[str]:
    """Split camelCase and PascalCase identifiers.

    Examples:
        >>> split_camel_case("pageTableWalk")
        ['page', 'table', 'walk']
        >>> split_camel_case("KallocMemory")
        ['kalloc', 'memory']
    """
    # If all uppercase, just lowercase it
    if text.isupper():
        return [text.lower()]

    # Insert spaces before uppercase letters (except at start)
    result = re.sub(r'(?<!^)(?=[A-Z])', ' ', text)
    return [t.lower() for t in result.split() if t]


def split_snake_case(text: str) -> List[str]:
    """Split snake_case identifiers.

    Examples:
        >>> split_snake_case("page_table_walk")
        ['page', 'table', 'walk']
        >>> split_snake_case("kalloc_memory")
        ['kalloc', 'memory']
    """
    return [t.lower() for t in text.split('_') if t]


def tokenize_identifier(identifier: str) -> List[str]:
    """Tokenize a code identifier into constituent words.

    Handles camelCase, PascalCase, snake_case, and SCREAMING_SNAKE_CASE.

    Examples:
        >>> tokenize_identifier("pageTableWalk")
        ['page', 'table', 'walk']
        >>> tokenize_identifier("kalloc_memory")
        ['kalloc', 'memory']
        >>> tokenize_identifier("PAGE_TABLE_WALK")
        ['page', 'table', 'walk']
    """
    # First try snake_case
    if '_' in identifier:
        return split_snake_case(identifier)

    # Then try camelCase/PascalCase
    if any(c.isupper() for c in identifier[1:] if len(identifier) > 1):
        return split_camel_case(identifier)

    # Return as-is (lowercased)
    return [identifier.lower()]


class QueryProcessor:
    """Process search queries for better recall in code search."""

    # Common code abbreviations and their expansions
    ABBREVIATIONS = {
        'alloc': ['allocate', 'allocation', 'allocator'],
        'kalloc': ['kernel_allocate', 'kernel_allocation'],
        'kfree': ['kernel_free', 'free'],
        'mem': ['memory'],
        'ptr': ['pointer'],
        'fn': ['function'],
        'func': ['function'],
        'proc': ['process', 'procedure'],
        'buf': ['buffer'],
        'cfg': ['config', 'configuration'],
        'ctx': ['context'],
        'init': ['initialize', 'initialization'],
        'sync': ['synchronize', 'synchronization'],
        'async': ['asynchronous'],
        'impl': ['implementation', 'implement'],
        'msg': ['message'],
        'err': ['error'],
        'val': ['value'],
        'idx': ['index'],
        'len': ['length'],
        'num': ['number'],
        'str': ['string'],
        'char': ['character'],
        'tmp': ['temporary'],
        'temp': ['temporary'],
        'info': ['information'],
        'desc': ['description', 'descriptor'],
        'def': ['definition', 'default'],
        'ref': ['reference'],
        'src': ['source'],
        'dst': ['destination'],
        'prev': ['previous'],
        'cur': ['current'],
        'max': ['maximum'],
        'min': ['minimum'],
        'avg': ['average'],
        'dev': ['device', 'development'],
        'env': ['environment'],
        'arg': ['argument'],
        'param': ['parameter'],
        'ret': ['return'],
        'res': ['result', 'response', 'resource'],
        'req': ['request', 'requirement'],
        'resp': ['response'],
        'ack': ['acknowledge'],
        'nack': ['not_acknowledge'],
        'irq': ['interrupt', 'interrupt_request'],
        'pid': ['process_id', 'process_identifier'],
        'tid': ['thread_id', 'thread_identifier'],
        'fd': ['file_descriptor'],
        'io': ['input_output'],
        'cpu': ['processor', 'central_processing_unit'],
        'gpu': ['graphics_processing_unit'],
        'ram': ['random_access_memory', 'memory'],
        'rom': ['read_only_memory'],
        'tlb': ['translation_lookaside_buffer'],
        'mmu': ['memory_management_unit'],
        'pfn': ['page_frame_number'],
        'va': ['virtual_address'],
        'pa': ['physical_address'],
    }

    def __init__(
        self,
        use_expansion: bool = True,
        use_code_tokenization: bool = True,
        min_term_length: int = 2,
    ):
        """
        Initialize the query processor.

        Args:
            use_expansion: Whether to expand abbreviations
            use_code_tokenization: Whether to tokenize code identifiers
            min_term_length: Minimum term length to include
        """
        self.use_expansion = use_expansion
        self.use_code_tokenization = use_code_tokenization
        self.min_term_length = min_term_length

    def process(self, query: str) -> ProcessedQuery:
        """
        Process a search query for better recall.

        Args:
            query: The raw search query

        Returns:
            ProcessedQuery with original, terms, expanded terms, and FTS query
        """
        if not query or not query.strip():
            return ProcessedQuery(
                original=query,
                terms=[],
                expanded_terms=[],
                fts_query=query,
            )

        # Clean and tokenize
        cleaned = self._clean_query(query)
        raw_terms = cleaned.split()

        # Process each term
        terms: List[str] = []
        expanded_terms: List[str] = []

        for term in raw_terms:
            term_lower = term.lower().strip()
            if len(term_lower) < self.min_term_length:
                continue

            terms.append(term_lower)
            expanded_terms.append(term_lower)

            # Code tokenization (camelCase, snake_case)
            # Use the original term (before lowercasing) for detection
            if self.use_code_tokenization:
                tokenized = tokenize_identifier(term)
                if len(tokenized) > 1:
                    expanded_terms.extend(tokenized)

            # Abbreviation expansion
            if self.use_expansion:
                expansions = self._expand_abbreviation(term_lower)
                expanded_terms.extend(expansions)

        # Remove duplicates while preserving order
        expanded_terms = list(dict.fromkeys(expanded_terms))

        # Build FTS query
        fts_query = self._build_fts_query(expanded_terms)

        return ProcessedQuery(
            original=query,
            terms=terms,
            expanded_terms=expanded_terms,
            fts_query=fts_query,
        )

    def _clean_query(self, query: str) -> str:
        """Clean the query by removing special characters."""
        # Remove FTS5 special characters
        cleaned = re.sub(r'[*^"()\-|]', ' ', query)
        # Remove other non-alphanumeric except spaces and underscores
        cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
        # Normalize whitespace
        cleaned = ' '.join(cleaned.split())
        return cleaned

    def _expand_abbreviation(self, term: str) -> List[str]:
        """Expand abbreviations to their full forms.

        Args:
            term: The term to expand

        Returns:
            List of expanded terms
        """
        term_lower = term.lower()
        return self.ABBREVIATIONS.get(term_lower, [])

    def _build_fts_query(self, terms: List[str]) -> str:
        """Build an FTS5 query from processed terms.

        Uses OR matching with wildcards for better recall.

        Args:
            terms: List of processed terms

        Returns:
            FTS5 query string
        """
        if not terms:
            return ""

        # Add wildcards and join with OR
        wildcard_terms = [f"{t}*" for t in terms]
        return ' OR '.join(wildcard_terms)


def process_query(
    query: str,
    use_expansion: bool = True,
    use_code_tokenization: bool = True,
) -> ProcessedQuery:
    """
    Convenience function to process a query.

    Args:
        query: The search query
        use_expansion: Whether to expand abbreviations
        use_code_tokenization: Whether to tokenize code identifiers

    Returns:
        ProcessedQuery object
    """
    processor = QueryProcessor(
        use_expansion=use_expansion,
        use_code_tokenization=use_code_tokenization,
    )
    return processor.process(query)