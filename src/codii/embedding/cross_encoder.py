"""Cross-encoder wrapper for re-ranking search results."""

import threading
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..indexers.hybrid_search import SearchResult


class CrossEncoderWrapper:
    """Wrapper for sentence-transformers CrossEncoder model.

    Uses singleton pattern to ensure only one model instance is loaded.
    Provides re-ranking functionality for search results.
    """

    _instance: Optional["CrossEncoderWrapper"] = None
    _lock = threading.Lock()

    def __new__(cls, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Singleton pattern to ensure only one model instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        if self._initialized:
            return

        self.model_name = model_name
        self._model = None
        self._initialized = True

    @property
    def model(self):
        """Lazy load the cross-encoder model."""
        if self._model is None:
            import sys
            print(f"Loading cross-encoder model: {self.model_name}...", file=sys.stderr)
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            print("Cross-encoder model loaded.", file=sys.stderr)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: List["SearchResult"],
        top_k: int = 10,
        threshold: float = 0.5,
    ) -> List["SearchResult"]:
        """Re-rank candidates using cross-encoder scoring.

        Args:
            query: The search query
            candidates: List of SearchResult objects to re-rank
            top_k: Maximum number of results to return
            threshold: Minimum score threshold (normalized 0-1).
                      Results below this threshold are filtered out.

        Returns:
            Re-ranked list of SearchResult objects with rerank_score set.
            Only results with rerank_score >= threshold are returned.
        """
        if not candidates:
            return []

        # Create (query, content) pairs for scoring
        pairs = [(query, candidate.content) for candidate in candidates]

        # Get raw logits from cross-encoder
        scores = self.model.predict(pairs)

        # Normalize scores using sigmoid: score / (1 + exp(-score)) -> 0-1 range
        # This converts logits to probabilities
        import math
        normalized_scores = [1 / (1 + math.exp(-s)) for s in scores]

        # Add scores to candidates and sort
        scored_candidates = list(zip(candidates, normalized_scores))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # Filter by threshold and limit to top_k
        results = []
        for candidate, score in scored_candidates:
            if score >= threshold:
                candidate.rerank_score = score
                results.append(candidate)

            if len(results) >= top_k:
                break

        return results


def get_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> CrossEncoderWrapper:
    """Get the singleton cross-encoder instance."""
    return CrossEncoderWrapper(model_name)