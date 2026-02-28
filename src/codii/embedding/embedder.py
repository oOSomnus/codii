"""Embedding utilities for codii."""

import numpy as np
from typing import List, Optional
import threading


class Embedder:
    """Wrapper for sentence-transformers embedding model."""

    _instance: Optional["Embedder"] = None
    _lock = threading.Lock()

    def __new__(cls, model_name: str = "all-MiniLM-L6-v2"):
        """Singleton pattern to ensure only one model instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if self._initialized:
            return

        self.model_name = model_name
        self._model = None
        self._embedding_dim = None
        self._initialized = True

    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            import sys
            print(f"Loading embedding model: {self.model_name}...", file=sys.stderr)
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._embedding_dim = self._model.get_sentence_embedding_dimension()
            print(f"Model loaded. Embedding dimension: {self._embedding_dim}", file=sys.stderr)
        return self._model

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        if self._embedding_dim is None:
            _ = self.model
        return self._embedding_dim

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts.

        Args:
            texts: List of strings to embed

        Returns:
            Numpy array of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])

        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text."""
        return self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)


def get_embedder(model_name: str = "all-MiniLM-L6-v2") -> Embedder:
    """Get the singleton embedder instance."""
    return Embedder(model_name)