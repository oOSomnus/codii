"""Embedding package for codii."""

from .embedder import Embedder, get_embedder
from .cross_encoder import CrossEncoderWrapper, get_cross_encoder

__all__ = ["Embedder", "get_embedder", "CrossEncoderWrapper", "get_cross_encoder"]