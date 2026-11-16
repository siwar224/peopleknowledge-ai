"""Embedding stage: turn texts (chunks, questions) into vectors."""

from .base import Embedder
from .sentence_transformer import SentenceTransformerEmbedder

__all__ = ["Embedder", "SentenceTransformerEmbedder"]
