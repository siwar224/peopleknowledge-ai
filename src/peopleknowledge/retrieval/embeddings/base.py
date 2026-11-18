"""Common interface of all embedding models.

The rest of the pipeline (indexing, search) only knows this interface, never a
particular library. A second implementation (llama.cpp server, another model)
can then be added and compared with the gold questions without touching the
indexing or search code.

Two methods, because some models encode a question and a passage differently
(e.g. multilingual-e5 needs "query: " / "passage: " prefixes). bge-m3 does
not, but the pipeline must not depend on that.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

Vector = list[float]


class Embedder(ABC):
    name: str  # model id, stored with every vector: vectors of two models are not comparable
    dimension: int  # length of every vector

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[Vector]:
        """Vectors of passages (chunks), in the same order. Normalized to length 1."""

    @abstractmethod
    def embed_query(self, text: str) -> Vector:
        """Vector of a question. Normalized to length 1."""
