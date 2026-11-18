"""Optional second-stage ranking with a cross-encoder (e.g. BAAI/bge-reranker-v2-m3).

The embedding model encodes the question and each chunk SEPARATELY, then
compares two vectors: fast, but approximate. A cross-encoder reads the
question and ONE chunk together and outputs a relevance score: much more
precise, but too slow for the whole corpus. So the usual pattern is: the
vector (+ keyword) search proposes ~20-40 candidates, the reranker sorts them.

The weights (bge-reranker-v2-m3, about 2.3 GB) are NOT downloaded by the API
(HF_HUB_OFFLINE): download them once on purpose, then set RERANKER_MODEL.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Reranker(ABC):
    name: str

    @abstractmethod
    def score(self, question: str, texts: list[str]) -> list[float]:
        """One relevance score per text; higher = more relevant."""


class CrossEncoderReranker(Reranker):
    """sentence-transformers CrossEncoder, loaded on first use (CPU)."""

    def __init__(self, model_name: str, batch_size: int = 8) -> None:
        self.name = model_name
        self.batch_size = batch_size
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.name, device="cpu", max_length=512)
        return self._model

    def score(self, question: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        scores = self._load().predict(
            [(question, text) for text in texts], batch_size=self.batch_size
        )
        return [float(value) for value in scores]
