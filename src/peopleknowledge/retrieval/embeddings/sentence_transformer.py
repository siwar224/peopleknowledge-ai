"""Embedding with a local Hugging Face model through sentence-transformers.

The model is downloaded once to the Hugging Face cache (``HF_HOME``, set to
``D:\\huggingface`` on the owner's machine), then read from disk: nothing is
sent outside. It is loaded LAZILY, at the first encoding, because loading
bge-m3 takes several seconds and a few GB of memory.
"""

from __future__ import annotations

from typing import Any

from .base import Embedder, Vector

DEFAULT_MODEL = "BAAI/bge-m3"


class SentenceTransformerEmbedder(Embedder):
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        batch_size: int = 16,
        dimension: int | None = None,
    ) -> None:
        """``dimension`` may be given to avoid loading the model just to know it
        (1024 for bge-m3); otherwise it is read from the model."""

        self.name = model_name
        self.device = device
        self.batch_size = batch_size
        self._dimension = dimension
        self._model: Any = None

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # heavy import (torch)

            self._model = SentenceTransformer(self.name, device=self.device)
        return self._model

    @property
    def dimension(self) -> int:  # type: ignore[override]
        if self._dimension is None:
            # renamed in sentence-transformers 5.x; the old name is kept for older versions
            model = self.model
            getter = getattr(model, "get_embedding_dimension", None)
            if getter is None:
                getter = model.get_sentence_embedding_dimension
            self._dimension = int(getter())
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[Vector]:
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,  # length 1: dot product = cosine similarity
            show_progress_bar=len(texts) > self.batch_size,
        )
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> Vector:
        return self.embed_documents([text])[0]
