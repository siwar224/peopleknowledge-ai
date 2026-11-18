"""The retrieval pipeline used by the API and the scripts: one place, every option.

    question
      -> vector search (Qdrant, bge-m3)          always
      -> + keyword search (BM25), fused by RRF   option ``hybrid``
      -> reranked by a cross-encoder             option ``reranker``
      -> profile rules (rows about others last)  option ``profile``
      -> the first ``limit`` hits

Each stage is an OPTION so that its effect can be measured alone with
scripts/evaluate_retrieval.py (one change at a time, against the baseline).
With every option off, the result is exactly the former ``qdrant_index.search``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qdrant_client import QdrantClient

from .embeddings.base import Embedder
from .keyword_index import KeywordIndex, reciprocal_rank_fusion
from .profile_filter import ProfileVocabulary, apply_profile
from .qdrant_index import search
from .reranker import Reranker


@dataclass
class Hit:
    """A ranked chunk: same ``score`` / ``payload`` attributes as a Qdrant ScoredPoint."""

    score: float
    payload: dict[str, Any]


@dataclass
class RetrievalOptions:
    hybrid: bool = False
    profile: bool = False
    max_records: int | None = None  # cap on table rows among the results (profile option)
    candidates: int = 30  # pool size before fusion / reranking / profile rules

    def describe(self) -> str:
        parts = ["dense"]
        if self.hybrid:
            parts.append("bm25-rrf")
        if self.profile:
            parts.append(f"profile(max_records={self.max_records})")
        return "+".join(parts)


def _all_payloads(client: QdrantClient, collection: str) -> list[dict[str, Any]]:
    payloads, offset = [], None
    while True:
        points, offset = client.scroll(
            collection, limit=256, offset=offset, with_payload=True, with_vectors=False
        )
        payloads.extend(point.payload or {} for point in points)
        if offset is None:
            return payloads


@dataclass
class Retriever:
    client: QdrantClient
    collection: str
    embedder: Embedder
    options: RetrievalOptions = field(default_factory=RetrievalOptions)
    vocabulary: ProfileVocabulary | None = None
    reranker: Reranker | None = None
    _keyword_index: KeywordIndex | None = None

    @property
    def keyword_index(self) -> KeywordIndex:
        if self._keyword_index is None:  # built at first use, from the indexed payloads
            self._keyword_index = KeywordIndex(_all_payloads(self.client, self.collection))
        return self._keyword_index

    def describe(self) -> str:
        text = self.options.describe()
        return text + (f"+rerank({self.reranker.name})" if self.reranker else "")

    def retrieve(
        self,
        question: str,
        limit: int = 5,
        include_archived: bool = False,
        user_context: dict[str, Any] | None = None,
    ) -> list[Hit]:
        options = self.options
        staged = options.hybrid or options.profile or self.reranker is not None
        pool = max(limit, options.candidates) if staged else limit

        dense = [
            Hit(point.score, point.payload or {})
            for point in search(
                self.client,
                self.collection,
                self.embedder,
                question,
                limit=pool,
                include_archived=include_archived,
            )
        ]
        hits = dense
        if options.hybrid:
            keyword = [
                Hit(score, payload)
                for score, payload in self.keyword_index.search(question, pool, include_archived)
            ]
            by_id = {hit.payload["chunk_id"]: hit.payload for hit in [*keyword, *dense]}
            fused = reciprocal_rank_fusion(
                [[h.payload["chunk_id"] for h in dense], [h.payload["chunk_id"] for h in keyword]]
            )
            hits = [
                Hit(score, by_id[chunk_id])
                for chunk_id, score in sorted(fused.items(), key=lambda item: -item[1])
            ][:pool]
        if self.reranker is not None:
            scores = self.reranker.score(question, [hit.payload.get("text", "") for hit in hits])
            hits = sorted(
                (Hit(score, hit.payload) for score, hit in zip(scores, hits, strict=True)),
                key=lambda hit: -hit.score,
            )
        if options.profile:
            hits = apply_profile(hits, user_context, self.vocabulary, options.max_records)
        return hits[:limit]
