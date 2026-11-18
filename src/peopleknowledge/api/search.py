"""Search endpoints of the API, used by the test page (GET /).

- POST /api/search: a question -> the closest chunks in Qdrant;
- GET /api/gold: the gold questions, so that the page can test them in one click
  and show whether the expected (or forbidden) sources come back.

The search service (Qdrant client + embedding model) is created at the first
request, not at startup: loading bge-m3 takes ~20 s. Note that the LOCAL
embedded Qdrant folder can be opened by one program at a time: while the API
runs, scripts/index_chunks.py cannot open the index (stop the API first).
"""

from __future__ import annotations

import threading
import time
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from ..config import Settings, get_settings
from ..evaluation.gold import GoldQuestion, Source, UserContext, load_gold, source_of_chunk
from ..retrieval.embeddings import Embedder
from ..retrieval.profile_filter import ProfileVocabulary
from ..retrieval.reranker import CrossEncoderReranker
from ..retrieval.retriever import RetrievalOptions, Retriever

router = APIRouter(prefix="/api", tags=["search"])


class SearchService:
    """Everything a search needs, created once and shared by the requests."""

    def __init__(
        self,
        client: QdrantClient,
        embedder: Embedder,
        collection: str,
        retriever: Retriever | None = None,
    ) -> None:
        self.client = client
        self.embedder = embedder
        self.collection = collection
        self.retriever = retriever or Retriever(client, collection, embedder)
        self.lock = threading.Lock()  # the local Qdrant client and the model are not thread-safe

    @classmethod
    def from_settings(cls, settings: Settings) -> SearchService:
        settings.apply_hugging_face_environment()
        from ..retrieval.embeddings import SentenceTransformerEmbedder

        if settings.qdrant_url:
            client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        else:
            if not settings.qdrant_path.exists():
                raise RuntimeError("no Qdrant index yet: run scripts/index_chunks.py first.")
            client = QdrantClient(path=str(settings.qdrant_path))
        embedder = SentenceTransformerEmbedder(settings.embedding_model)
        retriever = Retriever(
            client,
            settings.qdrant_collection,
            embedder,
            options=RetrievalOptions(
                hybrid=settings.retrieval_hybrid,
                profile=settings.retrieval_profile,
                max_records=settings.retrieval_max_records,
                candidates=settings.retrieval_candidates,
            ),
            vocabulary=ProfileVocabulary.from_file(settings.company_profile_path)
            if settings.retrieval_profile
            else None,
            reranker=CrossEncoderReranker(settings.reranker_model)
            if settings.reranker_model
            else None,
        )
        return cls(client, embedder, settings.qdrant_collection, retriever)


_creation_lock = threading.Lock()


def get_search_service(
    request: Request, settings: Annotated[Settings, Depends(get_settings)]
) -> SearchService:
    with _creation_lock:
        service = getattr(request.app.state, "search_service", None)
        if service is None:
            try:
                service = SearchService.from_settings(settings)
            except RuntimeError as error:  # includes Qdrant's "folder already in use"
                raise HTTPException(status_code=503, detail=str(error)) from error
            request.app.state.search_service = service
        return service


class SearchRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)
    limit: int = Field(default=5, ge=1, le=20)
    include_archived: bool = False
    user_context: UserContext = Field(default_factory=UserContext)  # used by the profile option


class SearchResult(BaseModel):
    rank: int
    score: float
    chunk_id: str
    source: Source  # what the gold questions refer to: {"document_id": "HR-005", "section": "4.2"}
    kind: str
    title: str
    version: str
    lifecycle_status: str
    section_number: str | None
    pages: list[int]
    prefix: str
    content: str


class SearchResponse(BaseModel):
    question: str
    retrieval: str  # the pipeline used, e.g. "dense+bm25-rrf"
    include_archived: bool
    elapsed_ms: int
    results: list[SearchResult]


def to_result(rank: int, score: float, payload: dict[str, Any]) -> SearchResult:
    document = payload["document"]
    return SearchResult(
        rank=rank,
        score=round(score, 4),
        chunk_id=payload["chunk_id"],
        source=source_of_chunk(payload),
        kind=payload["kind"],
        title=document["title"],
        version=document["version"],
        lifecycle_status=document["lifecycle_status"],
        section_number=payload.get("section_number"),
        pages=payload.get("pages") or [],
        prefix=payload["prefix"],
        content=payload["content"],
    )


@router.post("/search")
def search_chunks(
    body: SearchRequest, service: Annotated[SearchService, Depends(get_search_service)]
) -> SearchResponse:
    started = time.perf_counter()
    try:
        with service.lock:
            points = service.retriever.retrieve(
                body.question,
                limit=body.limit,
                include_archived=body.include_archived,
                user_context=body.user_context.model_dump(exclude_none=True),
            )
    except OSError as error:  # model not in the Hugging Face cache (downloads are disabled)
        raise HTTPException(
            status_code=503,
            detail=(
                f"embedding model not available: {error}. Is HF_HOME set (e.g. D:\\huggingface)?"
            ),
        ) from error
    return SearchResponse(
        question=body.question,
        retrieval=service.retriever.describe(),
        include_archived=body.include_archived,
        elapsed_ms=int((time.perf_counter() - started) * 1000),
        results=[to_result(rank, p.score, p.payload or {}) for rank, p in enumerate(points, 1)],
    )


@router.get("/gold")
def gold_questions(settings: Annotated[Settings, Depends(get_settings)]) -> list[GoldQuestion]:
    if not settings.gold_path.exists():
        return []
    try:
        return load_gold(settings.gold_path)
    except ValueError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
