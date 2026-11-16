"""End-to-end RAG answer endpoint and profile options for the web interface."""

from __future__ import annotations

import threading
import time
from datetime import date
from typing import Annotated

import yaml
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, ValidationError

from .config import Settings, get_settings
from .rag.chunking.models import Chunk
from .rag.evaluation.gold import Source, UserContext
from .rag.generation import (
    ContextSource,
    GenerationError,
    Generator,
    OpenAICompatibleGenerator,
    RetrievedChunk,
    build_context,
    build_prompt,
)
from .rag.indexing.qdrant_index import search
from .search_api import SearchResult, SearchService, get_search_service, to_result

router = APIRouter(prefix="/api", tags=["answers"])


class GenerationService:
    """One generator shared by requests, with serialized local model access."""

    def __init__(self, generator: Generator) -> None:
        self.generator = generator
        self.lock = threading.Lock()

    @classmethod
    def from_settings(cls, settings: Settings) -> GenerationService:
        return cls(
            OpenAICompatibleGenerator(
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key,
                model=settings.llm_model,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
                timeout_seconds=settings.llm_timeout_seconds,
            )
        )


_generation_creation_lock = threading.Lock()


def get_generation_service(
    request: Request, settings: Annotated[Settings, Depends(get_settings)]
) -> GenerationService:
    with _generation_creation_lock:
        service = getattr(request.app.state, "generation_service", None)
        if service is None:
            service = GenerationService.from_settings(settings)
            request.app.state.generation_service = service
        return service


class AnswerRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)
    user_context: UserContext = Field(default_factory=UserContext)
    reference_date: date = Field(default_factory=date.today)
    limit: int = Field(default=5, ge=1, le=20)
    include_archived: bool = False


class AnswerCitation(BaseModel):
    source_id: str
    rank: int
    score: float
    chunk_id: str
    source: Source
    title: str
    version: str
    lifecycle_status: str
    pages: list[int]


class AnswerTiming(BaseModel):
    retrieval_ms: int
    generation_ms: int
    total_ms: int


class AnswerResponse(BaseModel):
    question: str
    status: str
    answer: str
    citations: list[AnswerCitation]
    model: str
    include_archived: bool
    reference_date: date
    retrieved_count: int
    timing: AnswerTiming
    retrieval_results: list[SearchResult]


def _chunk_from_payload(payload: dict) -> Chunk:
    chunk_payload = dict(payload)
    chunk_payload.pop("embedding_model", None)
    chunk_payload.pop("text_hash", None)
    return Chunk.model_validate(chunk_payload)


def _citation(source_id: str, sources: dict[str, ContextSource]) -> AnswerCitation:
    context_source = sources[source_id]
    return AnswerCitation(
        source_id=context_source.source_id,
        rank=context_source.rank,
        score=round(context_source.score, 4),
        chunk_id=context_source.chunk_id,
        source=context_source.source,
        title=context_source.title,
        version=context_source.version,
        lifecycle_status=context_source.lifecycle_status,
        pages=context_source.pages,
    )


@router.post("/answer")
def answer_question(
    body: AnswerRequest,
    search_service: Annotated[SearchService, Depends(get_search_service)],
    generation_service: Annotated[GenerationService, Depends(get_generation_service)],
) -> AnswerResponse:
    """Retrieve evidence, build the prompt, generate, and resolve trusted citations."""

    started = time.perf_counter()
    retrieval_started = time.perf_counter()
    try:
        with search_service.lock:
            points = search(
                search_service.client,
                search_service.collection,
                search_service.embedder,
                body.question,
                limit=body.limit,
                include_archived=body.include_archived,
            )
    except OSError as error:
        raise HTTPException(
            status_code=503,
            detail=(
                f"embedding model not available: {error}. "
                "Is HF_HOME set (e.g. D:\\huggingface)?"
            ),
        ) from error
    retrieval_ms = int((time.perf_counter() - retrieval_started) * 1000)

    try:
        retrieved = [
            RetrievedChunk(
                rank=rank,
                score=point.score,
                chunk=_chunk_from_payload(point.payload or {}),
            )
            for rank, point in enumerate(points, 1)
        ]
    except (KeyError, ValidationError) as error:
        raise HTTPException(
            status_code=500, detail=f"invalid chunk in the index: {error}"
        ) from error

    context = build_context(
        retrieved,
        max_chunks=body.limit,
        include_archived=body.include_archived,
    )
    prompt = build_prompt(
        body.question,
        context,
        reference_date=body.reference_date,
        user_context=body.user_context,
    )
    try:
        with generation_service.lock:
            result = generation_service.generator.generate(prompt)
    except GenerationError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error

    source_map = {source.source_id: source for source in context.sources}
    citations = [_citation(source_id, source_map) for source_id in result.answer.citations]
    retrieval_results = [
        to_result(rank, point.score, point.payload or {})
        for rank, point in enumerate(points, 1)
    ]
    return AnswerResponse(
        question=body.question,
        status=result.answer.status,
        answer=result.answer.answer,
        citations=citations,
        model=result.model,
        include_archived=body.include_archived,
        reference_date=body.reference_date,
        retrieved_count=len(points),
        timing=AnswerTiming(
            retrieval_ms=retrieval_ms,
            generation_ms=result.elapsed_ms,
            total_ms=int((time.perf_counter() - started) * 1000),
        ),
        retrieval_results=retrieval_results,
    )


class ProfileOption(BaseModel):
    id: str
    label: str
    department: str | None = None


class ProfileOptionsResponse(BaseModel):
    departments: list[ProfileOption]
    job_families: list[ProfileOption]
    employment_types: list[ProfileOption]


@router.get("/profile-options")
def profile_options(
    settings: Annotated[Settings, Depends(get_settings)],
) -> ProfileOptionsResponse:
    """Return stable profile identifiers without duplicating them in the HTML."""

    try:
        data = yaml.safe_load(settings.company_profile_path.read_text(encoding="utf-8"))
        return ProfileOptionsResponse(
            departments=data["departments"],
            job_families=data["job_families"],
            employment_types=data["employment_types"],
        )
    except (OSError, KeyError, TypeError, ValidationError, yaml.YAMLError) as error:
        raise HTTPException(status_code=500, detail=f"invalid company profile: {error}") from error
