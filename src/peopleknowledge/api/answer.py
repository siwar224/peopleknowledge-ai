"""End-to-end RAG answer endpoint and profile options for the web interface.

Every answer is also EVALUATED (evaluation/generation_eval.py): numbers absent
from the sources, prompt leak, repeated question for any question; with a
``gold_id``, also the expected status, the key facts and the cited sources.
"""

from __future__ import annotations

import threading
import time
from datetime import date
from typing import Annotated

import yaml
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, ValidationError

from ..config import Settings, get_settings
from ..evaluation.generation_eval import GenerationEvaluation, evaluate_answer
from ..evaluation.gold import GoldQuestion, Source, UserContext, load_gold
from ..generation import (
    ContextSource,
    GenerationError,
    Generator,
    OpenAICompatibleGenerator,
    RetrievedChunk,
    build_context,
    build_prompt,
)
from ..retrieval.qdrant_index import chunk_from_payload, search
from .search import SearchResult, SearchService, get_search_service, to_result

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
    # A gold question id ("Q016"): the answer is then compared with the expected one.
    gold_id: str | None = Field(default=None, pattern=r"^Q\d{3}$")


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
    evaluation: GenerationEvaluation


def _gold_and_rules(settings: Settings, gold_id: str | None) -> tuple[GoldQuestion | None, dict]:
    """The gold question to compare with (404 if unknown) and the rules of its key facts."""

    if gold_id is None:
        return None, {}
    try:
        questions = {question.id: question for question in load_gold(settings.gold_path)}
        rules = yaml.safe_load(settings.rules_path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError, yaml.YAMLError) as error:
        raise HTTPException(status_code=500, detail=f"evaluation files: {error}") from error
    if gold_id not in questions:
        raise HTTPException(status_code=404, detail=f"unknown gold question {gold_id!r}.")
    return questions[gold_id], rules


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
    settings: Annotated[Settings, Depends(get_settings)],
) -> AnswerResponse:
    """Retrieve evidence, build the prompt, generate, resolve citations, evaluate."""

    gold, rules = _gold_and_rules(settings, body.gold_id)
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
                chunk=chunk_from_payload(point.payload or {}),
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
    evaluation = evaluate_answer(
        question_text=body.question,
        status=result.answer.status,
        answer=result.answer.answer,
        cited=[citation.source for citation in citations],
        prompt_user_text=prompt.messages[1].content,
        system_prompt=prompt.messages[0].content,
        gold=gold,
        rules=rules,
    )
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
        evaluation=evaluation,
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
