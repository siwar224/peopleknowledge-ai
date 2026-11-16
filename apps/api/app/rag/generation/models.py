"""Contracts between retrieval, context building and answer generation."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..chunking.models import Chunk
from ..evaluation.gold import Source


class RetrievedChunk(BaseModel):
    """One ranked retrieval result, before it is prepared for the LLM."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: int = Field(ge=1)
    score: float
    chunk: Chunk


class ContextSource(BaseModel):
    """The trusted mapping from a short LLM citation to a corpus source."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(pattern=r"^S[1-9]\d*$")
    rank: int = Field(ge=1)
    score: float
    chunk_id: str = Field(min_length=1)
    source: Source
    title: str = Field(min_length=1)
    version: str = Field(min_length=1)
    lifecycle_status: str = Field(min_length=1)
    effective_from: date
    effective_until: date | None = None
    superseded_by: str | None = None
    pages: list[int] = Field(default_factory=list)


class BuiltContext(BaseModel):
    """Prompt-ready evidence and the sources that its labels refer to."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    sources: list[ContextSource] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_source_labels(self) -> BuiltContext:
        expected = [f"S{number}" for number in range(1, len(self.sources) + 1)]
        actual = [source.source_id for source in self.sources]
        if actual != expected:
            raise ValueError(f"source ids must be consecutive and ordered: expected {expected}.")
        if bool(self.text) != bool(self.sources):
            raise ValueError("text and sources must either both be empty or both be present.")
        return self
