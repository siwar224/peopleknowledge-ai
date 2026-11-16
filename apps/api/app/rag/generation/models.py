"""Contracts between retrieval, context building and answer generation."""

from __future__ import annotations

import re
from datetime import date
from typing import Literal

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


class PromptMessage(BaseModel):
    """One provider-neutral chat message for an OpenAI-compatible LLM."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["system", "user"]
    content: str = Field(min_length=1)


class BuiltPrompt(BaseModel):
    """The stable instructions and dynamic evidence sent to the generator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    messages: list[PromptMessage]
    allowed_source_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_message_order(self) -> BuiltPrompt:
        roles = [message.role for message in self.messages]
        if roles != ["system", "user"]:
            raise ValueError(
                "a prompt must contain exactly one system message then one user message."
            )
        if len(self.allowed_source_ids) != len(set(self.allowed_source_ids)):
            raise ValueError("allowed_source_ids must not contain duplicates.")
        invalid = [
            source_id
            for source_id in self.allowed_source_ids
            if not re.fullmatch(r"S[1-9]\d*", source_id)
        ]
        if invalid:
            raise ValueError(f"invalid allowed source id(s): {invalid}.")
        return self


AnswerStatus = Literal["complete", "partial", "not_found"]
_INLINE_CITATION = re.compile(r"\[(S[1-9]\d*)\]")


class GeneratedAnswer(BaseModel):
    """Validated structured content returned by the language model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: AnswerStatus
    answer: str = Field(min_length=1)
    citations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_citations(self) -> GeneratedAnswer:
        if len(self.citations) != len(set(self.citations)):
            raise ValueError("citations must not contain duplicates.")
        invalid = [
            citation for citation in self.citations if not re.fullmatch(r"S[1-9]\d*", citation)
        ]
        if invalid:
            raise ValueError(f"invalid citation label(s): {invalid}.")
        inline = set(_INLINE_CITATION.findall(self.answer))
        listed = set(self.citations)
        if not inline.issubset(listed):
            raise ValueError("every inline citation must also appear in the citations list.")
        if self.status in {"complete", "partial"} and not self.citations:
            raise ValueError(f"a {self.status} answer needs at least one citation.")
        return self


class GenerationResult(BaseModel):
    """A validated answer plus operational metadata from the generator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    answer: GeneratedAnswer
    model: str = Field(min_length=1)
    elapsed_ms: int = Field(ge=0)
