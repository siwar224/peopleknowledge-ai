"""Output contract of the chunking stage.

A ``Chunk`` is the unit that is embedded, stored in the vector database,
retrieved and finally shown to the generation model. Everything the later
stages need is IN the chunk, so that they never have to open another file:

- ``text``: what is embedded = breadcrumb ``prefix`` + ``content``;
- ``document``: ALL the document metadata (identity, validity, scope), copied
  from the parsed document, for filtering (``lifecycle_status``...) and citing;
- ``section_number`` / ``section_path`` / ``pages``: where it comes from, so
  that an answer can cite "HR-005 §4.2, p. 3";
- ``element_ids``: the exact parsed elements it was built from (traceability).

The chunks of one document version are written, one per line, to
``datasets/processed/chunks/<version_id>.jsonl``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..parsers.models import DocumentMetadata, SourceInfo

CHUNK_SCHEMA_VERSION = 1

ChunkKind = Literal[
    "section",  # a heading and everything below it (or one part of it, if it was too long)
    "document_card",  # the cover page: title, subtitle, identity table
    "record",  # one CSV row
]


class ChunkingInfo(BaseModel):
    """How the chunk was produced, so that chunks made with other settings can be found."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunker_version: str = Field(min_length=1)
    tokenizer: str = Field(min_length=1)  # e.g. "BAAI/bge-m3"
    max_tokens: int = Field(gt=0)


class Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = CHUNK_SCHEMA_VERSION
    chunk_id: str = Field(min_length=1)  # "HR-005-v1.2#4.2", "HR-005-v1.2#4.2-2"
    kind: ChunkKind

    prefix: str = Field(min_length=1)  # "HR-005 v1.2 — Politique ... > 4. ... > 4.2 Exception"
    content: str = Field(min_length=1)  # the section's own text
    text: str = Field(min_length=1)  # prefix + "\n" + content: what is embedded
    token_count: int = Field(gt=0)  # tokens of ``text``, with the embedding model's tokenizer

    section_number: str | None = None  # "4.2"; None for a document card or a CSV row
    section_path: list[str] = Field(default_factory=list)  # every heading above, own one included
    part: int = Field(default=1, ge=1)  # 2 for "HR-005-v1.2#4.2-2"
    part_count: int = Field(default=1, ge=1)
    pages: list[int] = Field(default_factory=list)  # empty for DOCX and CSV
    element_ids: list[str] = Field(min_length=1)  # parsed elements this chunk was built from

    document: DocumentMetadata
    source: SourceInfo
    chunking: ChunkingInfo

    @model_validator(mode="after")
    def check_consistency(self) -> Chunk:
        if self.text != f"{self.prefix}\n{self.content}":
            raise ValueError("text must be exactly prefix + newline + content.")
        if not self.content.strip():
            raise ValueError("content must not be blank.")
        if not self.chunk_id.startswith(f"{self.document.version_id}#"):
            raise ValueError(
                f"chunk_id {self.chunk_id!r} must start with '{self.document.version_id}#'."
            )
        if self.part > self.part_count:
            raise ValueError(f"part {self.part} is greater than part_count {self.part_count}.")
        if self.token_count > self.chunking.max_tokens:
            raise ValueError(
                f"{self.token_count} tokens is above the maximum of {self.chunking.max_tokens}."
            )
        if len(set(self.element_ids)) != len(self.element_ids):
            raise ValueError("element_ids must not contain duplicates.")
        return self

    def to_json_line(self) -> str:
        """One line of the .jsonl file (accents kept as is)."""

        return self.model_dump_json() + "\n"
