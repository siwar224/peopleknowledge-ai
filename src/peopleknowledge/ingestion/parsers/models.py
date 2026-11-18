"""Output contract of the parsing stage.

Every parser (PDF, DOCX, CSV) produces a ``ParsedDocument``, whatever the input
format. The next stages (chunking, embeddings) only read this contract and never
need to know where the text came from.

One ``ParsedDocument`` is written per document version, as
``datasets/processed/parsed/<version_id>.json``.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

PARSED_SCHEMA_VERSION = 1

ElementType = Literal[
    "heading",  # numbered title, e.g. "4.2 Exception : jusqu'à 2 jours pendant l'essai"
    "paragraph",  # running text, examples included
    "list_item",  # one list item, with its marker re-attached ("1. le manager donne son accord")
    "table",  # a table from a PDF or DOCX; ``data`` holds headers and rows
    "record",  # one CSV row; ``data`` holds the raw field values
    "disclaimer",  # the "Document fictif" notice, so that chunking can exclude it
]

# Element types that must carry structured ``data`` next to their ``text``.
TYPES_WITH_DATA = {"table", "record"}
SECTION_NUMBER_PATTERN = re.compile(r"^\d+(?:\.\d+)*$")


class DocumentMetadata(BaseModel):
    """Identity and validity of the document, copied from the manifest.

    Extra fields are ignored on purpose: a whole manifest record can be passed
    in, and the scan-related fields (``change_status``, ``is_duplicate``...)
    are dropped. They describe the last scan, not the document, and would go
    stale. Values are not re-validated here: ``build_manifest.py`` already
    validates them against the catalog.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    document_id: str = Field(min_length=1)
    version_id: str = Field(min_length=1)
    family_id: str
    title: str
    doc_type: str
    version: str
    lifecycle_status: str
    effective_from: date
    effective_until: date | None = None
    supersedes: str | None = None
    superseded_by: str | None = None
    language: str
    category: str
    department_scope: str
    owner: str


class SourceInfo(BaseModel):
    """Which file was parsed, and its fingerprint at that time.

    If the manifest's ``sha256`` differs from this one, the parsed file is stale.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    relative_path: str = Field(min_length=1)
    extension: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ParsingInfo(BaseModel):
    """Which parser produced the output.

    ``parser_version`` is bumped whenever a parser's output changes, so that
    files produced by an older version can be found and regenerated. There is
    deliberately no parsing date: parsing the same file twice must give an
    identical result.

    ``parser`` is a free name ("pdf", "docx", "csv"...), not a closed list:
    adding a parser for a new format must not require changing this contract.
    The list of available parsers lives in the registry of ``base.py``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    parser: str = Field(min_length=1)
    parser_version: str = Field(min_length=1)


class ParsedElement(BaseModel):
    """One piece of content, in reading order."""

    model_config = ConfigDict(extra="forbid")

    element_id: str = Field(min_length=1)
    index: int = Field(ge=0)
    type: ElementType
    text: str = Field(min_length=1)
    page: int | None = Field(default=None, ge=1)
    section_path: list[str] = Field(default_factory=list)
    section_number: str | None = None
    level: int | None = Field(default=None, ge=1)
    data: dict[str, Any] | None = None

    @model_validator(mode="after")
    def check_type_rules(self) -> ParsedElement:
        """Enforce the rules that depend on the element type."""

        if not self.text.strip():
            raise ValueError("text must not be blank: it is what will be embedded.")
        if self.type == "heading" and self.level is None:
            raise ValueError("a heading must have a level.")
        if self.type != "heading" and self.level is not None:
            raise ValueError("only a heading can have a level.")
        if self.type in TYPES_WITH_DATA and self.data is None:
            raise ValueError(f"a {self.type} element must have structured data.")
        number = self.section_number
        if number is not None and not SECTION_NUMBER_PATTERN.match(number):
            raise ValueError(f"invalid section_number {number!r} (expected e.g. '4.2').")
        return self


class ParsedDocument(BaseModel):
    """The full output of a parser for one document version."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = PARSED_SCHEMA_VERSION
    document: DocumentMetadata
    source: SourceInfo
    parsing: ParsingInfo
    elements: list[ParsedElement]
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_elements(self) -> ParsedDocument:
        """Elements must be in reading order, with unique identifiers."""

        for position, element in enumerate(self.elements):
            if element.index != position:
                raise ValueError(
                    f"element {element.element_id!r} has index {element.index}, "
                    f"expected {position}."
                )
        id_counts = Counter(element.element_id for element in self.elements)
        duplicates = sorted(eid for eid, count in id_counts.items() if count > 1)
        if duplicates:
            raise ValueError(f"duplicate element_id(s): {duplicates}.")
        return self

    def to_json(self) -> str:
        """Serialize to indented UTF-8 JSON (accents are kept as is)."""

        return self.model_dump_json(indent=2) + "\n"
