"""Common interface of all parsers, and the registry that picks one by extension.

A parser only knows how to read ONE kind of file. It never reads the manifest:
the caller (``scripts/parse_documents.py``) gives it a file and the document
metadata, and gets a validated ``ParsedDocument`` back.

Writing a new parser means:

1. subclass ``BaseParser``, set ``name``, ``version`` and ``extensions``;
2. implement ``extract()``, which yields ``ElementDraft`` objects in reading order;
3. decorate the class with ``@register_parser``.

Everything else (index, element ids, file fingerprint, validation) is done
here, once, for every parser. The shared text helpers (normalization, heading
numbers, list markers, section paths, table serialization) live at the end of
this module so that the PDF and DOCX parsers apply exactly the same rules.

No rule here may depend on one particular corpus: corpus conventions (such as
the wording of a disclaimer) are passed in as options.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from pydantic import ValidationError

from .models import (
    DocumentMetadata,
    ElementType,
    ParsedDocument,
    ParsedElement,
    ParsingInfo,
    SourceInfo,
)


class ParsingError(Exception):
    """A document could not be parsed. The batch goes on with the next one."""


class UnsupportedFormatError(ParsingError):
    """No parser is registered for this file extension."""


@dataclass
class ElementDraft:
    """One element as a parser sees it, before numbering.

    The parser only describes WHAT it found. ``index`` and, by default,
    ``element_id`` are assigned by ``BaseParser.parse()``, so that numbering
    is identical for every parser and never has to be handled by hand.
    """

    type: ElementType
    text: str
    page: int | None = None
    section_path: list[str] = field(default_factory=list)
    section_number: str | None = None
    level: int | None = None
    data: dict[str, Any] | None = None
    # Set it only when the source has its own stable id (e.g. a CSV ``row_id``).
    element_id: str | None = None


class BaseParser(ABC):
    """Base class of every parser (template method pattern).

    ``parse()`` is the fixed procedure, the same for all formats.
    ``extract()`` is the only format-specific step, written by each subclass.
    """

    name: ClassVar[str]  # short name stored in ParsingInfo: "pdf", "docx", "csv"...
    version: ClassVar[str]  # bump it whenever the output of this parser changes
    extensions: ClassVar[frozenset[str]]  # lowercase, with the dot: frozenset({".pdf"})

    def __init__(self, disclaimer_patterns: Iterable[str] = ()) -> None:
        """``disclaimer_patterns``: regexes (case-insensitive) that mark a block
        as a ``disclaimer``. They come from configuration, never from the code,
        because the wording of a disclaimer belongs to a corpus.
        """

        self._disclaimer_patterns = [re.compile(p, re.IGNORECASE) for p in disclaimer_patterns]

    def is_disclaimer(self, text: str) -> bool:
        return any(pattern.search(text) for pattern in self._disclaimer_patterns)

    def parse(self, path: Path, document: DocumentMetadata, relative_path: str) -> ParsedDocument:
        """Parse one file into a validated ``ParsedDocument``.

        ``relative_path`` is only recorded in the output (where the file lives
        in the corpus); ``path`` is where the file is actually read from.
        Raises ``ParsingError`` if the file cannot be turned into a valid document.
        """

        path = Path(path)
        extension = path.suffix.lower()
        if extension not in self.extensions:
            raise UnsupportedFormatError(
                f"{self.name} parser cannot read {extension!r} files ({path.name})."
            )
        if not path.is_file():
            raise ParsingError(f"file not found: {path}")

        warnings: list[str] = []
        try:
            drafts = list(self.extract(path, warnings))
        except ParsingError:
            raise
        except Exception as error:  # a broken or unreadable file must not stop the batch
            raise ParsingError(f"{self.name} parser failed on {path.name}: {error}") from error

        if not drafts:
            raise ParsingError(
                f"no content extracted from {path.name} "
                "(empty file, or a scanned PDF that needs OCR?)."
            )

        try:
            return ParsedDocument(
                document=document,
                source=SourceInfo(
                    relative_path=relative_path,
                    extension=extension,
                    sha256=file_sha256(path),
                ),
                parsing=ParsingInfo(parser=self.name, parser_version=self.version),
                elements=[
                    self._to_element(draft, index, document.version_id)
                    for index, draft in enumerate(drafts)
                ],
                warnings=warnings,
            )
        except ValidationError as error:  # the parser produced something the contract refuses
            raise ParsingError(f"invalid output for {path.name}: {error}") from error

    @abstractmethod
    def extract(self, path: Path, warnings: list[str]) -> Iterator[ElementDraft]:
        """Yield the content of the file, in reading order.

        Append a message to ``warnings`` for anything suspicious that does not
        prevent parsing (table without header, empty page...).
        """

    @staticmethod
    def _to_element(draft: ElementDraft, index: int, version_id: str) -> ParsedElement:
        return ParsedElement(
            element_id=draft.element_id or f"{version_id}#{index:04d}",
            index=index,
            type=draft.type,
            text=draft.text,
            page=draft.page,
            section_path=list(draft.section_path),
            section_number=draft.section_number,
            level=draft.level,
            data=draft.data,
        )


def file_sha256(path: Path) -> str:
    """Fingerprint of the bytes actually read, read in blocks (large files are fine)."""

    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


# --- Registry -----------------------------------------------------------------

_REGISTRY: dict[str, type[BaseParser]] = {}


def register_parser(parser_class: type[BaseParser]) -> type[BaseParser]:
    """Class decorator: make a parser available for its extensions."""

    for attribute in ("name", "version", "extensions"):
        if not getattr(parser_class, attribute, None):
            raise TypeError(f"{parser_class.__name__} must define a non-empty {attribute!r}.")
    for extension in parser_class.extensions:
        if extension != extension.lower() or not extension.startswith("."):
            raise TypeError(f"invalid extension {extension!r}: expected lowercase, e.g. '.pdf'.")
        if extension in _REGISTRY:
            raise ValueError(
                f"{extension!r} is already handled by {_REGISTRY[extension].__name__}."
            )
    for extension in parser_class.extensions:
        _REGISTRY[extension] = parser_class
    return parser_class


def get_parser(extension: str, **options: Any) -> BaseParser:
    """Return a parser for this extension (".pdf", ".PDF" and "pdf" are all accepted).

    ``options`` are passed to the parser's constructor (e.g. ``disclaimer_patterns``).
    """

    normalized = extension.lower()
    if not normalized.startswith("."):
        normalized = f".{normalized}"
    try:
        parser_class = _REGISTRY[normalized]
    except KeyError:
        raise UnsupportedFormatError(
            f"no parser for {normalized!r}; supported: {sorted(_REGISTRY) or 'none'}."
        ) from None
    return parser_class(**options)


def supported_extensions() -> list[str]:
    """Extensions that have a registered parser, sorted."""

    return sorted(_REGISTRY)


# --- Shared text helpers --------------------------------------------------------

_LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st"}
_INVISIBLE = dict.fromkeys(map(ord, "­​‌‍⁠﻿"))  # soft hyphen, zero-width
_SPACES = re.compile(r"[\s    ]+")


def normalize_text(text: str) -> str:
    """Clean extracted text without changing its meaning.

    Unicode NFC (an accent is always one character), typographic ligatures
    split, invisible characters removed, every kind of space (non-breaking,
    thin...) turned into one plain space. Figures and punctuation are kept.
    """

    text = unicodedata.normalize("NFC", text).translate(_INVISIBLE)
    for ligature, letters in _LIGATURES.items():
        text = text.replace(ligature, letters)
    return _SPACES.sub(" ", text).strip()


# "4.2 Exception", "1. Objet", "12) Annexe" -> "4.2", "1", "12"
_HEADING_NUMBER = re.compile(r"^(\d{1,3}(?:\.\d{1,3})*)[.)]?\s+\S")


def heading_number(text: str) -> str | None:
    """Section number at the start of a heading, or None."""

    match = _HEADING_NUMBER.match(text)
    return match.group(1) if match else None


# A bullet ("•", "-", "–"...) or an explicit ordered marker ("1.", "2)", "a)") followed by text.
_BULLET_MARKER = re.compile(r"^[•◦▪▫‣⁃●○■□►▸–—\-*·]\s+\S")
_ORDERED_MARKER = re.compile(r"^(?:\d{1,2}|[a-zA-Z])[.)]\s+\S")
# A bare number ("1 le manager donne...") is only a list item inside a 1, 2, 3... run.
_BARE_NUMBER = re.compile(r"^(\d{1,2})\s+\S")


def has_list_marker(text: str) -> bool:
    """True if the text starts with an explicit bullet or ordered-list marker."""

    return bool(_BULLET_MARKER.match(text) or _ORDERED_MARKER.match(text))


def has_bullet_marker(text: str) -> bool:
    """True if the text starts with a bullet. Unlike "1.", a bullet never starts a heading."""

    return bool(_BULLET_MARKER.match(text))


def mark_numbered_runs(drafts: Sequence[ElementDraft]) -> None:
    """Turn paragraphs that start with a bare number into list items, but only
    when they form a run of consecutive numbers (1, 2, 3...).

    "1 le manager donne son accord" followed by "2 l'équipe RH valide" is a
    list; a lone "15 jours sont accordés..." is an ordinary sentence.
    """

    def number(draft: ElementDraft) -> int | None:
        if draft.type != "paragraph":
            return None
        match = _BARE_NUMBER.match(draft.text)
        return int(match.group(1)) if match else None

    numbers = [number(draft) for draft in drafts]
    for position, value in enumerate(numbers):
        if value is None:
            continue
        before = numbers[position - 1] if position > 0 else None
        after = numbers[position + 1] if position + 1 < len(numbers) else None
        if before == value - 1 or after == value + 1:
            drafts[position].type = "list_item"


def apply_section_paths(drafts: Sequence[ElementDraft]) -> None:
    """Fill ``section_path`` and ``section_number`` from the headings, in order.

    A heading's path holds its parents only; any other element's path holds
    every heading above it. A level-2 heading closes the previous level-2 (and
    deeper) section. Non-heading elements inherit the number of the innermost
    numbered section, so that an answer can cite "HR-005 §4.2".
    """

    stack: list[tuple[int, str, str | None]] = []  # (level, title, number)
    for draft in drafts:
        if draft.type == "heading" and draft.level is not None:
            while stack and stack[-1][0] >= draft.level:
                stack.pop()
            draft.section_path = [title for _, title, _ in stack]
            draft.section_number = heading_number(draft.text)
            stack.append((draft.level, draft.text, draft.section_number))
        else:
            draft.section_path = [title for _, title, _ in stack]
            draft.section_number = next((num for _, _, num in reversed(stack) if num), None)


def table_to_text(header: list[str] | None, rows: list[list[str]]) -> str:
    """Readable text of a table, one line per row, for the embedding.

    With a header, each value carries its column name ("Durée : 3 à 5 jours"),
    so that a row keeps its meaning even when it is read alone.
    """

    lines = []
    if header is None:
        lines = [" | ".join(cell for cell in row if cell) for row in rows]
    else:
        for row in rows:
            pairs = [
                f"{name}: {value}" if name else value
                for name, value in zip(header, row, strict=False)
                if value
            ]
            lines.append(" | ".join(pairs))
    return "\n".join(line for line in lines if line)
