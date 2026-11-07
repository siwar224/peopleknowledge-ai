"""Common interface of all parsers, and the registry that picks one by extension.

A parser only knows how to read ONE kind of file. It never reads the manifest:
the caller (``scripts/parse_documents.py``) gives it a file and the document
metadata, and gets a validated ``ParsedDocument`` back.

Writing a new parser means:

1. subclass ``BaseParser``, set ``name``, ``version`` and ``extensions``;
2. implement ``extract()``, which yields ``ElementDraft`` objects in reading order;
3. decorate the class with ``@register_parser``.

Everything else (index, element ids, file fingerprint, validation) is done
here, once, for every parser.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Iterator
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


def get_parser(extension: str) -> BaseParser:
    """Return a parser for this extension (".pdf", ".PDF" and "pdf" are all accepted)."""

    normalized = extension.lower()
    if not normalized.startswith("."):
        normalized = f".{normalized}"
    try:
        return _REGISTRY[normalized]()
    except KeyError:
        raise UnsupportedFormatError(
            f"no parser for {normalized!r}; supported: {sorted(_REGISTRY) or 'none'}."
        ) from None


def supported_extensions() -> list[str]:
    """Extensions that have a registered parser, sorted."""

    return sorted(_REGISTRY)
