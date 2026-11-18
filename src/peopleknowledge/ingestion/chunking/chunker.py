"""Turn one parsed document into chunks (see models.py for the output contract).

The parsing stage already did the hard work: elements are clean, in reading
order, and each one knows its section. Chunking is therefore mostly GROUPING:

1. disclaimers are left out;
2. the elements are grouped:
   - document card: what comes before the first section (title, subtitle,
     identity table), as one chunk;
   - section: a heading and the elements below it, until the next heading;
     a heading with no text of its own (directly followed by a sub-heading)
     gives no chunk: it only appears in the prefix of its sub-sections;
   - record: one CSV row;
3. each chunk gets a breadcrumb prefix ("HR-005 v1.2 — Title > 4. ... > 4.2 ...")
   which is embedded with the content;
4. a chunk above ``max_tokens`` (prefix included) is split, never inside a
   sentence or a table row: first between elements, then a table between its
   rows, then a paragraph between its sentences (words only as a last resort);
5. ids: ``<version_id>#<section number>``, ``-2``, ``-3``... for the next
   parts, ``<version_id>#card`` for the card, ``<version_id>#<row id>`` for rows.

Nothing here depends on one corpus. Token counting is passed in as a function,
so that tests can use a fake counter and nothing has to be downloaded.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from ..parsers.models import ParsedDocument, ParsedElement
from .models import Chunk, ChunkingInfo, ChunkKind

CHUNKER_VERSION = "1.0"
DEFAULT_MAX_TOKENS = 512

TokenCounter = Callable[[str], int]

# Where a too-long paragraph may be cut: after . ! ? ; or : followed by a space.
_SENTENCE_END = re.compile(r"(?<=[.!?;:])\s+")


class ChunkingError(Exception):
    """A document cannot be chunked within the limits (e.g. a prefix alone is too long)."""


@dataclass
class _Group:
    kind: ChunkKind
    id_suffix: str  # "4.2", "card", "RW-ENG-008"
    section_path: list[str]  # full path, the group's own heading included
    section_number: str | None
    elements: list[ParsedElement] = field(default_factory=list)  # content only, no heading


@dataclass
class _Block:
    """The smallest piece of text that is never cut further (unless it is too long alone)."""

    text: str
    element: ParsedElement


def chunk_document(
    document: ParsedDocument,
    count_tokens: TokenCounter,
    tokenizer_name: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> list[Chunk]:
    """All the chunks of one parsed document, in reading order."""

    info = ChunkingInfo(
        chunker_version=CHUNKER_VERSION, tokenizer=tokenizer_name, max_tokens=max_tokens
    )
    doc = document.document
    head = f"{doc.document_id} v{doc.version} — {doc.title}"
    groups, title = _group(document)

    chunks: list[Chunk] = []
    used_ids: set[str] = set()
    for group in groups:
        prefix = _prefix(head, group.section_path, title)
        if count_tokens(prefix + "\n") >= max_tokens:
            raise ChunkingError(f"prefix alone is {max_tokens}+ tokens: {prefix[:80]!r}")
        parts = _split(prefix, _blocks(group.elements), count_tokens, max_tokens)
        base_id = _unique(f"{document.document.version_id}#{group.id_suffix}", used_ids, group)
        for number, blocks in enumerate(parts, 1):
            content = _join(blocks)
            text = f"{prefix}\n{content}"
            elements = list({id(b.element): b.element for b in blocks}.values())  # ordered, unique
            chunks.append(
                Chunk(
                    chunk_id=base_id if number == 1 else f"{base_id}-{number}",
                    kind=group.kind,
                    prefix=prefix,
                    content=content,
                    text=text,
                    token_count=count_tokens(text),
                    section_number=group.section_number,
                    section_path=group.section_path,
                    part=number,
                    part_count=len(parts),
                    pages=sorted({e.page for e in elements if e.page is not None}),
                    element_ids=[e.element_id for e in elements],
                    document=document.document,
                    source=document.source,
                    chunking=info,
                )
            )
    return chunks


# --- 1-2. Grouping ---------------------------------------------------------------------


def _group(document: ParsedDocument) -> tuple[list[_Group], str | None]:
    """Groups in reading order, and the text of the document title heading (if any).

    The first heading is the document TITLE (and opens the card) only if it is
    level 1, not numbered, and deeper headings exist. Otherwise a document whose
    sections are all level 1 ("1. Objet", "2. ...") would lose its first section.
    """

    version_id = document.document.version_id
    has_deeper = any(e.type == "heading" and (e.level or 0) > 1 for e in document.elements)
    groups: list[_Group] = []
    current: _Group | None = None
    title: str | None = None
    seen_section = False  # becomes True at the first heading that is not the document title

    for element in document.elements:
        if element.type == "disclaimer":
            continue
        if element.type == "record":
            suffix = element.element_id.removeprefix(f"{version_id}#")
            groups.append(_Group("record", suffix, list(element.section_path), None, [element]))
            continue
        if element.type == "heading":
            is_title = (
                not seen_section
                and title is None
                and element.level == 1
                and element.section_number is None
                and has_deeper
            )
            if is_title:
                title = element.text
                current = _Group("document_card", "card", [element.text], None)
            else:
                seen_section = True
                current = _Group(
                    "section",
                    element.section_number or f"s{element.index:04d}",
                    [*element.section_path, element.text],
                    element.section_number,
                )
            groups.append(current)
            continue
        if current is None:  # content before any heading: it belongs to the card
            current = _Group("document_card", "card", [], None)
            groups.append(current)
        current.elements.append(element)

    return [group for group in groups if group.elements], title  # no chunk for a heading alone


def _prefix(head: str, section_path: list[str], title: str | None) -> str:
    """Breadcrumb. The document title is already in ``head``: it is not repeated."""

    path = section_path[1:] if title is not None and section_path[:1] == [title] else section_path
    return " > ".join([head, *path])


def _unique(chunk_id: str, used: set[str], group: _Group) -> str:
    """Two sections with the same number (a typo in a document) must not share an id."""

    if chunk_id in used:
        chunk_id = f"{chunk_id}@{group.elements[0].index:04d}"
    used.add(chunk_id)
    return chunk_id


# --- 4. Splitting -----------------------------------------------------------------------


def _blocks(elements: list[ParsedElement]) -> list[_Block]:
    """Elements as uncuttable blocks; a table gives one block per row (one line of its text)."""

    blocks: list[_Block] = []
    for element in elements:
        if element.type == "table":
            rows = [line for line in element.text.splitlines() if line.strip()]
            blocks.extend(_Block(row, element) for row in rows)
        else:
            blocks.append(_Block(element.text, element))
    return blocks


def _join(blocks: list[_Block]) -> str:
    """Pieces of the same paragraph (cut into sentences) are joined back with a space;
    different elements and table rows go on separate lines."""

    text = ""
    for position, block in enumerate(blocks):
        if position == 0:
            text = block.text
        elif block.element is blocks[position - 1].element and block.element.type != "table":
            text += " " + block.text
        else:
            text += "\n" + block.text
    return text


def _split(
    prefix: str, blocks: list[_Block], count_tokens: TokenCounter, max_tokens: int
) -> list[list[_Block]]:
    """Pack blocks, in order, into as few parts as possible, each within ``max_tokens``."""

    def fits(candidate: list[_Block]) -> bool:
        return count_tokens(prefix + "\n" + _join(candidate)) <= max_tokens

    parts: list[list[_Block]] = []
    current: list[_Block] = []
    queue = list(blocks)
    while queue:
        block = queue.pop(0)
        if fits([*current, block]):
            current.append(block)
            continue
        if current:  # close the current part, retry the block in a new one
            parts.append(current)
            current = []
            queue.insert(0, block)
            continue
        # The block alone is too long: cut it into sentences, or words as a last resort.
        pieces = _cut(block.text)
        if len(pieces) == 1:
            raise ChunkingError(f"cannot split a block small enough: {block.text[:80]!r}")
        queue[:0] = [_Block(piece, block.element) for piece in pieces]
    if current:
        parts.append(current)
    return parts


def _cut(text: str) -> list[str]:
    sentences = [s for s in _SENTENCE_END.split(text) if s]
    if len(sentences) > 1:
        return sentences
    words = text.split()
    if len(words) > 1:
        middle = len(words) // 2
        return [" ".join(words[:middle]), " ".join(words[middle:])]
    return [text]
