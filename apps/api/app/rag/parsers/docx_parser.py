"""DOCX parser: walks the document body in order, paragraphs and tables mixed.

A Word file already says what each paragraph is, through its style or its
outline level, so nothing has to be guessed from the look of the text:

- heading: outline level (set on the paragraph or its style), else a style
  named "Heading N" / "Titre N"; the "Title" style is level 1 and "Heading N"
  is level N + 1, so that level 1 is always the document title, as in PDFs;
- list item: Word numbering (``numPr``) or a "List..." style; the marker that
  Word draws on screen ("1.", "•") is not in the text, so it is re-attached;
- table: header row = first row if it is marked as repeating header or if all
  its text is bold.

Page headers and footers are not part of the body, so they are left out
naturally. There are no page numbers in a DOCX file: Word computes them.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import docx
from docx.document import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from .base import (
    BaseParser,
    ElementDraft,
    apply_section_paths,
    has_list_marker,
    mark_numbered_runs,
    normalize_text,
    register_parser,
    table_to_text,
)

_HEADING_STYLE = re.compile(r"^(?:heading|titre)\s*(\d)$", re.IGNORECASE)
_TITLE_STYLES = {"title", "titre"}
_ORDERED_FORMATS = {
    "decimal", "decimalZero", "upperRoman", "lowerRoman", "upperLetter", "lowerLetter"
}


@register_parser
class DocxParser(BaseParser):
    name = "docx"
    version = "1.0"
    extensions = frozenset({".docx"})

    def extract(self, path: Path, warnings: list[str]) -> Iterator[ElementDraft]:
        document = docx.Document(str(path))
        numbering = _ListNumbering(document)
        drafts: list[ElementDraft] = []

        for child in document.element.body.iterchildren():
            if child.tag == qn("w:p"):
                draft = self._paragraph(Paragraph(child, document), numbering)
                if draft is not None:
                    drafts.append(draft)
            elif child.tag == qn("w:tbl"):
                draft = _table(Table(child, document), warnings)
                if draft is not None:
                    drafts.append(draft)

        mark_numbered_runs(drafts)
        apply_section_paths(drafts)
        yield from drafts

    def _paragraph(self, paragraph: Paragraph, numbering: _ListNumbering) -> ElementDraft | None:
        text = normalize_text(paragraph.text)
        if not text:
            return None

        level = _heading_level(paragraph)
        if level is not None:
            return ElementDraft(type="heading", text=text, level=level)

        marker = numbering.marker(paragraph)
        if marker is not None:
            if marker and not has_list_marker(text):
                text = f"{marker} {text}"
            return ElementDraft(type="list_item", text=text)
        if has_list_marker(text):
            return ElementDraft(type="list_item", text=text)
        if self.is_disclaimer(text):
            return ElementDraft(type="disclaimer", text=text)
        return ElementDraft(type="paragraph", text=text)


def _style_chain(paragraph: Paragraph) -> Iterator:
    """The paragraph's style, then the style it is based on, and so on."""

    style = paragraph.style
    seen = set()
    while style is not None and style.style_id not in seen:
        seen.add(style.style_id)
        yield style
        style = style.base_style


def _outline_level(element) -> int | None:
    ppr = element.find(qn("w:pPr"))
    node = ppr.find(qn("w:outlineLvl")) if ppr is not None else None
    if node is None:
        return None
    value = int(node.get(qn("w:val")))
    return value if value < 9 else None  # 9 means "body text" in Word


def _heading_level(paragraph: Paragraph) -> int | None:
    level = _outline_level(paragraph._p)
    if level is not None:
        return level + 2
    for style in _style_chain(paragraph):
        name = (style.name or "").strip()
        if name.lower() in _TITLE_STYLES:
            return 1
        match = _HEADING_STYLE.match(name)
        if match:
            return int(match.group(1)) + 1
        level = _outline_level(style.element)
        if level is not None:
            return level + 2
    return None


class _ListNumbering:
    """Rebuilds the markers that Word draws for numbered and bulleted lists."""

    def __init__(self, document: Document) -> None:
        self._formats: dict[tuple[str, int], str] = {}
        self._counters: dict[str, list[int]] = {}
        try:
            numbering = document.part.numbering_part.element
        except (KeyError, NotImplementedError):  # no numbering part in the file
            return
        abstract_formats: dict[str, dict[int, str]] = {}
        for abstract in numbering.findall(qn("w:abstractNum")):
            levels = {}
            for lvl in abstract.findall(qn("w:lvl")):
                fmt = lvl.find(qn("w:numFmt"))
                levels[int(lvl.get(qn("w:ilvl")))] = fmt.get(qn("w:val")) if fmt is not None else ""
            abstract_formats[abstract.get(qn("w:abstractNumId"))] = levels
        for num in numbering.findall(qn("w:num")):
            abstract_id = num.find(qn("w:abstractNumId")).get(qn("w:val"))
            for ilvl, fmt in abstract_formats.get(abstract_id, {}).items():
                self._formats[(num.get(qn("w:numId")), ilvl)] = fmt

    def marker(self, paragraph: Paragraph) -> str | None:
        """None if the paragraph is not a list item; else its marker ("3.", "•" or "")."""

        num_id, ilvl = _numbering_properties(paragraph)
        if num_id is None:
            names = [(style.name or "").lower() for style in _style_chain(paragraph)]
            if not any(name.startswith("list") for name in names):
                return None
            # Style-only list ("List Number", "List Bullet") without numbering data.
            if any("number" in name for name in names):
                return f"{self._next('style:' + ' '.join(names), 0)}."
            return "" if has_list_marker(normalize_text(paragraph.text)) else "•"
        if num_id == "0":  # numbering explicitly removed
            return None
        fmt = self._formats.get((num_id, ilvl), "bullet")
        if fmt in _ORDERED_FORMATS:
            return f"{self._next(num_id, ilvl)}."
        return "•"

    def _next(self, key: str, ilvl: int) -> int:
        counters = self._counters.setdefault(key, [])
        while len(counters) <= ilvl:
            counters.append(0)
        counters[ilvl] += 1
        del counters[ilvl + 1 :]  # a new item restarts the deeper levels
        return counters[ilvl]


def _numbering_properties(paragraph: Paragraph) -> tuple[str | None, int]:
    """numId and level of the paragraph, from itself or from its style chain."""

    for element in [paragraph._p, *(s.element for s in _style_chain(paragraph))]:
        ppr = element.find(qn("w:pPr"))
        numpr = ppr.find(qn("w:numPr")) if ppr is not None else None
        if numpr is None:
            continue
        num_id = numpr.find(qn("w:numId"))
        ilvl = numpr.find(qn("w:ilvl"))
        if num_id is not None:
            return num_id.get(qn("w:val")), int(ilvl.get(qn("w:val"))) if ilvl is not None else 0
    return None, 0


def _table(table: Table, warnings: list[str]) -> ElementDraft | None:
    rows: list[list[str]] = []
    header_flags: list[bool] = []
    for row in table.rows:
        cells, seen = [], set()
        for cell in row.cells:
            if id(cell._tc) in seen:  # a merged cell is returned once per grid column
                continue
            seen.add(id(cell._tc))
            cells.append(normalize_text(cell.text))
            if cell._tc.find(".//" + qn("w:tbl")) is not None:
                warnings.append("nested table found: its text is kept inside the outer cell.")
        if any(cells):
            rows.append(cells)
            header_flags.append(_is_header_row(row))
    if not rows:
        return None

    header = rows[0] if header_flags[0] and len(rows) > 1 else None
    body = rows[1:] if header is not None else rows
    if len(rows) == 1 and len(rows[0]) == 1:  # a one-cell frame around some text
        return ElementDraft(type="paragraph", text=rows[0][0])
    return ElementDraft(
        type="table",
        text=table_to_text(header, body),
        data={"header": header, "rows": body},
    )


def _is_header_row(row) -> bool:
    trpr = row._tr.trPr
    if trpr is not None and trpr.find(qn("w:tblHeader")) is not None:
        return True
    paragraphs = [paragraph for cell in row.cells for paragraph in cell.paragraphs]
    runs = [run for paragraph in paragraphs for run in paragraph.runs if run.text.strip()]
    return bool(runs) and all(_run_is_bold(run) for run in runs)


def _run_is_bold(run) -> bool:
    if run.bold is not None:
        return run.bold
    style = run.style
    while style is not None:
        if style.font.bold is not None:
            return style.font.bold
        style = style.base_style
    return False
