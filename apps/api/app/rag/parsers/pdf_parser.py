"""PDF parser (pdfplumber): rebuilds headings, paragraphs, lists and tables
from the position, size and font of every character.

A PDF has no notion of "paragraph" or "heading": it only places characters on
a page. Everything below is inferred from the layout, with rules relative to
EACH document (its own body font size, its own line spacing), never from
values measured on one particular corpus:

1. read the text lines of every page, with their size and boldness;
2. drop running headers and footers: lines found in the top or bottom margin
   of most pages (digits ignored, so "Page 3 / 6" matches "Page 4 / 6");
3. extract tables, and drop their lines from the running text, otherwise
   every table would appear twice;
4. headings: lines clearly larger than the body text, or bold and numbered;
   their level is the rank of their font size (largest = level 1);
5. group the remaining lines into blocks, split on larger vertical gaps,
   font size changes and list markers; a paragraph cut by a page break is
   joined back.

A table cut by a page break is joined back when the columns match (a header
row repeated on the new page is dropped).

Known limits: no OCR for scanned PDFs (reported as an error); no multi-column
reading order. Lines are read across the whole page width, so two text columns
side by side get interleaved. Such pages are DETECTED and reported in the
warnings (see ``_looks_multi_column``), so that the problem is never silent.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

import pdfplumber

from .base import (
    BaseParser,
    ElementDraft,
    ParsingError,
    apply_section_paths,
    has_bullet_marker,
    has_list_marker,
    heading_number,
    mark_numbered_runs,
    normalize_text,
    register_parser,
    table_to_text,
)

_BOLD_FONT = re.compile(r"bold|black|heavy", re.IGNORECASE)
_SENTENCE_END = re.compile(r"[.!?:;»)\]]$")

MARGIN_RATIO = 0.10  # top and bottom 10 % of the page are where headers and footers live
REPEATED_RATIO = 0.5  # a margin line found on at least half of the pages is a header/footer
MAX_HEADING_LENGTH = 150
# Multi-column detection (see _looks_multi_column)
MIN_LINES_FOR_COLUMNS = 6  # too few lines on a page to say anything
SPLIT_LINES_RATIO = 0.4  # at least 40 % of the lines are cut by the same empty band...
CROSSING_LINES_RATIO = 0.2  # ...and at most 20 % run through it (full-width titles)


@dataclass
class _Line:
    page: int
    top: float
    bottom: float
    x0: float
    x1: float
    text: str
    size: float  # median character size, rounded to 0.5
    bold: float  # share of bold characters, 0 to 1
    spans: tuple[tuple[float, float], ...] = ()  # horizontal extent of each visible character


@dataclass
class _Block:
    page: int
    top: float
    lines: list[_Line] = field(default_factory=list)
    heading: bool = False


@register_parser
class PdfParser(BaseParser):
    name = "pdf"
    version = "1.1"  # 1.1: multi-column pages reported in the warnings
    extensions = frozenset({".pdf"})

    def extract(self, path: Path, warnings: list[str]) -> Iterator[ElementDraft]:
        with pdfplumber.open(path) as pdf:
            pages = [_read_page(number, page) for number, page in enumerate(pdf.pages, 1)]
        if not any(lines for lines, _, _ in pages):
            raise ParsingError(f"{path.name} has no text layer (scanned PDF?): OCR is needed.")

        repeated = _repeated_margin_texts(pages)
        page_lines: list[list[_Line]] = []
        page_tables: list[list[tuple[float, ElementDraft]]] = []
        for number, (lines, tables, height) in enumerate(pages, 1):
            lines = [line for line in lines if not _is_running_text(line, height, repeated)]
            table_drafts = []
            for bbox, draft in tables:
                lines = [line for line in lines if not _inside(line, bbox)]
                draft.page = number
                table_drafts.append((bbox[1], draft))
            if not lines and not table_drafts:
                warnings.append(f"page {number} has no text.")
            page_lines.append(lines)
            page_tables.append(table_drafts)

        all_lines = [line for lines in page_lines for line in lines]
        body_size = _body_size(all_lines)
        for number, lines in enumerate(page_lines, 1):
            if _looks_multi_column(lines, body_size):
                warnings.append(
                    f"page {number} seems to have several text columns: "
                    "its reading order may be wrong (columns may be interleaved)."
                )
        headings = {id(line) for line in all_lines if _looks_like_heading(line, body_size)}
        levels = _heading_levels([line for line in all_lines if id(line) in headings])
        threshold = _line_pitch(page_lines, headings, body_size) * 1.25

        drafts: list[ElementDraft] = []
        for lines, tables in zip(page_lines, page_tables, strict=True):
            items: list[tuple[float, ElementDraft]] = list(tables)
            for block in _blocks(lines, headings, threshold):
                items.append((block.top, self._block_to_draft(block, levels)))
            items.sort(key=lambda item: item[0])  # stable: equal tops keep their order
            for position, (_, draft) in enumerate(items):
                if position == 0 and _continues_previous_page(drafts, draft):
                    drafts[-1].text = _join([drafts[-1].text, draft.text])
                elif position == 0 and _continues_previous_table(drafts, draft):
                    _merge_tables(drafts[-1], draft)
                else:
                    drafts.append(draft)

        mark_numbered_runs(drafts)
        apply_section_paths(drafts)
        yield from drafts

    def _block_to_draft(self, block: _Block, levels: dict[float, int]) -> ElementDraft:
        text = normalize_text(_join([line.text for line in block.lines]))
        if block.heading:
            return ElementDraft(
                type="heading", text=text, page=block.page, level=levels[block.lines[0].size]
            )
        if has_list_marker(text):
            return ElementDraft(type="list_item", text=text, page=block.page)
        if self.is_disclaimer(text):
            return ElementDraft(type="disclaimer", text=text, page=block.page)
        return ElementDraft(type="paragraph", text=text, page=block.page)


# --- Reading ----------------------------------------------------------------------


def _read_page(number: int, page) -> tuple[list[_Line], list[tuple[tuple, ElementDraft]], float]:
    lines = []
    for raw in page.extract_text_lines(return_chars=True, strip=True):
        chars = [char for char in raw["chars"] if char["text"].strip()]
        text = normalize_text(raw["text"])
        if not chars or not text:
            continue
        bold = sum(1 for char in chars if _BOLD_FONT.search(char.get("fontname", ""))) / len(chars)
        lines.append(
            _Line(
                page=number,
                top=raw["top"],
                bottom=raw["bottom"],
                x0=raw["x0"],
                x1=raw["x1"],
                text=text,
                size=round(median(char["size"] for char in chars) * 2) / 2,
                bold=bold,
                spans=tuple(sorted((char["x0"], char["x1"]) for char in chars)),
            )
        )
    lines.sort(key=lambda line: (round(line.top), line.x0))
    read_tables = (_read_table(page, found) for found in page.find_tables())
    tables = [table for table in read_tables if table is not None]
    return lines, tables, float(page.height)


def _read_table(page, table) -> tuple[tuple, ElementDraft] | None:
    raw_rows = table.extract()
    rows = [[normalize_text(cell or "") for cell in row] for row in raw_rows]
    kept = [position for position, row in enumerate(rows) if any(row)]
    if not kept:
        return None
    rows = [rows[position] for position in kept]
    if len(rows) == 1 and len(rows[0]) == 1:  # a frame around some text: it is a paragraph
        return table.bbox, ElementDraft(type="paragraph", text=rows[0][0])

    first_row_bbox = table.rows[kept[0]].bbox
    header = rows[0] if len(rows) > 1 and _is_bold_area(page, first_row_bbox) else None
    body = rows[1:] if header is not None else rows
    text = table_to_text(header, body)
    return table.bbox, ElementDraft(type="table", text=text, data={"header": header, "rows": body})


def _is_bold_area(page, bbox: tuple) -> bool:
    x0, top, x1, bottom = bbox
    chars = [
        char
        for char in page.chars
        if char["text"].strip()
        and char["x0"] >= x0 - 1
        and char["x1"] <= x1 + 1
        and char["top"] >= top - 1
        and char["bottom"] <= bottom + 1
    ]
    if not chars:
        return False
    return sum(1 for c in chars if _BOLD_FONT.search(c.get("fontname", ""))) / len(chars) >= 0.9


# --- Headers, footers, tables --------------------------------------------------------


def _margin_key(text: str) -> str:
    return re.sub(r"\d+", "#", text)


def _in_margin(line: _Line, height: float) -> bool:
    return line.top < height * MARGIN_RATIO or line.bottom > height * (1 - MARGIN_RATIO)


def _repeated_margin_texts(pages: list[tuple[list[_Line], list, float]]) -> set[str]:
    if len(pages) < 2:
        return set()  # one page: nothing can be "repeated"
    counts: Counter[str] = Counter()
    for lines, _, height in pages:
        counts.update({_margin_key(line.text) for line in lines if _in_margin(line, height)})
    minimum = max(2, math.ceil(len(pages) * REPEATED_RATIO))
    return {key for key, count in counts.items() if count >= minimum}


def _is_running_text(line: _Line, height: float, repeated: set[str]) -> bool:
    return _in_margin(line, height) and _margin_key(line.text) in repeated


def _inside(line: _Line, bbox: tuple) -> bool:
    x0, top, x1, bottom = bbox
    middle = (line.top + line.bottom) / 2
    return top <= middle <= bottom and line.x0 < x1 and line.x1 > x0


# --- Typography ---------------------------------------------------------------------


def _body_size(lines: list[_Line]) -> float:
    """The font size that carries the most text in this document."""

    sizes: Counter[float] = Counter()
    for line in lines:
        sizes[line.size] += len(line.text)
    return sizes.most_common(1)[0][0] if sizes else 10.0


def _looks_like_heading(line: _Line, body_size: float) -> bool:
    # "1. Objet" may be a heading, "• Objet" never is.
    if len(line.text) > MAX_HEADING_LENGTH or has_bullet_marker(line.text):
        return False
    if line.size >= body_size + 2:  # clearly larger than the body text
        return True
    if line.bold < 0.9:
        return False
    # Bold and a bit larger, or bold and numbered at body size ("3.2 Stagiaires").
    return line.size >= body_size + 0.75 or (
        line.size >= body_size - 0.25 and heading_number(line.text) is not None
    )


def _looks_multi_column(lines: list[_Line], body_size: float) -> bool:
    """True if the page has an empty vertical band (a "gutter") between two columns.

    For a candidate band in the middle half of the text area, a line is
    "split" if it has characters on both sides of the band and none inside,
    and "crossing" if it has characters inside. In single-column text the
    spaces between words never line up, so no band splits many lines. In two
    columns, the gutter splits most lines and only full-width titles cross it.
    The band is 1.5 x the body font size wide: much wider than a word space.
    """

    if len(lines) < MIN_LINES_FOR_COLUMNS:
        return False
    left = min(line.x0 for line in lines)
    right = max(line.x1 for line in lines)
    gutter = body_size * 1.5
    step = max(body_size / 5, 1.0)

    x = left + (right - left) * 0.25
    while x + gutter <= left + (right - left) * 0.75:
        split = crossing = 0
        for line in lines:
            if any(x0 < x + gutter and x1 > x for x0, x1 in line.spans):
                crossing += 1
            elif any(x1 <= x for _, x1 in line.spans) and any(x0 >= x + gutter for x0, _ in line.spans):
                split += 1
        if split >= len(lines) * SPLIT_LINES_RATIO and crossing <= len(lines) * CROSSING_LINES_RATIO:
            return True
        x += step
    return False


def _heading_levels(heading_lines: list[_Line]) -> dict[float, int]:
    sizes = sorted({line.size for line in heading_lines}, reverse=True)
    return {size: rank for rank, size in enumerate(sizes, 1)}


def _line_pitch(page_lines: list[list[_Line]], headings: set[int], body_size: float) -> float:
    """Usual distance between two lines of the same body paragraph.

    Gaps between consecutive body lines mix two populations: line spacing
    inside a paragraph (small) and spacing between paragraphs (larger). When
    most paragraphs are one line long, the median is a paragraph gap, so the
    median cannot be used. The line pitch is the SMALLEST gap that is frequent
    (at least 10 % of the gaps, and seen twice), which ignores rare odd gaps.
    """

    gaps = []
    for lines in page_lines:
        for previous, line in zip(lines, lines[1:], strict=False):
            if id(previous) in headings or id(line) in headings:
                continue
            if abs(previous.size - body_size) > 0.5 or abs(line.size - body_size) > 0.5:
                continue
            gap = line.top - previous.top
            if 0 < gap < body_size * 3:
                gaps.append(gap)
    if not gaps:
        return body_size * 1.45
    counts = Counter(round(gap * 2) / 2 for gap in gaps)
    frequent = [gap for gap, count in counts.items() if count >= max(2, len(gaps) * 0.1)]
    return min(frequent) if frequent else min(median(gaps), body_size * 1.45)


# --- Blocks -------------------------------------------------------------------------


def _blocks(lines: list[_Line], headings: set[int], threshold: float) -> list[_Block]:
    blocks: list[_Block] = []
    for line in lines:
        is_heading = id(line) in headings
        current = blocks[-1] if blocks else None
        if current is not None and current.heading == is_heading and _same_block(
            current.lines[-1], line, threshold, is_heading
        ):
            current.lines.append(line)
        else:
            blocks.append(_Block(page=line.page, top=line.top, lines=[line], heading=is_heading))
    return blocks


def _same_block(previous: _Line, line: _Line, threshold: float, heading: bool) -> bool:
    if line.top - previous.top > threshold:
        return False
    if heading:  # a long heading wrapped over two lines
        return line.size == previous.size
    return abs(line.size - previous.size) <= 0.5 and not has_list_marker(line.text)


def _join(parts: list[str]) -> str:
    """Join lines; a line ending with a hyphen is glued to the next ("demi-" + "journée")."""

    text = ""
    for part in parts:
        if not text:
            text = part
        elif text.endswith("-") and len(text) > 1 and text[-2].isalpha():
            text += part
        else:
            text += " " + part
    return text


def _continues_previous_page(drafts: list[ElementDraft], draft: ElementDraft) -> bool:
    """A paragraph cut by a page break: no final punctuation, then a lowercase start."""

    if not drafts or draft.type != "paragraph" or drafts[-1].type != "paragraph":
        return False
    return not _SENTENCE_END.search(drafts[-1].text) and draft.text[:1].islower()


def _continues_previous_table(drafts: list[ElementDraft], draft: ElementDraft) -> bool:
    """A table at the top of a page, with the columns of the table that ended the previous page."""

    if not drafts or draft.type != "table" or drafts[-1].type != "table":
        return False
    previous, current = drafts[-1].data or {}, draft.data or {}
    width = len(previous["rows"][0]) if previous.get("rows") else None
    if width is None or any(len(row) != width for row in current.get("rows", [])):
        return False
    return current.get("header") in (None, previous.get("header"))


def _merge_tables(previous: ElementDraft, current: ElementDraft) -> None:
    assert previous.data is not None and current.data is not None
    previous.data["rows"] = previous.data["rows"] + current.data["rows"]
    previous.text = table_to_text(previous.data["header"], previous.data["rows"])
