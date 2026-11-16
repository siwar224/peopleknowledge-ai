"""CSV parser: one data row = one ``record`` element.

Nothing about a particular file is written here: column names are read from
the header row, the encoding and the delimiter are detected. Two optional
settings let a document refine the output:

- ``id_column``: a column whose values are stable row ids (e.g. ``row_id``);
  they become the ``element_id``, so that a citation survives a re-ordering.
- ``group_by``: a column whose value is used as ``section_path`` (e.g. the
  department), so that retrieval can filter on it.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable, Iterator
from pathlib import Path

from .base import BaseParser, ElementDraft, ParsingError, normalize_text, register_parser

# Tried in this order. cp1252 is the usual encoding of a CSV saved by Excel on Windows.
ENCODINGS = ("utf-8-sig", "cp1252")
DELIMITERS = ",;\t|"


@register_parser
class CsvParser(BaseParser):
    name = "csv"
    version = "1.0"
    extensions = frozenset({".csv"})

    def __init__(
        self,
        disclaimer_patterns: Iterable[str] = (),
        id_column: str | None = None,
        group_by: str | None = None,
    ) -> None:
        super().__init__(disclaimer_patterns)
        self.id_column = id_column
        self.group_by = group_by

    def extract(self, path: Path, warnings: list[str]) -> Iterator[ElementDraft]:
        content = _decode(path.read_bytes(), warnings)
        # StringIO (not splitlines) so that a quoted value may contain a line break.
        rows = list(csv.reader(io.StringIO(content, newline=""), _dialect(content, warnings)))
        rows = [row for row in rows if any(cell.strip() for cell in row)]  # skip blank lines
        if not rows:
            return

        header = _clean_header(rows[0], warnings)
        records = [_to_record(header, row, line, warnings) for line, row in enumerate(rows[1:], 2)]
        for option in (self.id_column, self.group_by):
            if option is not None and option not in header:
                raise ParsingError(f"column {option!r} not found; columns are {header}.")
        use_ids = self.id_column is not None and _ids_are_usable(
            [record[self.id_column] for record in records], self.id_column, warnings
        )

        for record in records:
            group = record[self.group_by] if self.group_by else ""
            yield ElementDraft(
                type="record",
                text=_record_text(record),
                data=record,
                element_id=record[self.id_column] if use_ids else None,
                section_path=[group] if group else [],
            )


def _decode(raw: bytes, warnings: list[str]) -> str:
    for encoding in ENCODINGS:
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        if encoding != ENCODINGS[0]:
            warnings.append(f"file is not UTF-8, decoded as {encoding}.")
        return text
    raise ParsingError(f"cannot decode the file with any of {ENCODINGS}.")


def _dialect(content: str, warnings: list[str]) -> type[csv.Dialect] | csv.Dialect:
    sample = "\n".join(content.splitlines()[:20])
    try:
        return csv.Sniffer().sniff(sample, delimiters=DELIMITERS)
    except csv.Error:
        warnings.append("delimiter could not be detected, ',' assumed.")
        return csv.excel


def _clean_header(raw_header: list[str], warnings: list[str]) -> list[str]:
    header: list[str] = []
    for position, raw_name in enumerate(raw_header, 1):
        name = normalize_text(raw_name) or f"column_{position}"
        if not raw_name.strip():
            warnings.append(f"column {position} has no name, called {name!r}.")
        if name in header:
            new_name = f"{name}_{position}"
            warnings.append(f"duplicate column {name!r} renamed {new_name!r}.")
            name = new_name
        header.append(name)
    return header


def _to_record(header: list[str], row: list[str], line: int, warnings: list[str]) -> dict[str, str]:
    """Map a row to its columns. Values are kept raw (only surrounding spaces are removed)."""

    if len(row) != len(header):
        warnings.append(f"line {line}: {len(row)} values for {len(header)} columns.")
    values = [cell.strip() for cell in row] + [""] * (len(header) - len(row))
    record = dict(zip(header, values, strict=False))
    for position, extra in enumerate(values[len(header) :], 1):
        record[f"extra_{position}"] = extra
    return record


def _ids_are_usable(ids: list[str], column: str, warnings: list[str]) -> bool:
    if all(ids) and len(set(ids)) == len(ids):
        return True
    warnings.append(f"column {column!r} has empty or duplicate values: generated ids used instead.")
    return False


def _record_text(record: dict[str, str]) -> str:
    """'column: value | column: value', empty values left out.

    The column names stay in the text on purpose: "2" alone means nothing,
    "max_days_per_week: 2" can be found by a question about remote days.
    """

    return " | ".join(f"{name}: {normalize_text(value)}" for name, value in record.items() if value)
