"""Measure every section of the parsed corpus in tokens of the embedding model.

Reads datasets/processed/parsed/*.json (output of parse_documents.py), writes
nothing (except an optional CSV). Its only goal is to choose the minimum and
maximum chunk sizes from real numbers instead of guessing.

What is a "unit" here (a candidate chunk, before any chunking decision):
- a SECTION: a heading and everything below it until the next heading
  (paragraphs, lists, tables). A heading directly followed by a sub-heading
  has no text of its own: it is not counted as a unit, it only lives in the
  prefix of its sub-sections;
- a RECORD: one CSV row, already self-contained;
- disclaimers are left out (they will be excluded from the chunks).

Sizes are counted with the tokenizer of the embedding model (BAAI/bge-m3 by
default), special tokens included, because that is what counts against the
model's input limit. The breadcrumb prefix that chunking will add
("HR-005 v1.2 — Title > 4. ... > 4.2 ...") is measured separately.

Usage:
    python scripts/measure_sections.py
    python scripts/measure_sections.py --csv datasets/processed/section_tokens.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from app.rag.parsers.models import ParsedDocument, ParsedElement

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARSED_DIR = PROJECT_ROOT / "datasets" / "processed" / "parsed"
DEFAULT_MODEL = "BAAI/bge-m3"

# (upper bound excluded, label). 512 = input limit of many embedding models
# (e.g. multilingual-e5); bge-m3 accepts 8192.
BUCKETS = [
    (50, "< 50"),
    (200, "50-199"),
    (512, "200-511"),
    (1024, "512-1023"),
    (float("inf"), ">= 1024"),
]


@dataclass
class Unit:
    version_id: str
    kind: str  # "section" | "record"
    label: str  # "4.2 Exception : ..." or "RW-ENG-008"
    prefix: str  # breadcrumb that chunking will put before the text
    text: str
    pages: list[int]
    tables: int = 0
    tokens: int = 0
    prefix_tokens: int = 0


def units_of(document: ParsedDocument) -> list[Unit]:
    """Cut a parsed document into sections and records (see module docstring)."""

    doc = document.document
    head = f"{doc.document_id} v{doc.version} — {doc.title}"
    units: list[Unit] = []
    current: list[ParsedElement] = []

    def close() -> None:
        content = [e for e in current if e.type != "heading"]
        if not content:  # a heading without text of its own
            return
        opener = current[0] if current[0].type == "heading" else None
        path = current[0].section_path + ([opener.text] if opener else [])
        units.append(
            Unit(
                version_id=doc.version_id,
                kind="section",
                label=opener.text if opener else "(before the first heading)",
                prefix=" > ".join([head, *path]),
                text="\n".join(e.text for e in current),
                pages=sorted({e.page for e in current if e.page is not None}),
                tables=sum(1 for e in content if e.type == "table"),
            )
        )

    for element in document.elements:
        if element.type == "disclaimer":
            continue
        if element.type == "record":
            units.append(
                Unit(
                    version_id=doc.version_id,
                    kind="record",
                    label=element.element_id,
                    prefix=" > ".join([head, *element.section_path]),
                    text=element.text,
                    pages=[],
                )
            )
            continue
        if element.type == "heading":
            close()
            current = [element]
        else:
            current.append(element)
    close()
    return units


def measure(units: list[Unit], count_tokens: Callable[[str], int]) -> None:
    for unit in units:
        unit.tokens = count_tokens(unit.text)
        unit.prefix_tokens = count_tokens(unit.prefix)


def bucket(tokens: int) -> str:
    return next(label for limit, label in BUCKETS if tokens < limit)


def load_documents(parsed_dir: Path) -> list[ParsedDocument]:
    paths = sorted(parsed_dir.glob("*.json"))
    if not paths:
        raise ValueError(f"no parsed file in {parsed_dir} (run scripts/parse_documents.py first).")
    return [ParsedDocument.model_validate_json(p.read_text(encoding="utf-8")) for p in paths]


# --- Report ---------------------------------------------------------------------------


def stats_line(name: str, units: list[Unit]) -> str:
    sizes = [unit.tokens for unit in units]
    counts = [sum(1 for size in sizes if bucket(size) == label) for _, label in BUCKETS]
    cells = "".join(f"{count:>10}" for count in counts)
    return (
        f"{name:<14}{len(units):>6}{min(sizes):>7}{int(median(sizes)):>8}{max(sizes):>7}{cells}"
    )


def print_report(units: list[Unit], top: int) -> None:
    header_buckets = "".join(f"{label:>10}" for _, label in BUCKETS)
    sections = [unit for unit in units if unit.kind == "section"]
    records = [unit for unit in units if unit.kind == "record"]

    print("\nSIZE OF EACH UNIT, in tokens (text only, without the prefix)\n")
    print(f"{'document':<14}{'units':>6}{'min':>7}{'median':>8}{'max':>7}{header_buckets}")
    for version_id in sorted({unit.version_id for unit in units}):
        print(stats_line(version_id, [u for u in units if u.version_id == version_id]))
    print("-" * (42 + 10 * len(BUCKETS)))
    if sections:
        print(stats_line("all sections", sections))
    if records:
        print(stats_line("all records", records))

    print(f"\nLONGEST {top} SECTIONS")
    for unit in sorted(sections, key=lambda u: -u.tokens)[:top]:
        pages = f"p.{unit.pages[0]}-{unit.pages[-1]}" if unit.pages else "-"
        tables = f", {unit.tables} table(s)" if unit.tables else ""
        print(f"  {unit.tokens:>6}  {unit.version_id:<12} {pages:<8} {unit.label[:60]}{tables}")

    print(f"\nSHORTEST {top} SECTIONS")
    for unit in sorted(sections, key=lambda u: u.tokens)[:top]:
        print(f"  {unit.tokens:>6}  {unit.version_id:<12} {unit.label[:50]}  | {unit.text[:60]!r}")

    total = sum(unit.tokens for unit in units)
    prefixes = [unit.prefix_tokens for unit in units]
    characters = sum(len(unit.text) for unit in units)
    words = sum(len(unit.text.split()) for unit in units)
    print("\nWHOLE CORPUS")
    print(f"  units: {len(sections)} sections + {len(records)} records = {len(units)}")
    print(f"  tokens (text only): {total}")
    print(f"  prefix: median {int(median(prefixes))} tokens, max {max(prefixes)} tokens per unit")
    print(f"  ratio: {total / words:.2f} tokens per word, {characters / total:.2f} characters per token")


def write_csv(units: list[Unit], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["version_id", "kind", "label", "pages", "tables", "tokens", "prefix_tokens"])
        for u in units:
            pages = " ".join(map(str, u.pages))
            writer.writerow([u.version_id, u.kind, u.label, pages, u.tables, u.tokens, u.prefix_tokens])


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parsed-dir", type=Path, default=DEFAULT_PARSED_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id.")
    parser.add_argument("--top", type=int, default=10, help="sections listed as longest/shortest")
    parser.add_argument("--csv", type=Path, help="also write every unit to this CSV file")
    arguments = parser.parse_args(argv)

    try:
        documents = load_documents(arguments.parsed_dir)
    except ValueError as error:
        print(f"ERROR: {error}")
        return 1

    from tokenizers import (
        Tokenizer,  # downloads tokenizer.json only (a few MB), not the model
    )

    tokenizer = Tokenizer.from_pretrained(arguments.model)
    units = [unit for document in documents for unit in units_of(document)]
    measure(units, lambda text: len(tokenizer.encode(text).ids))

    print(f"Tokenizer: {arguments.model}  |  {len(documents)} parsed documents")
    print_report(units, arguments.top)
    if arguments.csv:
        write_csv(units, arguments.csv)
        print(f"\nCSV: {arguments.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
