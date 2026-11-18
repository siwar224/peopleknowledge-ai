"""Chunk every parsed document into datasets/processed/chunks/<version_id>.jsonl.

Input: datasets/processed/parsed/*.json (output of parse_documents.py).
Output: one .jsonl file per document version, one chunk per line.

Same behaviour as parse_documents.py: everything is chunked again and a file
is written only if its content changed (chunking is deterministic, so this is
always correct, whatever changed: parsed file, settings or chunker version);
one document failing never stops the others; exit code 1 on any failure;
outputs whose parsed file is gone are reported (``--prune`` deletes them).

Tokens are counted with the tokenizer of the embedding model (BAAI/bge-m3).
Only its tokenizer is loaded (a few MB), never the model itself.

Usage:
    python scripts/chunk_documents.py
    python scripts/chunk_documents.py --only HR-005          # document_id or version_id
    python scripts/chunk_documents.py --max-tokens 512 --prune
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path

from pydantic import ValidationError

from peopleknowledge.ingestion.chunking.chunker import (
    DEFAULT_MAX_TOKENS,
    ChunkingError,
    TokenCounter,
    chunk_document,
)
from peopleknowledge.ingestion.parsers.models import ParsedDocument
from peopleknowledge.ingestion.pipeline_io import (
    Outcome,
    Report,
    obsolete_outputs,
    print_report,
    write_text_if_changed,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARSED_DIR = PROJECT_ROOT / "datasets" / "processed" / "parsed"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "processed" / "chunks"
DEFAULT_TOKENIZER = "BAAI/bge-m3"


def load_token_counter(tokenizer_name: str) -> TokenCounter:
    """Token counter of a Hugging Face tokenizer (downloaded once, then cached)."""

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_pretrained(tokenizer_name)
    return lambda text: len(tokenizer.encode(text).ids)


def run(
    parsed_dir: Path,
    output_dir: Path,
    count_tokens: TokenCounter,
    tokenizer_name: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    only: Iterable[str] = (),
    prune: bool = False,
) -> Report:
    paths = sorted(parsed_dir.glob("*.json"))
    if not paths:
        raise ValueError(f"no parsed file in {parsed_dir} (run scripts/parse_documents.py first).")
    only = set(only)
    report = Report(pruned=prune)

    expected_outputs = {f"{path.stem}.jsonl" for path in paths}
    for path in paths:
        try:
            document = ParsedDocument.model_validate_json(path.read_text(encoding="utf-8"))
        except ValidationError as error:
            report.outcomes.append(Outcome(path.stem, "failed", f"invalid parsed file: {error}"))
            continue
        doc = document.document
        if only and not ({doc.document_id, doc.version_id} & only):
            continue
        try:
            chunks = chunk_document(document, count_tokens, tokenizer_name, max_tokens)
        except (ChunkingError, ValidationError) as error:
            report.outcomes.append(Outcome(doc.version_id, "failed", str(error)))
            continue

        content = "".join(chunk.to_json_line() for chunk in chunks)
        status = write_text_if_changed(output_dir / f"{doc.version_id}.jsonl", content)
        split = sum(1 for chunk in chunks if chunk.part == 2)
        detail = f"{len(chunks)} chunks, max {max(c.token_count for c in chunks)} tokens"
        if split:
            detail += f", {split} section(s) split"
        report.outcomes.append(Outcome(doc.version_id, status, detail))

    if only and not report.outcomes:
        report.outcomes.append(Outcome(", ".join(sorted(only)), "failed", "no such parsed document"))

    report.obsolete = obsolete_outputs(output_dir, "*.jsonl", expected_outputs, prune)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parsed-dir", type=Path, default=DEFAULT_PARSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER, help="Hugging Face model id.")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="ID",
        help="chunk only this document_id or version_id (repeatable).",
    )
    parser.add_argument(
        "--prune", action="store_true", help="delete outputs whose parsed file is gone."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, count_tokens: TokenCounter | None = None) -> int:
    """``count_tokens`` replaces the real tokenizer (used by the tests)."""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # never crash on a character the console lacks
    arguments = parse_args(argv)
    if arguments.max_tokens <= 0:
        print("ERROR: --max-tokens must be positive.")
        return 1
    if count_tokens is None:
        try:
            count_tokens = load_token_counter(arguments.tokenizer)
        except Exception as error:  # noqa: BLE001  (offline, unknown model...: any loading error)
            print(f"ERROR: cannot load the tokenizer {arguments.tokenizer!r}: {error}")
            return 1
    try:
        report = run(
            arguments.parsed_dir.resolve(),
            arguments.output_dir.resolve(),
            count_tokens,
            arguments.tokenizer,
            arguments.max_tokens,
            only=arguments.only,
            prune=arguments.prune,
        )
    except ValueError as error:  # no parsed file
        print(f"ERROR: {error}")
        return 1
    print_report(report, arguments.output_dir.resolve(), "Chunking")
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
