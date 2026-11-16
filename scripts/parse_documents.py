"""Parse every document of the manifest into datasets/processed/parsed/<version_id>.json.

This script is the ONLY place that knows the manifest and the configuration.
For each document it:

1. picks the parser by extension (``get_parser``);
2. gives it its options: the corpus-wide ones from ``parsing_config.yaml``
   plus the document's own ones (``parsing`` key of its catalog entry,
   copied into the manifest by ``build_manifest.py``);
3. parses the file and writes the JSON, atomically, ONLY if it changed.

Why reparse everything instead of skipping "up-to-date" documents? A parsed
file depends on the source file, the parser version, the document metadata
(e.g. a policy becomes ``archived``) and the options. Tracking all of that is
error-prone; parsing is deterministic, so parsing again and comparing the
result is always right, and cheap at this corpus size. ``--only`` limits a run
to some documents.

One document failing never stops the others. The exit code is 1 if any
document failed, so that a later CI step can stop on it.

Usage:
    python scripts/parse_documents.py
    python scripts/parse_documents.py --only HR-005          # document_id or version_id
    python scripts/parse_documents.py --prune                # also delete obsolete outputs
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml
from app.rag.parsers import ParsingError, get_parser
from app.rag.parsers.models import DocumentMetadata, ParsedDocument
from app.rag.pipeline_io import (
    Outcome,
    Report,
    obsolete_outputs,
    print_report,
    write_text_if_changed,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "datasets" / "metadata" / "document_manifest.jsonl"
DEFAULT_CONFIG = PROJECT_ROOT / "datasets" / "metadata" / "parsing_config.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "processed" / "parsed"

# Corpus-wide options accepted in parsing_config.yaml (a typo must not be ignored).
CONFIG_KEYS = {"disclaimer_patterns"}


# --- Inputs -------------------------------------------------------------------------


def load_config(path: Path) -> dict[str, Any]:
    """Corpus-wide parser options. A missing file means "no option"."""

    if not path.exists():
        return {}
    content = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(content, dict):
        raise ValueError(f"{path.name}: expected a mapping of settings.")  # noqa: TRY004  (bad file content, not a bad argument)
    unknown = sorted(set(content) - CONFIG_KEYS)
    if unknown:
        raise ValueError(f"{path.name}: unknown setting(s) {unknown}; allowed: {sorted(CONFIG_KEYS)}.")
    patterns = content.get("disclaimer_patterns") or []
    if not isinstance(patterns, list) or not all(isinstance(p, str) for p in patterns):
        raise ValueError(f"{path.name}: disclaimer_patterns must be a list of texts.")
    for pattern in patterns:
        try:
            re.compile(pattern)
        except re.error as error:
            raise ValueError(f"{path.name}: invalid regex {pattern!r}: {error}.") from error
    return {"disclaimer_patterns": patterns}


def load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise ValueError(f"manifest not found: {path} (run scripts/build_manifest.py first).")
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def matches(record: dict[str, Any], only: Iterable[str]) -> bool:
    wanted = set(only)
    return not wanted or bool({record.get("document_id"), record.get("version_id")} & wanted)


# --- One document -------------------------------------------------------------------


def parse_record(record: dict[str, Any], config: dict[str, Any], root: Path) -> ParsedDocument:
    """Parse one manifest record. Raises ParsingError with a readable message."""

    if record.get("catalog_status") != "matched":
        raise ParsingError(
            f"{record['relative_path']} has no catalog entry: add it to document_catalog.yaml, "
            "then rerun build_manifest.py."
        )
    options = {**config, **(record.get("parsing") or {})}
    try:
        parser = get_parser(str(record["extension"]), **options)
    except TypeError as error:  # an option name the parser does not know
        raise ParsingError(f"invalid parsing options {sorted(options)}: {error}") from error

    document = parser.parse(
        root / str(record["source_path"]),
        DocumentMetadata(**record),
        str(record["relative_path"]),
    )
    if document.source.sha256 != record["sha256"]:
        raise ParsingError(
            "the file changed since the manifest was built: rerun build_manifest.py first."
        )
    return document


def write_if_changed(document: ParsedDocument, path: Path) -> str:
    """Write the JSON atomically; return "new", "updated" or "unchanged"."""

    return write_text_if_changed(path, document.to_json())


# --- Run ----------------------------------------------------------------------------


def run(
    manifest_path: Path,
    config_path: Path,
    output_dir: Path,
    only: Iterable[str] = (),
    prune: bool = False,
    root: Path = PROJECT_ROOT,
) -> Report:
    config = load_config(config_path)
    records = load_manifest(manifest_path)
    only = list(only)
    report = Report(pruned=prune)

    expected_outputs: set[str] = set()
    for record in records:
        version_id = str(record.get("version_id") or record.get("relative_path"))
        if record.get("is_duplicate"):
            if matches(record, only):
                detail = f"duplicate of {record.get('duplicate_of')}"
                report.outcomes.append(Outcome(version_id, "skipped", detail))
            continue
        expected_outputs.add(f"{version_id}.json")
        if not matches(record, only):
            continue
        try:
            document = parse_record(record, config, root)
        except ParsingError as error:
            report.outcomes.append(Outcome(version_id, "failed", str(error)))
            continue
        status = write_if_changed(document, output_dir / f"{version_id}.json")
        detail = f"{len(document.elements)} elements"
        report.outcomes.append(Outcome(version_id, status, detail, list(document.warnings)))

    if only and not report.outcomes:
        report.outcomes.append(Outcome(", ".join(only), "failed", "no such document in the manifest"))

    report.obsolete = obsolete_outputs(output_dir, "*.json", expected_outputs, prune)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="folder that the manifest's source_path values are relative to.",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="ID",
        help="parse only this document_id or version_id (repeatable).",
    )
    parser.add_argument(
        "--prune", action="store_true", help="delete outputs whose document left the manifest."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # never crash on a character the console lacks
    arguments = parse_args(argv)
    try:
        report = run(
            arguments.manifest.resolve(),
            arguments.config.resolve(),
            arguments.output_dir.resolve(),
            only=arguments.only,
            prune=arguments.prune,
            root=arguments.project_root.resolve(),
        )
    except ValueError as error:  # invalid configuration or missing manifest
        print(f"ERROR: {error}")
        return 1
    print_report(report, arguments.output_dir.resolve(), "Parsing")
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
