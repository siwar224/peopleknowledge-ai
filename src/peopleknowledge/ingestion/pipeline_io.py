"""Small helpers shared by the pipeline runners (scripts/parse_documents.py,
scripts/chunk_documents.py...): write only what changed, atomically, and
report what happened to each document.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile


@dataclass
class Outcome:
    version_id: str
    status: str  # new | updated | unchanged | skipped | failed
    detail: str = ""
    warnings: list[str] = field(default_factory=list)


@dataclass
class Report:
    outcomes: list[Outcome] = field(default_factory=list)
    obsolete: list[Path] = field(default_factory=list)
    pruned: bool = False

    def count(self, status: str) -> int:
        return sum(1 for outcome in self.outcomes if outcome.status == status)

    @property
    def failed(self) -> bool:
        return self.count("failed") > 0


def write_text_if_changed(path: Path, content: str) -> str:
    """Write ``content`` to ``path`` only if it differs; return "new", "updated" or "unchanged".

    The file is written to a temporary file first, then renamed: a crash never
    leaves a half-written output. Outputs are deterministic, so comparing the
    text is enough to know whether anything changed.
    """

    if path.exists():
        if path.read_text(encoding="utf-8") == content:
            return "unchanged"
        status = "updated"
    else:
        status = "new"

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n", dir=path.parent, suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return status


def obsolete_outputs(output_dir: Path, pattern: str, expected: set[str], prune: bool) -> list[Path]:
    """Outputs whose source is gone; deleted when ``prune`` is true."""

    if not output_dir.exists():
        return []
    obsolete = sorted(path for path in output_dir.glob(pattern) if path.name not in expected)
    if prune:
        for path in obsolete:
            path.unlink()
    return obsolete


def print_report(report: Report, output_dir: Path, label: str) -> None:
    for outcome in report.outcomes:
        line = f"  {outcome.status:<9} {outcome.version_id}"
        if outcome.detail:
            line += f"  ({outcome.detail})"
        print(line)
        for warning in outcome.warnings:
            print(f"              WARNING: {warning}")
    for path in report.obsolete:
        action = "deleted" if report.pruned else "use --prune to delete it"
        print(f"  obsolete  {path.name}  (its source is gone: {action})")

    counts = ", ".join(
        f"{status}={report.count(status)}"
        for status in ("new", "updated", "unchanged", "skipped", "failed")
    )
    warning_count = sum(len(outcome.warnings) for outcome in report.outcomes)
    print(f"{label}: {counts}, warnings={warning_count}.")
    print(f"Output: {output_dir}")
