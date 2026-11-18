"""Tests of the chunk contract."""

import json

import pytest
from pydantic import ValidationError

from peopleknowledge.ingestion.chunking.models import Chunk

DOCUMENT = {
    "document_id": "HR-005",
    "version_id": "HR-005-v1.2",
    "family_id": "annual_leave_policy",
    "title": "Politique des congés annuels",
    "doc_type": "policy",
    "version": "1.2",
    "lifecycle_status": "active",
    "effective_from": "2026-01-01",
    "language": "fr",
    "category": "leave",
    "department_scope": "all",
    "owner": "people_finance",
}
PREFIX = (
    "HR-005 v1.2 — Politique des congés annuels > 4. Congés pendant la période d'essai"
    " > 4.2 Exception : jusqu'à 2 jours pendant l'essai"
)
CONTENT = "Par exception, un collaborateur en période d'essai peut poser jusqu'à 2 jours."


def make_chunk(**overrides: object) -> dict[str, object]:
    chunk: dict[str, object] = {
        "chunk_id": "HR-005-v1.2#4.2",
        "kind": "section",
        "prefix": PREFIX,
        "content": CONTENT,
        "text": f"{PREFIX}\n{CONTENT}",
        "token_count": 60,
        "section_number": "4.2",
        "section_path": [
            "Politique des congés annuels",
            "4. Congés pendant la période d'essai",
            "4.2 Exception : jusqu'à 2 jours pendant l'essai",
        ],
        "pages": [3],
        "element_ids": ["HR-005-v1.2#0039", "HR-005-v1.2#0040"],
        "document": DOCUMENT,
        "source": {"relative_path": "02/annual_leave_policy.pdf", "extension": ".pdf", "sha256": "a" * 64},
        "chunking": {"chunker_version": "1.0", "tokenizer": "BAAI/bge-m3", "max_tokens": 512},
    }
    chunk.update(overrides)
    return chunk


def test_valid_chunk_round_trips_through_a_json_line() -> None:
    chunk = Chunk.model_validate(make_chunk())

    line = chunk.to_json_line()

    assert line.endswith("\n") and line.count("\n") == 1  # exactly one line in the .jsonl
    assert "période" in line  # accents kept, not escaped
    assert Chunk.model_validate(json.loads(line)) == chunk


def test_metadata_needed_for_filtering_and_citing_is_in_the_chunk() -> None:
    chunk = Chunk.model_validate(make_chunk())

    assert chunk.document.lifecycle_status == "active"
    assert chunk.document.document_id == "HR-005"
    assert (chunk.section_number, chunk.pages) == ("4.2", [3])


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"text": CONTENT}, "prefix \\+ newline \\+ content"),
        ({"chunk_id": "HR-006-v1.1#4.2"}, "must start with 'HR-005-v1.2#'"),
        ({"part": 3, "part_count": 2}, "greater than part_count"),
        ({"token_count": 600}, "above the maximum of 512"),
        ({"element_ids": []}, "element_ids"),
        ({"element_ids": ["e1", "e1"]}, "duplicates"),
        ({"kind": "paragraph"}, "kind"),
        ({"unexpected": 1}, "unexpected"),
    ],
)
def test_invalid_chunk_is_rejected(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        Chunk.model_validate(make_chunk(**overrides))


def test_csv_row_chunk() -> None:
    prefix = "HR-013 v1.0 — Matrice d'éligibilité au télétravail > Engineering"
    content = "row_id: RW-ENG-008 | job_family: SRE (astreinte) | max_days_per_week: 3"
    document = {**DOCUMENT, "document_id": "HR-013", "version_id": "HR-013-v1.0", "version": "1.0"}

    chunk = Chunk.model_validate(
        make_chunk(
            chunk_id="HR-013-v1.0#RW-ENG-008",
            kind="record",
            prefix=prefix,
            content=content,
            text=f"{prefix}\n{content}",
            section_number=None,
            section_path=["Engineering"],
            pages=[],
            element_ids=["RW-ENG-008"],
            document=document,
        )
    )

    assert chunk.kind == "record"
    assert chunk.part == chunk.part_count == 1
