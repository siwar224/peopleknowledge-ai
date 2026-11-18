import json

import pytest
from app.rag.parsers.models import (
    DocumentMetadata,
    ParsedDocument,
    ParsedElement,
    ParsingInfo,
)
from pydantic import ValidationError

MANIFEST_RECORD = {
    # Document fields (kept)
    "document_id": "HR-005",
    "version_id": "HR-005-v1.2",
    "family_id": "annual_leave_policy",
    "title": "Politique des congés annuels",
    "doc_type": "policy",
    "version": "1.2",
    "lifecycle_status": "active",
    "effective_from": "2026-01-01",
    "effective_until": None,
    "supersedes": None,
    "superseded_by": None,
    "language": "fr",
    "category": "leave",
    "department_scope": "all",
    "owner": "people_finance",
    "relative_path": "02_leave_and_work/annual_leave_policy.pdf",
    "extension": ".pdf",
    "sha256": "a" * 64,
    # Scan fields (dropped)
    "change_status": "unchanged",
    "is_duplicate": False,
    "identity_resolution": "catalog",
}


def make_element(index: int, **overrides: object) -> dict[str, object]:
    element: dict[str, object] = {
        "element_id": f"HR-005-v1.2#{index:04d}",
        "index": index,
        "type": "paragraph",
        "text": "Par exception, un collaborateur en période d'essai peut poser jusqu'à 2 jours.",
        "page": 3,
        "section_path": ["4. Congés pendant la période d'essai", "4.2 Exception"],
        "section_number": "4.2",
    }
    element.update(overrides)
    return element


def make_document(elements: list[dict[str, object]]) -> ParsedDocument:
    return ParsedDocument(
        document=MANIFEST_RECORD,
        source=MANIFEST_RECORD,
        parsing={"parser": "pdf", "parser_version": "0.1.0"},
        elements=elements,
    )


def test_manifest_record_keeps_document_fields_and_drops_scan_fields() -> None:
    metadata = DocumentMetadata.model_validate(MANIFEST_RECORD)

    assert metadata.document_id == "HR-005"
    assert metadata.effective_from.isoformat() == "2026-01-01"
    assert "change_status" not in metadata.model_dump()


def test_valid_document_round_trips_through_json() -> None:
    document = make_document(
        [
            make_element(0, type="heading", level=2, text="4.2 Exception", page=3),
            make_element(1),
        ]
    )

    payload = document.to_json()

    assert "période d'essai" in payload  # accents are not escaped
    assert ParsedDocument.model_validate(json.loads(payload)) == document


@pytest.mark.parametrize(
    "overrides",
    [
        {"type": "titre"},  # unknown type
        {"text": "   "},  # blank text
        {"type": "heading"},  # heading without level
        {"level": 2},  # level on a non-heading
        {"type": "table"},  # table without data
        {"type": "record", "data": None},  # record without data
        {"section_number": "§4.2"},  # badly formed section number
        {"page": 0},  # pages start at 1
        {"unexpected": "field"},  # contract is closed
    ],
)
def test_invalid_element_is_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ParsedElement.model_validate(make_element(0, **overrides))


def test_elements_must_be_in_order() -> None:
    with pytest.raises(ValidationError, match="expected 1"):
        make_document([make_element(0), make_element(2)])


def test_element_ids_must_be_unique() -> None:
    with pytest.raises(ValidationError, match="duplicate element_id"):
        make_document([make_element(0), make_element(1, element_id="HR-005-v1.2#0000")])


def test_csv_record_keeps_raw_values() -> None:
    element = ParsedElement(
        element_id="RW-ENG-001",
        index=0,
        type="record",
        text="department: Engineering | max_days_per_week: 2 | remote_eligible: oui",
        data={"row_id": "RW-ENG-001", "max_days_per_week": "2", "remote_eligible": "oui"},
    )

    assert element.data == {"row_id": "RW-ENG-001", "max_days_per_week": "2", "remote_eligible": "oui"}
    assert element.page is None
    assert element.section_path == []


def test_parser_name_is_open_to_new_formats() -> None:
    # A new format must not require changing the contract.
    info = ParsingInfo(parser="xlsx", parser_version="1.0")

    assert info.parser == "xlsx"


def test_parser_name_must_not_be_empty() -> None:
    with pytest.raises(ValidationError):
        ParsingInfo(parser="", parser_version="1.0")
