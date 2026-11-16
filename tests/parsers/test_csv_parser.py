"""CSV parser, on small synthetic files (nothing here depends on the real corpus)."""

from pathlib import Path

import pytest
from app.rag.parsers import get_parser
from app.rag.parsers.base import ParsingError
from app.rag.parsers.csv_parser import CsvParser
from app.rag.parsers.models import DocumentMetadata

DOCUMENT = DocumentMetadata(
    document_id="T-001",
    version_id="T-001-v1.0",
    family_id="test",
    title="Test",
    doc_type="matrix",
    version="1.0",
    lifecycle_status="active",
    effective_from="2026-01-01",
    language="fr",
    category="test",
    department_scope="all",
    owner="test",
)


def parse(tmp_path: Path, content: str | bytes, **options: object):
    path = tmp_path / "data.csv"
    if isinstance(content, str):
        # newline="": write line breaks as given (no "\r\n" conversion on Windows)
        path.write_text(content, encoding="utf-8", newline="")
    else:
        path.write_bytes(content)
    return CsvParser(**options).parse(path, DOCUMENT, "data.csv")


def test_registered_for_csv() -> None:
    assert isinstance(get_parser(".csv"), CsvParser)


def test_one_record_per_row_with_raw_values(tmp_path: Path) -> None:
    parsed = parse(tmp_path, "code,ville,jours\nA1,Tunis,2\nA2,Sfax,0\n")

    assert [e.type for e in parsed.elements] == ["record", "record"]
    assert parsed.elements[0].data == {"code": "A1", "ville": "Tunis", "jours": "2"}
    assert parsed.elements[0].text == "code: A1 | ville: Tunis | jours: 2"
    assert parsed.elements[0].element_id == "T-001-v1.0#0000"
    assert parsed.elements[0].page is None
    assert parsed.warnings == []


def test_empty_values_are_left_out_of_the_text_but_kept_in_data(tmp_path: Path) -> None:
    parsed = parse(tmp_path, "code,notes,jours\nA1,,2\n")

    assert parsed.elements[0].text == "code: A1 | jours: 2"
    assert parsed.elements[0].data == {"code": "A1", "notes": "", "jours": "2"}


def test_id_column_and_group_by(tmp_path: Path) -> None:
    parsed = parse(tmp_path, "id,team,x\nR-1,Ops,1\nR-2,Sales,2\n", id_column="id", group_by="team")

    assert [e.element_id for e in parsed.elements] == ["R-1", "R-2"]
    assert [e.section_path for e in parsed.elements] == [["Ops"], ["Sales"]]


def test_duplicate_ids_fall_back_to_generated_ids(tmp_path: Path) -> None:
    parsed = parse(tmp_path, "id,x\nR-1,1\nR-1,2\n", id_column="id")

    assert [e.element_id for e in parsed.elements] == ["T-001-v1.0#0000", "T-001-v1.0#0001"]
    assert "duplicate" in parsed.warnings[0]


def test_unknown_option_column_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ParsingError, match="column 'nope' not found"):
        parse(tmp_path, "id,x\nR-1,1\n", group_by="nope")


def test_semicolon_and_windows_encoding_are_detected(tmp_path: Path) -> None:
    content = "poste;télétravail\nDéveloppeur;oui\nAccueil;non\n".encode("cp1252")

    parsed = parse(tmp_path, content)

    assert parsed.elements[0].data == {"poste": "Développeur", "télétravail": "oui"}
    assert parsed.warnings == ["file is not UTF-8, decoded as cp1252."]


def test_utf8_bom_is_removed(tmp_path: Path) -> None:
    parsed = parse(tmp_path, "﻿code,x\nA,1\n")

    assert list(parsed.elements[0].data) == ["code", "x"]


def test_quoted_value_may_contain_comma_and_line_break(tmp_path: Path) -> None:
    parsed = parse(tmp_path, 'code,notes\nA,"un, deux\ntrois"\n')

    assert parsed.elements[0].data["notes"] == "un, deux\ntrois"
    assert parsed.elements[0].text == "code: A | notes: un, deux trois"


def test_irregular_rows_are_kept_and_reported(tmp_path: Path) -> None:
    parsed = parse(tmp_path, "a,b\n1\n1,2,3\n")

    assert parsed.elements[0].data == {"a": "1", "b": ""}
    assert parsed.elements[1].data == {"a": "1", "b": "2", "extra_1": "3"}
    assert sum("values for 2 columns" in warning for warning in parsed.warnings) == 2


def test_blank_and_duplicate_column_names(tmp_path: Path) -> None:
    parsed = parse(tmp_path, "a,,a\n1,2,3\n")

    assert list(parsed.elements[0].data) == ["a", "column_2", "a_3"]


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    parsed = parse(tmp_path, "a,b\n\n1,2\n,\n")

    assert len(parsed.elements) == 1


def test_header_only_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ParsingError, match="no content extracted"):
        parse(tmp_path, "a,b\n")
