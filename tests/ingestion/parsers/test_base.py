"""Tests of the common parser machinery, with a tiny fake parser for .txt files."""

import hashlib
from collections.abc import Iterator
from pathlib import Path

import pytest
from app.rag.parsers import base
from app.rag.parsers.base import (
    BaseParser,
    ElementDraft,
    ParsingError,
    UnsupportedFormatError,
    get_parser,
    register_parser,
    supported_extensions,
)
from app.rag.parsers.models import DocumentMetadata

DOCUMENT = DocumentMetadata(
    document_id="HR-999",
    version_id="HR-999-v1.0",
    family_id="test_family",
    title="Document de test",
    doc_type="policy",
    version="1.0",
    lifecycle_status="active",
    effective_from="2026-01-01",
    language="fr",
    category="leave",
    department_scope="all",
    owner="people_finance",
)


class TxtParser(BaseParser):
    """Fake parser: one non-blank line = one paragraph; 'WARN' lines add a warning."""

    name = "txt"
    version = "0.1"
    extensions = frozenset({".txt"})

    def extract(self, path: Path, warnings: list[str]) -> Iterator[ElementDraft]:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line == "WARN":
                warnings.append("a WARN line was found")
            elif line.startswith("ID "):
                _, element_id, text = line.split(" ", 2)
                yield ElementDraft(type="paragraph", text=text, element_id=element_id)
            elif line == "BROKEN":
                raise RuntimeError("cannot decode this line")
            elif line == "BAD_HEADING":
                yield ElementDraft(type="heading", text="Titre sans niveau")  # level missing
            elif line.strip():
                yield ElementDraft(type="paragraph", text=line)


@pytest.fixture(autouse=True)
def empty_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test starts with its own registry, so tests never leak parsers."""

    monkeypatch.setattr(base, "_REGISTRY", {})


def write(tmp_path: Path, content: str, name: str = "doc.txt") -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_parse_numbers_elements_and_fills_document(tmp_path: Path) -> None:
    path = write(tmp_path, "Premier paragraphe.\n\nDeuxième paragraphe.\n")

    parsed = TxtParser().parse(path, DOCUMENT, "01_test/doc.txt")

    assert [e.index for e in parsed.elements] == [0, 1]
    assert [e.element_id for e in parsed.elements] == ["HR-999-v1.0#0000", "HR-999-v1.0#0001"]
    assert parsed.elements[1].text == "Deuxième paragraphe."
    assert parsed.document.document_id == "HR-999"
    assert parsed.parsing.parser == "txt"
    assert parsed.parsing.parser_version == "0.1"
    assert parsed.source.relative_path == "01_test/doc.txt"
    assert parsed.source.extension == ".txt"
    assert parsed.source.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_parse_is_deterministic(tmp_path: Path) -> None:
    path = write(tmp_path, "Un.\nDeux.\n")

    first = TxtParser().parse(path, DOCUMENT, "doc.txt").to_json()
    second = TxtParser().parse(path, DOCUMENT, "doc.txt").to_json()

    assert first == second


def test_parser_can_keep_its_own_element_id(tmp_path: Path) -> None:
    path = write(tmp_path, "ID RW-ENG-001 Ligne avec identifiant.\nSans identifiant.\n")

    parsed = TxtParser().parse(path, DOCUMENT, "doc.txt")

    assert [e.element_id for e in parsed.elements] == ["RW-ENG-001", "HR-999-v1.0#0001"]


def test_warnings_are_kept(tmp_path: Path) -> None:
    path = write(tmp_path, "Texte.\nWARN\n")

    parsed = TxtParser().parse(path, DOCUMENT, "doc.txt")

    assert parsed.warnings == ["a WARN line was found"]


def test_uppercase_extension_is_accepted(tmp_path: Path) -> None:
    path = write(tmp_path, "Texte.\n", name="DOC.TXT")

    assert TxtParser().parse(path, DOCUMENT, "DOC.TXT").source.extension == ".txt"


def test_wrong_extension_is_refused(tmp_path: Path) -> None:
    path = write(tmp_path, "Texte.\n", name="doc.md")

    with pytest.raises(UnsupportedFormatError, match="cannot read '.md'"):
        TxtParser().parse(path, DOCUMENT, "doc.md")


def test_missing_file_is_a_parsing_error(tmp_path: Path) -> None:
    with pytest.raises(ParsingError, match="file not found"):
        TxtParser().parse(tmp_path / "absent.txt", DOCUMENT, "absent.txt")


def test_empty_file_is_a_parsing_error(tmp_path: Path) -> None:
    path = write(tmp_path, "\n\n")

    with pytest.raises(ParsingError, match="no content extracted"):
        TxtParser().parse(path, DOCUMENT, "doc.txt")


def test_unexpected_error_becomes_a_parsing_error(tmp_path: Path) -> None:
    path = write(tmp_path, "Texte.\nBROKEN\n")

    with pytest.raises(ParsingError, match="txt parser failed on doc.txt"):
        TxtParser().parse(path, DOCUMENT, "doc.txt")


def test_output_refused_by_the_contract_becomes_a_parsing_error(tmp_path: Path) -> None:
    path = write(tmp_path, "BAD_HEADING\n")

    with pytest.raises(ParsingError, match="invalid output"):
        TxtParser().parse(path, DOCUMENT, "doc.txt")


def test_registry_finds_parser_by_extension() -> None:
    register_parser(TxtParser)

    assert isinstance(get_parser(".txt"), TxtParser)
    assert isinstance(get_parser(".TXT"), TxtParser)
    assert isinstance(get_parser("txt"), TxtParser)
    assert supported_extensions() == [".txt"]


def test_unknown_extension_is_refused() -> None:
    with pytest.raises(UnsupportedFormatError, match="no parser for '.xlsx'"):
        get_parser(".xlsx")


def test_extension_cannot_be_registered_twice() -> None:
    register_parser(TxtParser)

    class OtherTxtParser(TxtParser):
        name = "other_txt"

    with pytest.raises(ValueError, match="already handled by TxtParser"):
        register_parser(OtherTxtParser)


def test_incomplete_parser_cannot_be_registered() -> None:
    class NoVersionParser(TxtParser):
        version = ""

    with pytest.raises(TypeError, match="non-empty 'version'"):
        register_parser(NoVersionParser)


def test_extension_must_be_lowercase_with_dot() -> None:
    class BadExtensionParser(TxtParser):
        extensions = frozenset({"TXT"})

    with pytest.raises(TypeError, match="invalid extension"):
        register_parser(BadExtensionParser)
