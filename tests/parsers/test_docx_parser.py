"""DOCX parser, on documents built by the tests with python-docx."""

from pathlib import Path

import docx
import pytest
from app.rag.parsers import get_parser
from app.rag.parsers.docx_parser import DocxParser
from app.rag.parsers.models import DocumentMetadata
from docx.enum.style import WD_STYLE_TYPE

DOCUMENT = DocumentMetadata(
    document_id="T-002",
    version_id="T-002-v1.0",
    family_id="test",
    title="Test",
    doc_type="procedure",
    version="1.0",
    lifecycle_status="active",
    effective_from="2026-01-01",
    language="fr",
    category="test",
    department_scope="all",
    owner="test",
)


def build(tmp_path: Path, fill) -> Path:
    document = docx.Document()
    fill(document)
    path = tmp_path / "doc.docx"
    document.save(str(path))
    return path


def parse(path: Path, **options: object):
    return DocxParser(**options).parse(path, DOCUMENT, "doc.docx")


def bold_header_table(document, rows: list[list[str]]) -> None:
    table = document.add_table(rows=len(rows), cols=len(rows[0]))
    for r, values in enumerate(rows):
        for c, value in enumerate(values):
            run = table.cell(r, c).paragraphs[0].add_run(value)
            run.bold = r == 0


def test_registered_for_docx() -> None:
    assert isinstance(get_parser(".docx"), DocxParser)


def test_headings_paragraphs_and_section_paths(tmp_path: Path) -> None:
    def fill(document):
        document.add_paragraph("Guide de test", style="Title")
        document.add_heading("1. Règles", level=1)
        document.add_paragraph("Texte de la section 1.")
        document.add_heading("1.1 Détail", level=2)
        document.add_paragraph("Texte du détail.")
        document.add_heading("2. Autre", level=1)
        document.add_paragraph("")  # empty paragraphs are skipped
        document.add_paragraph("Texte de la section 2.")

    parsed = parse(build(tmp_path, fill))
    by_text = {e.text: e for e in parsed.elements}

    assert [e.type for e in parsed.elements] == [
        "heading", "heading", "paragraph", "heading", "paragraph", "heading", "paragraph"
    ]
    assert [by_text[t].level for t in ("Guide de test", "1. Règles", "1.1 Détail")] == [1, 2, 3]
    assert by_text["Texte du détail."].section_path == ["Guide de test", "1. Règles", "1.1 Détail"]
    assert by_text["Texte du détail."].section_number == "1.1"
    assert by_text["Texte de la section 2."].section_path == ["Guide de test", "2. Autre"]
    assert by_text["1.1 Détail"].section_path == ["Guide de test", "1. Règles"]
    assert all(e.page is None for e in parsed.elements)


def test_custom_style_based_on_a_heading_is_a_heading(tmp_path: Path) -> None:
    def fill(document):
        style = document.styles.add_style("Mon titre maison", WD_STYLE_TYPE.PARAGRAPH)
        style.base_style = document.styles["Heading 1"]
        document.add_paragraph("Section maison", style="Mon titre maison")
        document.add_paragraph("Contenu.")

    parsed = parse(build(tmp_path, fill))

    assert parsed.elements[0].type == "heading"
    assert parsed.elements[0].level == 2


def test_list_markers_drawn_by_word_are_reattached(tmp_path: Path) -> None:
    def fill(document):
        document.add_paragraph("premier point", style="List Bullet")
        document.add_paragraph("second point", style="List Bullet")
        document.add_paragraph("Entre deux listes.")
        document.add_paragraph("première étape", style="List Number")
        document.add_paragraph("deuxième étape", style="List Number")

    parsed = parse(build(tmp_path, fill))

    assert [(e.type, e.text) for e in parsed.elements] == [
        ("list_item", "• premier point"),
        ("list_item", "• second point"),
        ("paragraph", "Entre deux listes."),
        ("list_item", "1. première étape"),
        ("list_item", "2. deuxième étape"),
    ]


def test_steps_typed_as_text_are_list_items(tmp_path: Path) -> None:
    def fill(document):
        document.add_paragraph("1. Ouvrez l'application.")
        document.add_paragraph("2. Cliquez sur Valider.")

    parsed = parse(build(tmp_path, fill))

    assert [e.type for e in parsed.elements] == ["list_item", "list_item"]
    assert parsed.elements[0].text == "1. Ouvrez l'application."


def test_table_with_bold_header(tmp_path: Path) -> None:
    def fill(document):
        bold_header_table(document, [["Durée", "Délai"], ["1 jour", "3 jours"], ["5 jours", "10 jours"]])

    element = parse(build(tmp_path, fill)).elements[0]

    assert element.type == "table"
    assert element.data == {"header": ["Durée", "Délai"], "rows": [["1 jour", "3 jours"], ["5 jours", "10 jours"]]}
    assert element.text == "Durée: 1 jour | Délai: 3 jours\nDurée: 5 jours | Délai: 10 jours"


def test_table_without_header(tmp_path: Path) -> None:
    def fill(document):
        table = document.add_table(rows=2, cols=2)
        for r, (key, value) in enumerate([("Version", "1.0"), ("Statut", "Actif")]):
            table.cell(r, 0).text = key
            table.cell(r, 1).text = value

    element = parse(build(tmp_path, fill)).elements[0]

    assert element.data == {"header": None, "rows": [["Version", "1.0"], ["Statut", "Actif"]]}
    assert element.text == "Version | 1.0\nStatut | Actif"


def test_merged_cells_are_not_repeated(tmp_path: Path) -> None:
    def fill(document):
        table = document.add_table(rows=2, cols=3)
        merged = table.cell(0, 0).merge(table.cell(0, 2))
        merged.text = "Titre fusionné"
        for c, value in enumerate(["a", "b", "c"]):
            table.cell(1, c).text = value

    element = parse(build(tmp_path, fill)).elements[0]

    assert element.data["rows"] == [["Titre fusionné"], ["a", "b", "c"]]


def test_tables_keep_their_place_between_paragraphs(tmp_path: Path) -> None:
    def fill(document):
        document.add_paragraph("Avant.")
        bold_header_table(document, [["A", "B"], ["1", "2"]])
        document.add_paragraph("Après.")

    parsed = parse(build(tmp_path, fill))

    assert [e.type for e in parsed.elements] == ["paragraph", "table", "paragraph"]


def test_disclaimer_comes_from_options(tmp_path: Path) -> None:
    def fill(document):
        document.add_paragraph("Mention légale : document d'exemple.")
        document.add_paragraph("Contenu.")

    path = build(tmp_path, fill)

    assert parse(path).elements[0].type == "paragraph"
    assert parse(path, disclaimer_patterns=[r"^mention légale"]).elements[0].type == "disclaimer"


def test_headers_and_footers_are_left_out(tmp_path: Path) -> None:
    def fill(document):
        document.sections[0].header.paragraphs[0].text = "En-tête répété"
        document.sections[0].footer.paragraphs[0].text = "Pied de page"
        document.add_paragraph("Contenu.")

    parsed = parse(build(tmp_path, fill))

    assert [e.text for e in parsed.elements] == ["Contenu."]


def test_empty_document_is_an_error(tmp_path: Path) -> None:
    from app.rag.parsers.base import ParsingError

    with pytest.raises(ParsingError, match="no content extracted"):
        parse(build(tmp_path, lambda document: None))
