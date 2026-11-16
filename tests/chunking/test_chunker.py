"""Chunker tests on small synthetic parsed documents.

The token counter is fake (one token per word) so that the rules can be
checked with small numbers and without downloading a tokenizer.
"""

import pytest
from app.rag.chunking.chunker import ChunkingError, chunk_document
from app.rag.parsers.models import ParsedDocument

DOCUMENT = {
    "document_id": "HR-900",
    "version_id": "HR-900-v1.0",
    "family_id": "test",
    "title": "Politique de test",
    "doc_type": "policy",
    "version": "1.0",
    "lifecycle_status": "archived",
    "effective_from": "2025-01-01",
    "effective_until": "2026-08-31",
    "superseded_by": "HR-901",
    "language": "fr",
    "category": "test",
    "department_scope": "all",
    "owner": "hr",
}


def words(text: str) -> int:
    return len(text.split())


def build(*elements: dict) -> ParsedDocument:
    """Elements as dicts with at least type and text; ids, index and paths are filled in."""

    built = []
    for index, element in enumerate(elements):
        built.append({"element_id": f"HR-900-v1.0#{index:04d}", "index": index, "page": 1, **element})
    return ParsedDocument(
        document=DOCUMENT,
        source={"relative_path": "t.pdf", "extension": ".pdf", "sha256": "b" * 64},
        parsing={"parser": "pdf", "parser_version": "1"},
        elements=built,
    )


def heading(text: str, level: int, path: list[str], number: str | None = None) -> dict:
    return {"type": "heading", "text": text, "level": level, "section_path": path, "section_number": number}


def para(text: str, path: list[str], kind: str = "paragraph", page: int = 1) -> dict:
    return {"type": kind, "text": text, "section_path": path, "page": page}


TITLE = "Politique de test"
POLICY = build(
    heading(TITLE, 1, []),
    para("Sous-titre du document", [TITLE]),
    {"type": "table", "text": "Version | 1.0\nStatut | Archivé", "section_path": [TITLE], "data": {}},
    para("Document fictif.", [TITLE], kind="disclaimer"),
    heading("4. Essai", 2, [TITLE], "4"),
    heading("4.2 Exception", 3, [TITLE, "4. Essai"], "4.2"),
    para("Par exception, jusqu'à 2 jours, à deux conditions :", [TITLE, "4. Essai", "4.2 Exception"]),
    para("1 le manager valide ;", [TITLE, "4. Essai", "4.2 Exception"], kind="list_item"),
    para("2 l'équipe RH confirme.", [TITLE, "4. Essai", "4.2 Exception"], kind="list_item", page=2),
)


def chunks_of(document: ParsedDocument, max_tokens: int = 512):
    return chunk_document(document, words, "fake-words", max_tokens)


def test_document_card_and_sections() -> None:
    chunks = chunks_of(POLICY)

    assert [(c.chunk_id, c.kind) for c in chunks] == [
        ("HR-900-v1.0#card", "document_card"),
        ("HR-900-v1.0#4.2", "section"),  # "4. Essai" has no text of its own: no chunk
    ]
    card = chunks[0]
    assert card.prefix == "HR-900 v1.0 — Politique de test"
    assert card.content == "Sous-titre du document\nVersion | 1.0\nStatut | Archivé"


def test_section_chunk_prefix_content_and_traceability() -> None:
    section = chunks_of(POLICY)[1]

    # The title is in the head of the prefix: it is not repeated in the path.
    assert section.prefix == "HR-900 v1.0 — Politique de test > 4. Essai > 4.2 Exception"
    assert section.content == (
        "Par exception, jusqu'à 2 jours, à deux conditions :\n"
        "1 le manager valide ;\n"
        "2 l'équipe RH confirme."
    )
    assert section.text == f"{section.prefix}\n{section.content}"
    assert section.token_count == words(section.text)
    assert section.section_number == "4.2"
    assert section.section_path == [TITLE, "4. Essai", "4.2 Exception"]
    assert section.pages == [1, 2]
    assert section.element_ids == ["HR-900-v1.0#0006", "HR-900-v1.0#0007", "HR-900-v1.0#0008"]


def test_disclaimer_is_never_in_a_chunk() -> None:
    assert not any("Document fictif" in c.text for c in chunks_of(POLICY))


def test_all_document_metadata_is_copied() -> None:
    for chunk in chunks_of(POLICY):
        assert chunk.document.lifecycle_status == "archived"
        assert chunk.document.superseded_by == "HR-901"
        assert chunk.chunking.max_tokens == 512
        assert chunk.chunking.tokenizer == "fake-words"


def test_records_use_their_own_ids() -> None:
    document = build(
        {"type": "record", "text": "row_id: R-1 | team: Ops", "section_path": ["Ops"], "data": {},
         "element_id": "R-1", "page": None},
        {"type": "record", "text": "team: Sales", "section_path": [], "data": {}, "page": None},
    )

    chunks = chunks_of(document)

    assert [c.chunk_id for c in chunks] == ["HR-900-v1.0#R-1", "HR-900-v1.0#0001"]
    assert chunks[0].prefix == "HR-900 v1.0 — Politique de test > Ops"
    assert chunks[0].kind == "record" and chunks[0].pages == []


def test_long_section_is_split_between_elements() -> None:
    path = [TITLE, "3. Long"]
    paragraphs = [para(f"Paragraphe {n} " + "mot " * 8, path) for n in range(6)]  # 10 words each
    document = build(heading(TITLE, 1, []), heading("3. Long", 2, [TITLE], "3"), *paragraphs)

    chunks = chunks_of(document, max_tokens=30)

    assert [c.chunk_id for c in chunks] == ["HR-900-v1.0#3", "HR-900-v1.0#3-2", "HR-900-v1.0#3-3"]
    assert all(c.token_count <= 30 for c in chunks)
    assert all(c.prefix == "HR-900 v1.0 — Politique de test > 3. Long" for c in chunks)
    assert [(c.part, c.part_count) for c in chunks] == [(1, 3), (2, 3), (3, 3)]
    # nothing lost, nothing duplicated, order kept
    ids = [element_id for c in chunks for element_id in c.element_ids]
    assert ids == [f"HR-900-v1.0#{n:04d}" for n in range(2, 8)]


def test_big_table_is_split_between_rows() -> None:
    rows = [f"Code: HR-{n:03d} | Thème: sujet numéro {n}" for n in range(20)]  # 7 words per row
    table = {"type": "table", "text": "\n".join(rows), "section_path": [TITLE, "18. Index"], "data": {}}
    document = build(heading(TITLE, 1, []), heading("18. Index", 2, [TITLE], "18"), table)

    chunks = chunks_of(document, max_tokens=60)

    assert len(chunks) > 1
    assert all(c.token_count <= 60 for c in chunks)
    kept_rows = [row for c in chunks for row in c.content.splitlines()]
    assert kept_rows == rows  # every row whole, once, in order
    assert all(c.element_ids == ["HR-900-v1.0#0002"] for c in chunks)
    assert {c.section_number for c in chunks} == {"18"}


def test_long_paragraph_is_split_between_sentences_and_joined_with_spaces() -> None:
    sentences = [f"Phrase numéro {n} avec quelques mots." for n in range(8)]  # 6 words each
    paragraph = para(" ".join(sentences), [TITLE, "5. Texte"])
    document = build(heading(TITLE, 1, []), heading("5. Texte", 2, [TITLE], "5"), paragraph)

    chunks = chunks_of(document, max_tokens=30)

    assert len(chunks) > 1
    assert all(c.token_count <= 30 for c in chunks)
    assert " ".join(c.content for c in chunks) == paragraph["text"]  # no sentence cut or lost
    assert all("\n" not in c.content for c in chunks)  # pieces of one paragraph joined by spaces


def test_a_sentence_too_long_alone_is_split_between_words() -> None:
    paragraph = para(" ".join(f"mot{n}" for n in range(40)), [TITLE, "6. Bloc"])
    document = build(heading(TITLE, 1, []), heading("6. Bloc", 2, [TITLE], "6"), paragraph)

    chunks = chunks_of(document, max_tokens=20)

    assert all(c.token_count <= 20 for c in chunks)
    assert " ".join(c.content for c in chunks) == paragraph["text"]


def test_prefix_too_long_is_an_error() -> None:
    with pytest.raises(ChunkingError, match="prefix alone"):
        chunks_of(POLICY, max_tokens=5)


def test_document_with_numbered_level_1_sections_has_no_card() -> None:
    # Only one heading size: "1. Objet" is level 1, but it is a section, not a title.
    document = build(
        heading("1. Objet", 1, [], "1"),
        para("Texte un.", ["1. Objet"]),
        heading("2. Règles", 1, [], "2"),
        para("Texte deux.", ["2. Règles"]),
    )

    chunks = chunks_of(document)

    assert [c.chunk_id for c in chunks] == ["HR-900-v1.0#1", "HR-900-v1.0#2"]
    assert chunks[0].prefix == "HR-900 v1.0 — Politique de test > 1. Objet"


def test_text_before_any_heading_goes_to_the_card() -> None:
    document = build(para("Introduction sans titre.", []), heading("1. A", 1, [], "1"), para("x", ["1. A"]))

    chunks = chunks_of(document)

    assert [(c.chunk_id, c.content) for c in chunks] == [
        ("HR-900-v1.0#card", "Introduction sans titre."),
        ("HR-900-v1.0#1", "x"),
    ]


def test_duplicate_section_numbers_get_distinct_ids() -> None:
    document = build(
        heading("1. A", 1, [], "1"), para("a", ["1. A"]),
        heading("1. B", 1, [], "1"), para("b", ["1. B"]),
    )

    ids = [c.chunk_id for c in chunks_of(document)]

    assert ids == ["HR-900-v1.0#1", "HR-900-v1.0#1@0003"]


def test_chunking_is_deterministic() -> None:
    first = [c.to_json_line() for c in chunks_of(POLICY)]

    assert [c.to_json_line() for c in chunks_of(POLICY)] == first
