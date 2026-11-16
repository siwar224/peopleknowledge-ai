"""How measure_sections.py cuts a parsed document into units (no tokenizer needed)."""

from app.rag.parsers.models import ParsedDocument

from scripts.measure_sections import bucket, measure, units_of

DOCUMENT = {
    "document_id": "HR-005",
    "version_id": "HR-005-v1.2",
    "family_id": "leave",
    "title": "Politique des congés",
    "doc_type": "policy",
    "version": "1.2",
    "lifecycle_status": "active",
    "effective_from": "2026-01-01",
    "language": "fr",
    "category": "leave",
    "department_scope": "all",
    "owner": "hr",
}
SOURCE = {"relative_path": "a.pdf", "extension": ".pdf", "sha256": "a" * 64}


def document(*elements: tuple) -> ParsedDocument:
    """elements: (type, text, section_path, level) tuples, in reading order."""

    built = []
    for index, (kind, text, path, level) in enumerate(elements):
        element = {
            "element_id": f"e{index}",
            "index": index,
            "type": kind,
            "text": text,
            "page": 1 + index // 3,
            "section_path": path,
            "level": level,
        }
        if kind in ("table", "record"):
            element["data"] = {}
        built.append(element)
    return ParsedDocument(
        document=DOCUMENT,
        source=SOURCE,
        parsing={"parser": "pdf", "parser_version": "1"},
        elements=built,
    )


def test_a_section_is_a_heading_with_everything_below_it() -> None:
    parsed = document(
        ("heading", "4. Essai", [], 2),
        ("heading", "4.2 Exception", ["4. Essai"], 3),
        ("paragraph", "Jusqu'à 2 jours.", ["4. Essai", "4.2 Exception"], None),
        ("list_item", "1 le manager valide", ["4. Essai", "4.2 Exception"], None),
        ("table", "Durée: 2 jours", ["4. Essai", "4.2 Exception"], None),
        ("heading", "4.3 Après", ["4. Essai"], 3),
        ("paragraph", "Sans limite.", ["4. Essai", "4.3 Après"], None),
    )

    units = units_of(parsed)

    # "4. Essai" has no text of its own: it only appears in the prefixes.
    assert [u.label for u in units] == ["4.2 Exception", "4.3 Après"]
    first = units[0]
    assert first.text == "4.2 Exception\nJusqu'à 2 jours.\n1 le manager valide\nDurée: 2 jours"
    assert first.prefix == "HR-005 v1.2 — Politique des congés > 4. Essai > 4.2 Exception"
    assert first.tables == 1
    assert first.pages == [1, 2]


def test_disclaimer_is_left_out_and_text_before_any_heading_is_kept() -> None:
    parsed = document(
        ("paragraph", "Sous-titre", [], None),
        ("disclaimer", "Document fictif.", [], None),
        ("heading", "1. Objet", [], 2),
        ("paragraph", "Texte.", ["1. Objet"], None),
    )

    units = units_of(parsed)

    assert [u.label for u in units] == ["(before the first heading)", "1. Objet"]
    assert "Document fictif" not in " ".join(u.text for u in units)


def test_each_record_is_its_own_unit() -> None:
    parsed = document(
        ("record", "row_id: R-1 | team: Ops", ["Ops"], None),
        ("record", "row_id: R-2 | team: Sales", ["Sales"], None),
    )

    units = units_of(parsed)

    assert [(u.kind, u.prefix) for u in units] == [
        ("record", "HR-005 v1.2 — Politique des congés > Ops"),
        ("record", "HR-005 v1.2 — Politique des congés > Sales"),
    ]


def test_measure_counts_text_and_prefix_separately() -> None:
    units = units_of(
        document(("heading", "1. Objet", [], 2), ("paragraph", "un deux trois", ["1. Objet"], None))
    )

    measure(units, lambda text: len(text.split()))  # fake tokenizer: one token per word

    assert units[0].tokens == 5  # "1. Objet\nun deux trois"
    assert units[0].prefix_tokens == len(units[0].prefix.split())


def test_buckets() -> None:
    assert [bucket(n) for n in (0, 49, 50, 511, 512, 1024)] == [
        "< 50", "< 50", "50-199", "200-511", "512-1023", ">= 1024"
    ]
