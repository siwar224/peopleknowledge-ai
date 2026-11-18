"""The chunking decisions, checked on the real batch 1 corpus with the real tokenizer.

Skipped when the parsed corpus is absent (run scripts/parse_documents.py) or
when the bge-m3 tokenizer cannot be loaded (first use needs the network).
"""

from collections import Counter
from functools import cache
from pathlib import Path

import pytest

from peopleknowledge.ingestion.chunking.chunker import chunk_document
from peopleknowledge.ingestion.chunking.models import Chunk
from peopleknowledge.ingestion.parsers.models import ParsedDocument

PARSED_DIR = Path(__file__).resolve().parents[3] / "datasets" / "processed" / "parsed"
TOKENIZER = "BAAI/bge-m3"
MAX_TOKENS = 512

pytestmark = pytest.mark.skipif(
    not any(PARSED_DIR.glob("*.json")), reason="parsed corpus not present"
)


@cache
def all_chunks() -> dict[str, list[Chunk]]:
    try:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_pretrained(TOKENIZER)
    except Exception as error:  # noqa: BLE001  (offline and not cached: any loading error)
        pytest.skip(f"tokenizer not available: {error}")
    result = {}
    for path in sorted(PARSED_DIR.glob("*.json")):
        document = ParsedDocument.model_validate_json(path.read_text(encoding="utf-8"))
        chunks = chunk_document(document, lambda t: len(tokenizer.encode(t).ids), TOKENIZER)
        result[document.document.version_id] = chunks
    return result


def chunk(chunk_id: str) -> Chunk:
    version_id = chunk_id.split("#")[0]
    return next(c for c in all_chunks()[version_id] if c.chunk_id == chunk_id)


def test_no_chunk_above_the_maximum_and_ids_are_unique() -> None:
    chunks = [c for chunks in all_chunks().values() for c in chunks]

    assert max(c.token_count for c in chunks) <= MAX_TOKENS
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_probation_exception_is_one_whole_chunk() -> None:
    exception = chunk("HR-005-v1.2#4.2")

    assert exception.part_count == 1
    assert "jusqu'à 2 jours" in exception.content
    assert "1 le manager donne son accord" in exception.content  # the conditions stay with the rule
    assert "2 l'équipe RH valide la demande" in exception.content
    assert exception.prefix.endswith("> 4. Congés pendant la période d'essai > " + exception.section_path[-1])


def test_csv_rows_are_chunks_with_their_own_ids() -> None:
    sre = chunk("HR-013-v1.0#RW-ENG-008")

    assert sre.kind == "record"
    assert "max_days_per_week: 3" in sre.content
    assert sre.prefix.endswith("> Engineering")
    assert len(all_chunks()["HR-013-v1.0"]) == 38


def test_archived_version_is_chunked_and_marked() -> None:
    v1 = all_chunks()["HR-029-v1.0"]

    assert v1 and all(c.document.lifecycle_status == "archived" for c in v1)
    assert any("Le lundi est exclu" in c.content for c in v1)
    assert all(c.document.lifecycle_status == "active" for c in all_chunks()["HR-011-v2.0"])


def test_disclaimer_is_left_out_and_each_text_document_has_a_card() -> None:
    for version_id, chunks in all_chunks().items():
        assert not any(c.content.startswith("Document fictif") for c in chunks), version_id
        kinds = Counter(c.kind for c in chunks)
        assert kinds["document_card"] == (0 if version_id == "HR-013-v1.0" else 1), version_id


def test_only_the_index_table_is_split_and_no_row_is_lost() -> None:
    split = [c.chunk_id for chunks in all_chunks().values() for c in chunks if c.part_count > 1]
    index = [c for c in all_chunks()["HR-001-v3.0"] if c.section_number == "18"]

    assert split == ["HR-001-v3.0#18", "HR-001-v3.0#18-2"]
    rows = [line for c in index for line in c.content.splitlines() if line.startswith("Code: ")]
    assert len(rows) == len(set(rows)) == 25
