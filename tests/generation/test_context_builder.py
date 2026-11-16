"""Tests for the prompt-ready context built from retrieval results."""

import pytest
from app.rag.chunking.models import Chunk
from app.rag.generation import RetrievedChunk, build_context


def make_chunk(
    suffix: str,
    *,
    document_id: str = "HR-005",
    status: str = "active",
    kind: str = "section",
    part: int = 1,
    part_count: int = 1,
    effective_until: str | None = None,
    superseded_by: str | None = None,
) -> Chunk:
    version = "1.2"
    version_id = f"{document_id}-v{version}"
    chunk_suffix = suffix if part == 1 else f"{suffix}-{part}"
    prefix = f"{document_id} v{version} — Test policy > {suffix} Test section"
    content = f"Evidence from {suffix}, part {part}."
    section_number = suffix if kind == "section" else None
    return Chunk.model_validate(
        {
            "chunk_id": f"{version_id}#{chunk_suffix}",
            "kind": kind,
            "prefix": prefix,
            "content": content,
            "text": f"{prefix}\n{content}",
            "token_count": 20,
            "section_number": section_number,
            "section_path": [f"{suffix} Test section"],
            "part": part,
            "part_count": part_count,
            "pages": [3],
            "element_ids": [f"{version_id}#element-{suffix}-{part}"],
            "document": {
                "document_id": document_id,
                "version_id": version_id,
                "family_id": "test_policy",
                "title": "Test policy",
                "doc_type": "policy",
                "version": version,
                "lifecycle_status": status,
                "effective_from": "2026-01-01",
                "effective_until": effective_until,
                "superseded_by": superseded_by,
                "language": "fr",
                "category": "test",
                "department_scope": "all",
                "owner": "hr",
            },
            "source": {
                "relative_path": "test.pdf",
                "extension": ".pdf",
                "sha256": "a" * 64,
            },
            "chunking": {
                "chunker_version": "1.0",
                "tokenizer": "BAAI/bge-m3",
                "max_tokens": 512,
            },
        }
    )


def result(rank: int, chunk: Chunk) -> RetrievedChunk:
    return RetrievedChunk(rank=rank, score=1 - rank / 100, chunk=chunk)


def test_builds_numbered_context_and_trusted_source_mapping() -> None:
    first = make_chunk("4.2")
    second = make_chunk("6.1")

    context = build_context([result(1, first), result(2, second)])

    assert "--- SOURCE [S1] ---" in context.text
    assert "Document: HR-005 v1.2 — Test policy" in context.text
    assert "Status: active" in context.text
    assert "Effective from: 2026-01-01" in context.text
    assert "Effective until: ongoing" in context.text
    assert "Section: 4.2" in context.text
    assert "Page: 3" in context.text
    assert context.text.index("[S1]") < context.text.index("[S2]")
    assert [source.source_id for source in context.sources] == ["S1", "S2"]
    assert context.sources[0].chunk_id == "HR-005-v1.2#4.2"
    assert context.sources[0].source.model_dump() == {
        "document_id": "HR-005",
        "section": "4.2",
    }


def test_preserves_input_order_and_applies_the_limit() -> None:
    retrieved = [result(3, make_chunk("3")), result(1, make_chunk("1"))]

    context = build_context(retrieved, max_chunks=1)

    assert [source.rank for source in context.sources] == [3]
    assert [source.source.section for source in context.sources] == ["3"]
    assert "section 1" not in context.text


def test_archived_chunks_are_excluded_by_default_and_can_be_included() -> None:
    archived = result(
        1,
        make_chunk(
            "3.2",
            document_id="HR-029",
            status="archived",
            effective_until="2026-08-31",
            superseded_by="HR-011",
        ),
    )
    active = result(2, make_chunk("4.2"))

    current_context = build_context([archived, active])
    history_context = build_context([archived, active], include_archived=True)

    assert [source.chunk_id for source in current_context.sources] == ["HR-005-v1.2#4.2"]
    assert [source.source_id for source in current_context.sources] == ["S1"]
    assert [source.chunk_id for source in history_context.sources] == [
        "HR-029-v1.2#3.2",
        "HR-005-v1.2#4.2",
    ]
    assert history_context.sources[0].effective_until.isoformat() == "2026-08-31"
    assert history_context.sources[0].superseded_by == "HR-011"
    assert "Status: archived" in history_context.text
    assert "Effective until: 2026-08-31" in history_context.text
    assert "Superseded by: HR-011" in history_context.text


def test_split_parts_are_kept_and_map_to_the_same_section() -> None:
    first = make_chunk("18", part=1, part_count=2)
    second = make_chunk("18", part=2, part_count=2)

    context = build_context([result(1, first), result(2, second)])

    assert [source.chunk_id for source in context.sources] == [
        "HR-005-v1.2#18",
        "HR-005-v1.2#18-2",
    ]
    assert [source.source.section for source in context.sources] == ["18", "18"]


def test_record_uses_its_row_id_as_the_citable_section() -> None:
    record = make_chunk("RW-ENG-008", document_id="HR-013", kind="record")

    context = build_context([result(1, record)])

    assert context.sources[0].source.section == "RW-ENG-008"
    assert "Section: RW-ENG-008" in context.text


def test_empty_results_give_an_empty_context() -> None:
    context = build_context([])

    assert context.text == ""
    assert context.sources == []


def test_non_positive_limit_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        build_context([], max_chunks=0)
