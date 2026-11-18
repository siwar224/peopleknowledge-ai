"""Turn ranked chunks into delimited evidence that an LLM can cite safely."""

from __future__ import annotations

from collections.abc import Sequence

from ..evaluation.gold import Source, source_of_chunk
from ..ingestion.chunking.models import Chunk
from .models import BuiltContext, ContextSource, RetrievedChunk

DEFAULT_MAX_CHUNKS = 10


def _source_of(chunk: Chunk) -> Source:
    return source_of_chunk(chunk.model_dump(mode="json"))


def _metadata_lines(chunk: Chunk, source: Source) -> list[str]:
    lines = [
        f"Document: {source.document_id} v{chunk.document.version} — {chunk.document.title}",
        f"Status: {chunk.document.lifecycle_status}",
        f"Effective from: {chunk.document.effective_from.isoformat()}",
        "Effective until: "
        + (
            chunk.document.effective_until.isoformat()
            if chunk.document.effective_until
            else "ongoing"
        ),
    ]
    if chunk.document.superseded_by:
        lines.append(f"Superseded by: {chunk.document.superseded_by}")
    lines.append(f"Section: {source.section}")
    if chunk.pages:
        label = "page" if len(chunk.pages) == 1 else "pages"
        lines.append(f"{label.title()}: {', '.join(str(page) for page in chunk.pages)}")
    return lines


def _format_block(source_id: str, chunk: Chunk, source: Source) -> str:
    return "\n".join(
        [
            f"--- SOURCE [{source_id}] ---",
            *_metadata_lines(chunk, source),
            "Content:",
            chunk.text,
            f"--- END SOURCE [{source_id}] ---",
        ]
    )


def build_context(
    retrieved: Sequence[RetrievedChunk],
    *,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
    include_archived: bool = False,
) -> BuiltContext:
    """Select ranked chunks and assign short, verifiable source labels.

    The input order is the retrieval order and is deliberately preserved. The
    builder does not rerank, merge or deduplicate evidence. ``max_chunks`` is a
    temporary context limit: chunk token counts use the embedding tokenizer,
    not the generation model's tokenizer.
    """

    if max_chunks < 1:
        raise ValueError("max_chunks must be at least 1.")

    selected: list[RetrievedChunk] = []
    for result in retrieved:
        if not include_archived and result.chunk.document.lifecycle_status != "active":
            continue
        selected.append(result)
        if len(selected) == max_chunks:
            break

    sources: list[ContextSource] = []
    blocks: list[str] = []
    for number, result in enumerate(selected, 1):
        source_id = f"S{number}"
        chunk = result.chunk
        source = _source_of(chunk)
        sources.append(
            ContextSource(
                source_id=source_id,
                rank=result.rank,
                score=result.score,
                chunk_id=chunk.chunk_id,
                source=source,
                title=chunk.document.title,
                version=chunk.document.version,
                lifecycle_status=chunk.document.lifecycle_status,
                effective_from=chunk.document.effective_from,
                effective_until=chunk.document.effective_until,
                superseded_by=chunk.document.superseded_by,
                pages=chunk.pages,
            )
        )
        blocks.append(_format_block(source_id, chunk, source))

    return BuiltContext(text="\n\n".join(blocks), sources=sources)
