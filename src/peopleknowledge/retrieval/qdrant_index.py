"""Keep a Qdrant collection in sync with the chunk files.

One Qdrant POINT per chunk:
- ``id``: a UUID derived from the chunk id (Qdrant ids must be integers or
  UUIDs); the same chunk always gets the same id, so re-indexing replaces it;
- ``vector``: the embedding of ``chunk.text`` (prefix + content);
- ``payload``: the WHOLE chunk (text, section, pages, all document metadata...)
  plus ``embedding_model`` and ``text_hash``. Search results therefore carry
  everything needed to filter (``document.lifecycle_status``...) and to cite.

``sync_chunks`` makes the collection mirror the chunks, like the runners of the
previous steps: it encodes ONLY chunks that are new or whose text or model
changed (encoding is the slow part, several minutes on a laptop CPU), and
deletes points of chunks that no longer exist.

The client is created by the caller: the same code works with the local
embedded mode (``QdrantClient(path=...)``), a local server or Qdrant Cloud.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from qdrant_client import QdrantClient, models

from ..ingestion.chunking.models import Chunk
from .embeddings.base import Embedder

# Fixed namespace: uuid5(namespace, chunk_id) is stable across runs and machines.
POINT_NAMESPACE = uuid.UUID("6f1c2d3e-4b5a-4c6d-8e7f-9a0b1c2d3e4f")


class IndexingError(Exception):
    """The collection cannot be used as is (e.g. built with another vector size)."""


@dataclass
class SyncReport:
    new: list[str] = field(default_factory=list)  # chunk ids
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)

    @property
    def encoded(self) -> int:
        return len(self.new) + len(self.updated)


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(POINT_NAMESPACE, chunk_id))


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ensure_collection(
    client: QdrantClient, name: str, dimension: int, recreate: bool = False
) -> None:
    """Create the collection (cosine distance) if needed; refuse a wrong vector size."""

    if client.collection_exists(name):
        if recreate:
            client.delete_collection(name)
        else:
            size = client.get_collection(name).config.params.vectors.size
            if size != dimension:
                raise IndexingError(
                    f"collection {name!r} holds vectors of size {size}, "
                    f"the model gives {dimension}: use --recreate to rebuild it."
                )
            return
    client.create_collection(
        name,
        vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
    )


def payload_of(chunk: Chunk, embedding_model: str) -> dict[str, Any]:
    payload = chunk.model_dump(mode="json")
    payload["embedding_model"] = embedding_model
    payload["text_hash"] = text_hash(chunk.text)
    return payload


def chunk_from_payload(payload: dict[str, Any]) -> Chunk:
    """The validated chunk stored in a point payload (index-only fields removed)."""

    chunk_payload = dict(payload)
    chunk_payload.pop("embedding_model", None)
    chunk_payload.pop("text_hash", None)
    return Chunk.model_validate(chunk_payload)


def indexed_state(client: QdrantClient, collection: str) -> dict[str, tuple[str, str]]:
    """chunk_id -> (text_hash, embedding_model) of every point already in the collection."""

    state: dict[str, tuple[str, str]] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection,
            limit=256,
            offset=offset,
            with_payload=["chunk_id", "text_hash", "embedding_model"],
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            state[payload["chunk_id"]] = (
                payload.get("text_hash", ""),
                payload.get("embedding_model", ""),
            )
        if offset is None:
            return state


def sync_chunks(
    client: QdrantClient,
    collection: str,
    chunks: Iterable[Chunk],
    embedder: Embedder,
    batch_size: int = 64,
) -> SyncReport:
    """Make the collection contain exactly these chunks, encoding only what changed."""

    chunks = list(chunks)
    ids = [chunk.chunk_id for chunk in chunks]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise IndexingError(f"duplicate chunk ids: {duplicates[:5]}")

    state = indexed_state(client, collection)
    report = SyncReport()
    to_encode: list[Chunk] = []
    for chunk in chunks:
        known = state.get(chunk.chunk_id)
        if known == (text_hash(chunk.text), embedder.name):
            report.unchanged.append(chunk.chunk_id)
        else:
            (report.updated if known else report.new).append(chunk.chunk_id)
            to_encode.append(chunk)

    for start in range(0, len(to_encode), batch_size):
        batch = to_encode[start : start + batch_size]
        vectors = embedder.embed_documents([chunk.text for chunk in batch])
        client.upsert(
            collection,
            points=[
                models.PointStruct(
                    id=point_id(chunk.chunk_id),
                    vector=vector,
                    payload=payload_of(chunk, embedder.name),
                )
                for chunk, vector in zip(batch, vectors, strict=True)
            ],
        )

    gone = sorted(set(state) - set(ids))
    if gone:
        client.delete(
            collection, points_selector=models.PointIdsList(points=[point_id(i) for i in gone])
        )
    report.deleted = gone
    return report


def search(
    client: QdrantClient,
    collection: str,
    embedder: Embedder,
    question: str,
    limit: int = 5,
    include_archived: bool = False,
) -> list[models.ScoredPoint]:
    """The chunks closest to the question. Archived documents are left out by default."""

    query_filter = None
    if not include_archived:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="document.lifecycle_status", match=models.MatchValue(value="active")
                )
            ]
        )
    return client.query_points(
        collection,
        query=embedder.embed_query(question),
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
    ).points
