"""Qdrant indexing and search, with an in-memory Qdrant and a fake embedder.

The fake embedder turns each word into a fixed direction (hashing), so that
texts sharing words are close: enough to test the logic in milliseconds,
without the 2 GB model.
"""

import hashlib
import json
import math
import uuid
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from peopleknowledge.ingestion.chunking.models import Chunk
from peopleknowledge.retrieval.embeddings.base import Embedder
from peopleknowledge.retrieval.qdrant_index import (
    IndexingError,
    ensure_collection,
    point_id,
    search,
    sync_chunks,
)
from scripts.index_chunks import main as index_main

COLLECTION = "test"


class FakeEmbedder(Embedder):
    def __init__(self, name: str = "fake-model", dimension: int = 64) -> None:
        self.name = name
        self.dimension = dimension
        self.encoded: list[str] = []  # every text encoded, to check what was (not) re-encoded

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for word in text.lower().split():
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dimension] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.encoded.extend(texts)
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def make_chunk(suffix: str, content: str, status: str = "active", doc: str = "HR-005") -> Chunk:
    version_id = f"{doc}-v1.0"
    prefix = f"{doc} v1.0 — Politique > {suffix}"
    return Chunk.model_validate(
        {
            "chunk_id": f"{version_id}#{suffix}",
            "kind": "section",
            "prefix": prefix,
            "content": content,
            "text": f"{prefix}\n{content}",
            "token_count": 10,
            "section_number": suffix,
            "element_ids": [f"{version_id}#0001"],
            "document": {
                "document_id": doc, "version_id": version_id, "family_id": "f", "title": "Politique",
                "doc_type": "policy", "version": "1.0", "lifecycle_status": status,
                "effective_from": "2025-01-01", "language": "fr", "category": "leave",
                "department_scope": "all", "owner": "hr",
            },
            "source": {"relative_path": "a.pdf", "extension": ".pdf", "sha256": "d" * 64},
            "chunking": {"chunker_version": "1.0", "tokenizer": "fake", "max_tokens": 512},
        }
    )


CHUNKS = [
    make_chunk("4.2", "période essai congés deux jours accord manager"),
    make_chunk("6.1", "report des jours non pris limite cinq jours"),
    make_chunk("3.2", "télétravail lundi exclu jour fixe", status="archived", doc="HR-029"),
]


@pytest.fixture
def client() -> QdrantClient:
    client = QdrantClient(":memory:")
    ensure_collection(client, COLLECTION, 64)
    return client


def test_point_id_is_a_stable_uuid() -> None:
    assert point_id("HR-005-v1.2#4.2") == point_id("HR-005-v1.2#4.2")
    assert point_id("HR-005-v1.2#4.2") != point_id("HR-005-v1.2#4.3")
    uuid.UUID(point_id("HR-005-v1.2#4.2"))  # valid UUID, as Qdrant requires


def test_collection_with_another_vector_size_is_refused_unless_recreated(client: QdrantClient) -> None:
    with pytest.raises(IndexingError, match="size 64"):
        ensure_collection(client, COLLECTION, 1024)

    ensure_collection(client, COLLECTION, 1024, recreate=True)
    assert client.get_collection(COLLECTION).config.params.vectors.size == 1024


def test_first_sync_indexes_every_chunk_with_its_metadata(client: QdrantClient) -> None:
    report = sync_chunks(client, COLLECTION, CHUNKS, FakeEmbedder())

    assert sorted(report.new) == sorted(c.chunk_id for c in CHUNKS)
    assert client.count(COLLECTION).count == 3
    point = client.retrieve(COLLECTION, [point_id("HR-005-v1.0#4.2")], with_payload=True)[0]
    assert point.payload["chunk_id"] == "HR-005-v1.0#4.2"
    assert point.payload["document"]["lifecycle_status"] == "active"
    assert point.payload["embedding_model"] == "fake-model"
    assert point.payload["section_number"] == "4.2"


def test_second_sync_encodes_nothing(client: QdrantClient) -> None:
    sync_chunks(client, COLLECTION, CHUNKS, FakeEmbedder())
    embedder = FakeEmbedder()

    report = sync_chunks(client, COLLECTION, CHUNKS, embedder)

    assert len(report.unchanged) == 3 and report.encoded == 0
    assert embedder.encoded == []


def test_only_changed_chunks_are_encoded_and_removed_ones_deleted(client: QdrantClient) -> None:
    sync_chunks(client, COLLECTION, CHUNKS, FakeEmbedder())
    changed = make_chunk("6.1", "report des jours non pris limite dix jours")
    embedder = FakeEmbedder()

    report = sync_chunks(client, COLLECTION, [CHUNKS[0], changed], embedder)

    assert report.updated == ["HR-005-v1.0#6.1"]
    assert report.unchanged == ["HR-005-v1.0#4.2"]
    assert report.deleted == ["HR-029-v1.0#3.2"]
    assert embedder.encoded == [changed.text]
    assert client.count(COLLECTION).count == 2


def test_a_new_model_re_encodes_everything(client: QdrantClient) -> None:
    sync_chunks(client, COLLECTION, CHUNKS, FakeEmbedder())

    report = sync_chunks(client, COLLECTION, CHUNKS, FakeEmbedder(name="other-model"))

    assert len(report.updated) == 3


def test_duplicate_chunk_ids_are_refused(client: QdrantClient) -> None:
    with pytest.raises(IndexingError, match="duplicate"):
        sync_chunks(client, COLLECTION, [CHUNKS[0], CHUNKS[0]], FakeEmbedder())


def test_search_finds_the_closest_chunk_and_leaves_archived_out(client: QdrantClient) -> None:
    embedder = FakeEmbedder()
    sync_chunks(client, COLLECTION, CHUNKS, embedder)

    best = search(client, COLLECTION, embedder, "congés pendant la période essai", limit=1)[0]
    assert best.payload["chunk_id"] == "HR-005-v1.0#4.2"

    active_only = search(client, COLLECTION, embedder, "télétravail lundi exclu", limit=5)
    assert all(p.payload["document"]["lifecycle_status"] == "active" for p in active_only)
    with_archived = search(
        client, COLLECTION, embedder, "télétravail lundi exclu", limit=1, include_archived=True
    )
    assert with_archived[0].payload["chunk_id"] == "HR-029-v1.0#3.2"


def test_runner_indexes_chunk_files_into_a_local_folder(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    (chunks_dir / "HR-005-v1.0.jsonl").write_text(
        "".join(c.to_json_line() for c in CHUNKS[:2]), encoding="utf-8"
    )
    arguments = ["--chunks-dir", str(chunks_dir), "--qdrant-path", str(tmp_path / "qdrant")]

    assert index_main(arguments, embedder=FakeEmbedder()) == 0
    assert "new=2" in capsys.readouterr().out
    assert index_main(arguments, embedder=FakeEmbedder()) == 0
    assert "unchanged=2" in capsys.readouterr().out
    assert (tmp_path / "qdrant").is_dir()  # local embedded mode: just a folder


def test_runner_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    qdrant = ["--qdrant-path", str(tmp_path / "q")]

    assert index_main(["--chunks-dir", str(empty), *qdrant], embedder=FakeEmbedder()) == 1
    assert "no chunk file" in capsys.readouterr().out
    (empty / "bad.jsonl").write_text(json.dumps({"chunk_id": "x"}) + "\n", encoding="utf-8")
    assert index_main(["--chunks-dir", str(empty), *qdrant], embedder=FakeEmbedder()) == 1
    assert "bad.jsonl line 1" in capsys.readouterr().out
