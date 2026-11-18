"""API endpoints of the search test page, with an in-memory Qdrant and a fake embedder."""

import json
import math
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from peopleknowledge.api.main import app
from peopleknowledge.api.search import SearchService, get_search_service
from peopleknowledge.config import Settings, get_settings
from peopleknowledge.ingestion.chunking.models import Chunk
from peopleknowledge.retrieval.embeddings.base import Embedder
from peopleknowledge.retrieval.qdrant_index import ensure_collection, sync_chunks


class WordEmbedder(Embedder):
    """One dimension per known word: texts sharing words are close."""

    name = "fake-words"
    WORDS = ("essai", "congés", "télétravail", "lundi", "report")
    dimension = len(WORDS)

    def _vector(self, text: str) -> list[float]:
        vector = [float(word in text.lower()) for word in self.WORDS]
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector] if any(vector) else [1.0] + [0.0] * (self.dimension - 1)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def chunk(suffix: str, content: str, status: str = "active", doc: str = "HR-005") -> Chunk:
    version_id = f"{doc}-v1.0"
    prefix = f"{doc} v1.0 — Politique test > {suffix} Titre"
    return Chunk.model_validate({
        "chunk_id": f"{version_id}#{suffix}", "kind": "section", "prefix": prefix,
        "content": content, "text": f"{prefix}\n{content}", "token_count": 10,
        "section_number": suffix, "pages": [3], "element_ids": [f"{version_id}#0001"],
        "document": {
            "document_id": doc, "version_id": version_id, "family_id": "f", "title": "Politique test",
            "doc_type": "policy", "version": "1.0", "lifecycle_status": status,
            "effective_from": "2025-01-01", "language": "fr", "category": "leave",
            "department_scope": "all", "owner": "hr",
        },
        "source": {"relative_path": "a.pdf", "extension": ".pdf", "sha256": "e" * 64},
        "chunking": {"chunker_version": "1.0", "tokenizer": "fake", "max_tokens": 512},
    })


@pytest.fixture
def api() -> TestClient:
    qdrant = QdrantClient(":memory:")
    embedder = WordEmbedder()
    ensure_collection(qdrant, "test", embedder.dimension)
    sync_chunks(qdrant, "test", [
        chunk("4.2", "congés pendant la période d'essai : 2 jours"),
        chunk("6.1", "report des jours non pris"),
        chunk("3.2", "télétravail : le lundi est exclu", status="archived", doc="HR-029"),
    ], embedder)
    app.dependency_overrides[get_search_service] = lambda: SearchService(qdrant, embedder, "test")
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_page_and_health_are_served(api: TestClient) -> None:
    assert api.get("/health").json() == {"status": "ok"}
    page = api.get("/")
    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]
    assert "/api/answer" in page.text
    assert "PeopleKnowledge · Assistant RH" in page.text


def test_search_returns_ranked_results_with_their_source(api: TestClient) -> None:
    response = api.post("/api/search", json={"question": "congés pendant mon essai", "limit": 2})

    assert response.status_code == 200
    data = response.json()
    first = data["results"][0]
    assert first["rank"] == 1
    assert first["chunk_id"] == "HR-005-v1.0#4.2"
    assert first["source"] == {"document_id": "HR-005", "section": "4.2"}
    assert first["lifecycle_status"] == "active"
    assert first["pages"] == [3]
    assert len(data["results"]) == 2


def test_archived_documents_only_when_asked(api: TestClient) -> None:
    question = {"question": "télétravail le lundi", "limit": 5}

    default = api.post("/api/search", json=question).json()["results"]
    assert all(r["lifecycle_status"] == "active" for r in default)
    with_archived = api.post("/api/search", json={**question, "include_archived": True}).json()
    assert with_archived["results"][0]["chunk_id"] == "HR-029-v1.0#3.2"


@pytest.mark.parametrize("body", [{"question": ""}, {"question": "ok ok", "limit": 0}, {}])
def test_invalid_search_request_is_refused(api: TestClient, body: dict) -> None:
    assert api.post("/api/search", json=body).status_code == 422


def test_gold_questions_are_served(api: TestClient, tmp_path: Path) -> None:
    gold = tmp_path / "gold.jsonl"
    gold.write_text(json.dumps({
        "id": "Q001", "question": "Congés pendant l'essai ?",
        "expected_sources": [{"document_id": "HR-005", "section": "4.2"}],
        "expected_answer": "2 jours", "category": "direct", "difficulty": "easy",
    }) + "\n", encoding="utf-8")
    app.dependency_overrides[get_settings] = lambda: Settings(gold_path=gold)

    questions = api.get("/api/gold").json()

    assert [q["id"] for q in questions] == ["Q001"]
    assert questions[0]["expected_sources"] == [{"document_id": "HR-005", "section": "4.2"}]


def test_missing_index_gives_a_clear_error(tmp_path: Path) -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(qdrant_path=tmp_path / "absent")
    app.state.search_service = None
    try:
        response = TestClient(app).post("/api/search", json={"question": "congés"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert "index_chunks.py" in response.json()["detail"]
