"""Tests for the complete retrieval-to-generation answer endpoint."""

import math
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from peopleknowledge.api.answer import GenerationService, get_generation_service
from peopleknowledge.api.main import app
from peopleknowledge.api.search import SearchService, get_search_service
from peopleknowledge.config import Settings, get_settings
from peopleknowledge.generation import (
    BuiltPrompt,
    GeneratedAnswer,
    GenerationError,
    GenerationResult,
    Generator,
)
from peopleknowledge.ingestion.chunking.models import Chunk
from peopleknowledge.retrieval.embeddings.base import Embedder
from peopleknowledge.retrieval.qdrant_index import ensure_collection, sync_chunks


class WordEmbedder(Embedder):
    name = "fake-words"
    WORDS = ("essai", "congés", "télétravail", "lundi", "report")
    dimension = len(WORDS)

    def _vector(self, text: str) -> list[float]:
        vector = [float(word in text.lower()) for word in self.WORDS]
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return (
            [value / norm for value in vector]
            if any(vector)
            else [1.0] + [0.0] * (self.dimension - 1)
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def chunk(suffix: str, content: str, *, status: str = "active", doc: str = "HR-005") -> Chunk:
    version_id = f"{doc}-v1.0"
    prefix = f"{doc} v1.0 — Test policy > {suffix} Test section"
    return Chunk.model_validate(
        {
            "chunk_id": f"{version_id}#{suffix}",
            "kind": "section",
            "prefix": prefix,
            "content": content,
            "text": f"{prefix}\n{content}",
            "token_count": 20,
            "section_number": suffix,
            "pages": [3],
            "element_ids": [f"{version_id}#0001"],
            "document": {
                "document_id": doc,
                "version_id": version_id,
                "family_id": "test_policy",
                "title": "Test policy",
                "doc_type": "policy",
                "version": "1.0",
                "lifecycle_status": status,
                "effective_from": "2025-01-01",
                "effective_until": "2026-08-31" if status == "archived" else None,
                "superseded_by": "HR-011" if status == "archived" else None,
                "language": "fr",
                "category": "leave",
                "department_scope": "all",
                "owner": "hr",
            },
            "source": {
                "relative_path": "test.pdf",
                "extension": ".pdf",
                "sha256": "e" * 64,
            },
            "chunking": {
                "chunker_version": "1.0",
                "tokenizer": "fake",
                "max_tokens": 512,
            },
        }
    )


class FakeGenerator(Generator):
    model = "fake-qwen"

    def __init__(self) -> None:
        self.prompts: list[BuiltPrompt] = []

    def generate(self, prompt: BuiltPrompt) -> GenerationResult:
        self.prompts.append(prompt)
        return GenerationResult(
            answer=GeneratedAnswer(
                status="complete",
                answer="Vous pouvez poser deux jours.",
                citations=["S1"],
            ),
            model=self.model,
            elapsed_ms=15,
        )


class FailingGenerator(Generator):
    model = "failing"

    def generate(self, prompt: BuiltPrompt) -> GenerationResult:
        raise GenerationError("local model is unavailable")


@pytest.fixture
def answer_api() -> tuple[TestClient, FakeGenerator]:
    qdrant = QdrantClient(":memory:")
    embedder = WordEmbedder()
    ensure_collection(qdrant, "test", embedder.dimension)
    sync_chunks(
        qdrant,
        "test",
        [
            chunk("4.2", "congés pendant la période d'essai : 2 jours"),
            chunk("6.1", "report des jours non pris"),
            chunk(
                "3.2",
                "télétravail : le lundi est exclu",
                status="archived",
                doc="HR-029",
            ),
        ],
        embedder,
    )
    generator = FakeGenerator()
    app.dependency_overrides[get_search_service] = lambda: SearchService(qdrant, embedder, "test")
    app.dependency_overrides[get_generation_service] = lambda: GenerationService(generator)
    yield TestClient(app), generator
    app.dependency_overrides.clear()


def test_answer_runs_the_complete_pipeline_and_resolves_citations(
    answer_api: tuple[TestClient, FakeGenerator],
) -> None:
    api, generator = answer_api

    response = api.post(
        "/api/answer",
        json={
            "question": "congés pendant mon essai",
            "limit": 2,
            "reference_date": "2026-11-04",
            "user_context": {"employment_type": "cdi", "in_probation": True},
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "complete"
    assert data["answer"] == "Vous pouvez poser deux jours."
    assert data["model"] == "fake-qwen"
    assert data["citations"][0]["source"] == {"document_id": "HR-005", "section": "4.2"}
    assert data["citations"][0]["source_id"] == "S1"
    assert data["retrieval_results"][0]["chunk_id"] == "HR-005-v1.0#4.2"
    assert data["timing"]["generation_ms"] == 15
    assert data["timing"]["total_ms"] >= 0
    user_message = generator.prompts[0].messages[1].content
    assert "employment_type: cdi" in user_message
    assert "in_probation: true" in user_message
    assert user_message.endswith("congés pendant mon essai")


def test_answer_only_includes_archived_evidence_when_requested(
    answer_api: tuple[TestClient, FakeGenerator],
) -> None:
    api, generator = answer_api

    current = api.post(
        "/api/answer",
        json={"question": "télétravail le lundi", "limit": 3},
    ).json()
    history = api.post(
        "/api/answer",
        json={"question": "télétravail le lundi", "limit": 3, "include_archived": True},
    ).json()

    assert all(item["lifecycle_status"] == "active" for item in current["retrieval_results"])
    assert history["retrieval_results"][0]["lifecycle_status"] == "archived"
    assert "Status: archived" in generator.prompts[-1].messages[1].content


def test_generation_failure_is_returned_as_bad_gateway(answer_api) -> None:
    api, _ = answer_api
    app.dependency_overrides[get_generation_service] = lambda: GenerationService(
        FailingGenerator()
    )

    response = api.post("/api/answer", json={"question": "congés pendant mon essai"})

    assert response.status_code == 502
    assert response.json()["detail"] == "local model is unavailable"


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"question": ""},
        {"question": "question", "limit": 0},
        {"question": "question", "reference_date": "not-a-date"},
    ],
)
def test_invalid_answer_request_is_refused(answer_api, body: dict) -> None:
    api, _ = answer_api
    assert api.post("/api/answer", json=body).status_code == 422


def test_profile_options_come_from_the_company_profile(tmp_path: Path) -> None:
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        """departments:
  - {id: engineering, label: Engineering}
job_families:
  - {id: software_engineer, label: Software Engineer, department: engineering}
employment_types:
  - {id: cdi, label: CDI}
""",
        encoding="utf-8",
    )
    app.dependency_overrides[get_settings] = lambda: Settings(company_profile_path=profile)
    try:
        response = TestClient(app).get("/api/profile-options")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["job_families"] == [
        {"id": "software_engineer", "label": "Software Engineer", "department": "engineering"}
    ]
