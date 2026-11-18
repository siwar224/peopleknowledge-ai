"""Keyword search, rank fusion, profile rules and the retrieval pipeline (no model needed)."""

from peopleknowledge.retrieval.keyword_index import (
    KeywordIndex,
    reciprocal_rank_fusion,
    tokenize,
)
from peopleknowledge.retrieval.profile_filter import ProfileVocabulary, apply_profile
from peopleknowledge.retrieval.retriever import Hit

PROFILE = {
    "departments": [
        {"id": "eng", "label": "Engineering", "aliases": ["tech"]},
        {"id": "sales", "label": "Sales", "aliases": ["ventes"]},
    ],
    "job_families": [
        {"id": "sre", "label": "SRE", "aliases": []},
        {"id": "dev", "label": "Developer", "aliases": []},
    ],
    "employment_types": [],
}


def payload(chunk_id: str, text: str, kind: str = "section", status: str = "active") -> dict:
    return {
        "chunk_id": chunk_id,
        "kind": kind,
        "text": text,
        "content": text,
        "document": {"lifecycle_status": status},
    }


def test_tokenize_drops_accents_case_and_stopwords() -> None:
    assert tokenize("Les Congés de l'astreinte J+3") == ["conges", "astreinte", "j+3"]


def test_bm25_ranks_rare_words_first_and_skips_archived() -> None:
    index = KeywordIndex(
        [
            payload("a", "congés annuels et report"),
            payload("b", "astreinte SRE trois jours"),
            payload("c", "astreinte ancienne règle", status="archived"),
            payload("d", "congés maladie"),
        ]
    )

    found = [p["chunk_id"] for _, p in index.search("jours d'astreinte", 5)]
    with_archive = [p["chunk_id"] for _, p in index.search("astreinte", 5, include_archived=True)]

    assert found == ["b"]
    assert set(with_archive) == {"b", "c"}


def test_rrf_rewards_agreement_between_rankings() -> None:
    fused = reciprocal_rank_fusion([["x", "y", "z"], ["y", "x"]])

    assert fused["x"] == fused["y"] > fused["z"]


def rows() -> list[Hit]:
    return [
        Hit(0.9, payload("row-sales", "department: Sales | job_family: Developer", "record")),
        Hit(
            0.8, payload("row-eng-dev", "department: Engineering | job_family: Developer", "record")
        ),
        Hit(0.7, payload("policy", "règle générale du télétravail")),
        Hit(0.6, payload("row-eng-sre", "department: Engineering | job_family: SRE", "record")),
    ]


def ids(hits) -> list[str]:
    return [hit.payload["chunk_id"] for hit in hits]


def test_rows_about_other_people_move_to_the_end() -> None:
    vocabulary = ProfileVocabulary.from_profile(PROFILE)

    ordered = apply_profile(rows(), {"department": "eng", "job_family": "sre"}, vocabulary, None)

    assert ids(ordered) == ["policy", "row-eng-sre", "row-sales", "row-eng-dev"]


def test_without_profile_only_the_row_cap_applies() -> None:
    ordered = apply_profile(rows(), None, None, max_records=1)

    assert ids(ordered) == ["row-sales", "policy", "row-eng-dev", "row-eng-sre"]


def test_pipeline_combines_dense_keyword_rerank_and_profile() -> None:
    from qdrant_client import QdrantClient

    from peopleknowledge.retrieval.qdrant_index import ensure_collection, sync_chunks
    from peopleknowledge.retrieval.reranker import Reranker
    from peopleknowledge.retrieval.retriever import RetrievalOptions, Retriever
    from tests.api.test_answer_api import WordEmbedder, chunk

    class LengthReranker(Reranker):
        name = "fake-rerank"

        def score(self, question: str, texts: list[str]) -> list[float]:
            return [-len(text) for text in texts]  # shortest text first

    client, embedder = QdrantClient(":memory:"), WordEmbedder()
    ensure_collection(client, "t", embedder.dimension)
    sync_chunks(
        client,
        "t",
        [
            chunk("1", "congés pendant l'essai : deux jours, avec accord du manager"),
            chunk("2", "report des congés"),
            chunk("3", "télétravail lundi", status="archived", doc="HR-029"),
        ],
        embedder,
    )
    plain = Retriever(client, "t", embedder)
    full = Retriever(
        client,
        "t",
        embedder,
        options=RetrievalOptions(hybrid=True, profile=True, max_records=2),
        reranker=LengthReranker(),
    )

    assert plain.describe() == "dense"
    assert full.describe() == "dense+bm25-rrf+profile(max_records=2)+rerank(fake-rerank)"
    assert [h.payload["chunk_id"] for h in plain.retrieve("congés essai", limit=1)] == [
        "HR-005-v1.0#1"
    ]
    reranked = [h.payload["chunk_id"] for h in full.retrieve("congés", limit=3)]
    assert reranked[0] == "HR-005-v1.0#2"  # the shortest text wins the fake reranker
    assert all("HR-029" not in chunk_id for chunk_id in reranked)  # archived stays out
