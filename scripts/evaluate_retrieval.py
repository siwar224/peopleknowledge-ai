"""Measure the retrieval on all the gold questions: hit@k, MRR, forbidden sources.

Runs every question of datasets/evaluation/rag_questions_gold.jsonl against
the Qdrant index, prints the indicators (overall and per trap category) and the
questions whose expected source is not first, and saves the full detail to
datasets/evaluation/results/retrieval_<label>.json so that two runs can be
compared (another model, another chunk size...).

By default only "history" questions search the archived documents, as the
assistant will do. ``--archived always`` is a stress test: with the old
version searchable, how often does it come back for a current question?

The local Qdrant folder can be opened by ONE program at a time: stop the API
(uvicorn) before running this script. Settings (HF_HOME...) come from .env.

Usage:
    python scripts/evaluate_retrieval.py
    python scripts/evaluate_retrieval.py --label baseline
    python scripts/evaluate_retrieval.py --archived always --label stress
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from statistics import median

from qdrant_client import QdrantClient

from peopleknowledge.config import get_settings
from peopleknowledge.evaluation.gold import Source, load_gold, source_of_chunk
from peopleknowledge.evaluation.retrieval_eval import (
    HIT_LEVELS,
    QuestionResult,
    Retrieved,
    Retriever,
    Summary,
    evaluate,
    summaries_by_category,
    summarize,
)
from peopleknowledge.retrieval.embeddings import Embedder
from peopleknowledge.retrieval.qdrant_index import search

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "datasets" / "evaluation" / "results"


def qdrant_retriever(
    client: QdrantClient, collection: str, embedder: Embedder, limit: int
) -> Retriever:
    def retrieve(question: str, include_archived: bool) -> list[Retrieved]:
        points = search(
            client, collection, embedder, question, limit=limit, include_archived=include_archived
        )
        return [
            Retrieved(
                source=source_of_chunk(point.payload),
                chunk_id=point.payload["chunk_id"],
                score=point.score,
            )
            for point in points
        ]

    return retrieve


PostJson = Callable[[str, dict], dict]


def http_post_json(url: str, body: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read().decode("utf-8"))


def api_retriever(api_url: str, limit: int, post_json: PostJson = http_post_json) -> Retriever:
    """Search through a running API (POST /api/search) instead of opening the index.

    Useful with the local embedded Qdrant, which only one program can open:
    the API keeps the index, and this script just asks it, like the test page.
    """

    endpoint = api_url.rstrip("/") + "/api/search"

    def retrieve(question: str, include_archived: bool) -> list[Retrieved]:
        body = {"question": question, "limit": limit, "include_archived": include_archived}
        return [
            Retrieved(
                source=Source(**result["source"]),
                chunk_id=result["chunk_id"],
                score=result["score"],
            )
            for result in post_json(endpoint, body)["results"]
        ]

    return retrieve


def name(source) -> str:
    return f"{source.document_id} §{source.section}"


def summary_line(summary: Summary) -> str:
    hits = "".join(f"{summary.hits[k]:>8.0%}" for k in HIT_LEVELS)
    useful = "".join(f"{summary.useful[k]:>10.0%}" for k in (1, 3))
    return (
        f"{summary.label:<22}{summary.questions:>4}{hits}{summary.mrr:>8.3f}"
        f"{useful}{summary.forbidden:>11}"
    )


def print_report(results: list[QuestionResult], limit: int) -> None:
    header = "".join(f"{'hit@' + str(k):>8}" for k in HIT_LEVELS)
    useful = "".join(f"{'useful@' + str(k):>10}" for k in (1, 3))
    print(f"{'category':<22}{'n':>4}{header}{'MRR':>8}{useful}{'forbidden':>11}")
    for summary in summaries_by_category(results):
        print(summary_line(summary))
    print("-" * 89)
    print(summary_line(summarize("ALL (answerable)", results)))
    print("hit@k: ALL expected sources in the top k.  useful@k: an expected OR acceptable source.")

    not_first = [r for r in results if r.answerable and r.complete_rank != 1]
    print(f"\nEXPECTED SOURCE NOT FIRST: {len(not_first)} question(s)")
    for result in not_first:
        ranks = ", ".join(
            f"{name(source)} = {'rank ' + str(rank) if rank else 'MISSING (top ' + str(limit) + ')'}"
            for source, rank in result.expected_ranks.items()
        )
        first = result.retrieved[0]
        if first.source in result.acceptable_ranks:
            role = "acceptable"
        elif first.source in result.expected_ranks:
            role = "expected"
        else:
            role = "other"
        print(f"  {result.question.id} [{result.question.category}] {ranks}")
        print(f"       first result: {name(first.source)} ({first.score:.3f}, {role})")
        print(f"       {result.question.question}")

    leaked = [r for r in results if r.forbidden_ranks]
    print(f"\nFORBIDDEN SOURCE RETRIEVED: {len(leaked)} question(s)")
    for result in leaked:
        ranks = ", ".join(f"{name(s)} at rank {rank}" for s, rank in result.forbidden_ranks.items())
        print(f"  {result.question.id} [{result.question.category}] {ranks}")

    answerable_scores = [r.top_score for r in results if r.answerable and r.top_score is not None]
    print("\nUNANSWERABLE QUESTIONS (best score; the corpus does not contain the answer)")
    for result in results:
        if not result.answerable and result.top_score is not None:
            first = result.retrieved[0]
            print(f"  {result.question.id} {result.top_score:.3f} ({name(first.source)})  {result.question.question}")
    if answerable_scores:
        print(
            f"  for comparison, answerable questions: best score median "
            f"{median(answerable_scores):.3f}, min {min(answerable_scores):.3f}"
        )


def to_json(results: list[QuestionResult], settings: dict) -> dict:
    overall = summarize("all", results)
    return {
        "settings": settings,
        "overall": {
            "questions": overall.questions,
            "hits": {f"hit@{k}": round(v, 4) for k, v in overall.hits.items()},
            "useful": {f"useful@{k}": round(v, 4) for k, v in overall.useful.items()},
            "mrr": round(overall.mrr, 4),
            "forbidden": overall.forbidden,
        },
        "by_category": {
            s.label: {
                "questions": s.questions,
                "hits": {f"hit@{k}": round(v, 4) for k, v in s.hits.items()},
                "useful": {f"useful@{k}": round(v, 4) for k, v in s.useful.items()},
                "mrr": round(s.mrr, 4),
                "forbidden": s.forbidden,
            }
            for s in summaries_by_category(results)
        },
        "questions": [
            {
                "id": r.question.id,
                "category": r.question.category,
                "include_archived": r.include_archived,
                "first_rank": r.first_rank,
                "complete_rank": r.complete_rank,
                "expected_ranks": {name(s): rank for s, rank in r.expected_ranks.items()},
                "forbidden_ranks": {name(s): rank for s, rank in r.forbidden_ranks.items()},
                "retrieved": [
                    {"rank": rank, "chunk_id": item.chunk_id, "score": round(item.score, 4)}
                    for rank, item in enumerate(r.retrieved, 1)
                ],
            }
            for r in results
        ],
    }


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gold", type=Path, default=settings.gold_path)
    parser.add_argument("--limit", type=int, default=10, help="results retrieved per question")
    parser.add_argument("--archived", choices=["history", "never", "always"], default="history")
    parser.add_argument("--label", default="latest", help="name of the saved result file")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--api-url",
        help="search through a running API (e.g. http://localhost:8000) instead of opening "
        "the index; needed while the API holds the local Qdrant folder. Top 20 at most.",
    )
    arguments = parser.parse_args(argv)

    try:
        questions = load_gold(arguments.gold)
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}")
        return 1
    if arguments.api_url:
        try:
            retriever = api_retriever(arguments.api_url, arguments.limit)
            results = evaluate(questions, retriever, archived=arguments.archived)
        except (urllib.error.URLError, OSError, KeyError, ValueError) as error:
            print(f"ERROR: cannot search through {arguments.api_url}: {error}")
            return 1
    else:
        if not settings.qdrant_url and not settings.qdrant_path.exists():
            print("ERROR: no Qdrant index yet (run scripts/index_chunks.py first).")
            return 1
        settings.apply_hugging_face_environment()  # before the model library is imported
        from peopleknowledge.retrieval.embeddings import SentenceTransformerEmbedder

        try:
            if settings.qdrant_url:
                client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
            else:
                client = QdrantClient(path=str(settings.qdrant_path))
        except RuntimeError as error:  # folder already opened by another program
            print(
                f"ERROR: {error}\nEither stop the API (uvicorn), or keep it running and add "
                "--api-url http://localhost:<its port> to search through it."
            )
            return 1
        try:
            embedder = SentenceTransformerEmbedder(settings.embedding_model)
            retriever = qdrant_retriever(
                client, settings.qdrant_collection, embedder, arguments.limit
            )
            results = evaluate(questions, retriever, archived=arguments.archived)
        finally:
            client.close()

    print(
        f"Retrieval evaluation: {len(questions)} gold questions, model {settings.embedding_model}, "
        f"top {arguments.limit}, archived documents: {arguments.archived}\n"
    )
    print_report(results, arguments.limit)

    run_settings = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "embedding_model": settings.embedding_model,
        "collection": settings.qdrant_collection,
        "limit": arguments.limit,
        "archived": arguments.archived,
        "gold_questions": len(questions),
    }
    arguments.results_dir.mkdir(parents=True, exist_ok=True)
    path = arguments.results_dir / f"retrieval_{arguments.label}.json"
    path.write_text(
        json.dumps(to_json(results, run_settings), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
