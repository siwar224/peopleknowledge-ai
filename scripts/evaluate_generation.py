"""Measure the generation on all the gold questions (automatic checks, no LLM judge).

Two modes, to tell retrieval errors from generation errors:

- ``e2e`` (default): the whole system. Each question goes through a RUNNING API
  (POST /api/answer with its gold_id), exactly like the test page; the API
  retrieves, generates and evaluates. Start uvicorn and llama-server first.
- ``oracle``: the generation ALONE. The model receives the gold's expected (and
  acceptable) sources, read from the chunk files, instead of Qdrant results.
  Only llama-server is needed. If oracle is bad, the prompt or the model is the
  problem; if oracle is good but e2e is bad, the retrieval is.

Indicators (see evaluation/generation_eval.py): pass rate, status, key facts
coverage, source recall / precision, forbidden sources, numbers absent from the
sources, prompt leaks, latency. The detail of every answer is saved to
datasets/evaluation/results/generation_<label>.json.

Usage:
    python scripts/evaluate_generation.py --label baseline
    python scripts/evaluate_generation.py --mode oracle --label baseline_oracle
    python scripts/evaluate_generation.py --only Q001,Q016
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

from peopleknowledge.config import get_settings
from peopleknowledge.evaluation.generation_eval import (
    GenerationEvaluation,
    GenerationSummary,
    evaluate_answer,
    summarize_generation,
)
from peopleknowledge.evaluation.gold import GoldQuestion, Source, load_gold, source_of_chunk
from peopleknowledge.generation import (
    GenerationError,
    Generator,
    OpenAICompatibleGenerator,
    RetrievedChunk,
    build_context,
    build_prompt,
)
from peopleknowledge.ingestion.chunking.models import Chunk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS_DIR = PROJECT_ROOT / "datasets" / "processed" / "chunks"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "datasets" / "evaluation" / "results"


def reference_date(rules: dict[str, Any]) -> date:
    value = (rules.get("meta") or {}).get("reference_date_for_questions")
    return value if isinstance(value, date) else date.fromisoformat(str(value))


def load_chunks(chunks_dir: Path) -> dict[Source, list[Chunk]]:
    """Every chunk of the corpus, grouped by the source a gold question cites."""

    by_source: dict[Source, list[Chunk]] = {}
    for path in sorted(chunks_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                data = json.loads(line)
                by_source.setdefault(source_of_chunk(data), []).append(Chunk.model_validate(data))
    return by_source


def oracle_chunks(
    question: GoldQuestion, chunks: dict[Source, list[Chunk]]
) -> list[RetrievedChunk]:
    """Expected sources first, then acceptable ones: the 'perfect retrieval'."""

    selected: list[Chunk] = []
    for source in [*question.expected_sources, *question.acceptable_sources]:
        selected.extend(chunks.get(source, []))
    return [
        RetrievedChunk(rank=rank, score=1.0, chunk=chunk) for rank, chunk in enumerate(selected, 1)
    ]


# Each runner answers ONE question and returns the stored record.
Runner = Callable[[GoldQuestion], dict[str, Any]]


def oracle_runner(
    generator: Generator, chunks: dict[Source, list[Chunk]], rules: dict[str, Any], day: date
) -> Runner:
    def run(question: GoldQuestion) -> dict[str, Any]:
        context = build_context(
            oracle_chunks(question, chunks), max_chunks=50, include_archived=True
        )
        prompt = build_prompt(
            question.question, context, reference_date=day, user_context=question.user_context
        )
        result = generator.generate(prompt)
        labels = {source.source_id: source.source for source in context.sources}
        cited = [labels[label] for label in result.answer.citations]
        evaluation = evaluate_answer(
            question_text=question.question,
            status=result.answer.status,
            answer=result.answer.answer,
            cited=cited,
            prompt_user_text=prompt.messages[1].content,
            system_prompt=prompt.messages[0].content,
            gold=question,
            rules=rules,
        )
        return {
            "status": result.answer.status,
            "answer": result.answer.answer,
            "citations": [source.model_dump() for source in cited],
            "context_sources": len(context.sources),
            "elapsed_ms": result.elapsed_ms,
            "evaluation": evaluation.model_dump(mode="json"),
        }

    return run


def http_post_json(url: str, body: dict) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:  # the API explains why in "detail"
        detail = error.read().decode("utf-8", errors="replace")
        raise GenerationError(f"API {error.code}: {detail}") from error


def api_runner(api_url: str, limit: int, day: date, post_json=http_post_json) -> Runner:
    endpoint = api_url.rstrip("/") + "/api/answer"

    def run(question: GoldQuestion) -> dict[str, Any]:
        data = post_json(
            endpoint,
            {
                "question": question.question,
                "user_context": question.user_context.model_dump(exclude_none=True),
                "reference_date": day.isoformat(),
                "limit": limit,
                "include_archived": question.category == "history",
                "gold_id": question.id,
            },
        )
        return {
            "status": data["status"],
            "answer": data["answer"],
            "citations": [citation["source"] for citation in data["citations"]],
            "context_sources": data["retrieved_count"],
            "elapsed_ms": data["timing"]["total_ms"],
            "evaluation": data["evaluation"],
        }

    return run


def run_all(questions: list[GoldQuestion], runner: Runner) -> list[dict[str, Any]]:
    records = []
    for number, question in enumerate(questions, 1):
        print(f"[{number}/{len(questions)}] {question.id} ...", end=" ", flush=True)
        started = time.perf_counter()
        record: dict[str, Any] = {"id": question.id, "category": question.category}
        try:
            record.update(runner(question))
            evaluation = record["evaluation"]
            print(
                ("PASS" if evaluation["passed"] else "FAIL")
                + f" ({record['elapsed_ms'] / 1000:.1f} s)"
            )
        except (GenerationError, OSError, KeyError) as error:
            record.update({"error": str(error), "evaluation": None})
            print(f"ERROR {error}")
        record.setdefault("elapsed_ms", int((time.perf_counter() - started) * 1000))
        records.append(record)
    return records


def summaries(
    label: str, records: list[dict[str, Any]]
) -> tuple[GenerationSummary, list[GenerationSummary]]:
    def summary(name: str, chosen: list[dict[str, Any]]) -> GenerationSummary:
        evaluations = [
            GenerationEvaluation.model_validate(r["evaluation"]) if r["evaluation"] else None
            for r in chosen
        ]
        latencies = [r["elapsed_ms"] for r in chosen if r["evaluation"]]
        return summarize_generation(name, evaluations, latencies)

    categories = sorted({r["category"] for r in records})
    return summary(label, records), [
        summary(c, [r for r in records if r["category"] == c]) for c in categories
    ]


def fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.0%}"


def print_report(overall: GenerationSummary, by_category: list[GenerationSummary], records) -> None:
    header = f"{'':<22}{'n':>4}{'pass':>7}{'status':>8}{'facts':>7}{'recall':>8}{'prec.':>7}"
    print("\n" + header + f"{'forbid':>8}{'unsrc':>7}{'leak':>6}{'err':>5}")
    for s in [overall, *by_category]:
        print(
            f"{s.label:<22}{s.questions:>4}{fmt(s.pass_rate):>7}{fmt(s.status_ok):>8}"
            f"{fmt(s.fact_coverage):>7}{fmt(s.source_recall):>8}{fmt(s.source_precision):>7}"
            f"{s.forbidden_cited:>8}{s.unsupported_numbers:>7}{s.prompt_leaks:>6}{s.errors:>5}"
        )
    if overall.latency_p50_ms is not None:
        p50, p95 = overall.latency_p50_ms / 1000, (overall.latency_p95_ms or 0) / 1000
        print(f"\nlatency p50 {p50:.1f} s · p95 {p95:.1f} s")
    failed = [r for r in records if not (r["evaluation"] and r["evaluation"]["passed"])]
    if failed:
        print("\nFailed questions:")
    for record in failed:
        if record["evaluation"] is None:
            print(f"  {record['id']}: ERROR {record['error']}")
            continue
        print(f"  {record['id']} [{record['category']}] {record['answer'][:90]!r}")
        for problem in record["evaluation"]["problems"]:
            print(f"      - {problem}")


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("e2e", "oracle"), default="e2e")
    parser.add_argument("--label", default=None, help="results file: generation_<label>.json")
    parser.add_argument("--only", default="", help="comma-separated question ids, e.g. Q001,Q016")
    parser.add_argument("--category", default="", help="only one category, e.g. csv_only")
    parser.add_argument("--api-url", default="http://localhost:8000", help="e2e: the running API")
    parser.add_argument("--limit", type=int, default=10, help="e2e: chunks given to the model")
    parser.add_argument("--temperature", type=float, default=0.0, help="oracle: 0 = repeatable")
    parser.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    arguments = parser.parse_args(argv)

    questions = load_gold(settings.gold_path)
    if arguments.only:
        wanted = {item.strip() for item in arguments.only.split(",") if item.strip()}
        questions = [q for q in questions if q.id in wanted]
    if arguments.category:
        questions = [q for q in questions if q.category == arguments.category]
    if not questions:
        print("ERROR: no question selected.")
        return 1
    rules = yaml.safe_load(settings.rules_path.read_text(encoding="utf-8"))
    day = reference_date(rules)

    if arguments.mode == "oracle":
        chunks = load_chunks(arguments.chunks_dir)
        if not chunks:
            print(f"ERROR: no chunk in {arguments.chunks_dir} (run scripts/chunk_documents.py).")
            return 1
        generator = OpenAICompatibleGenerator(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            temperature=arguments.temperature,
            max_tokens=settings.llm_max_tokens,
            timeout_seconds=settings.llm_timeout_seconds,
        )
        runner = oracle_runner(generator, chunks, rules, day)
        print(
            f"oracle mode: {settings.llm_model} at {settings.llm_base_url},"
            f" temperature {arguments.temperature}"
        )
    else:
        runner = api_runner(arguments.api_url, arguments.limit, day)
        print(
            f"e2e mode: API {arguments.api_url}, {arguments.limit} chunks (temperature = the API's)"
        )

    records = run_all(questions, runner)
    label = arguments.label or f"{arguments.mode}_{datetime.now():%Y%m%d_%H%M}"
    overall, by_category = summaries(label, records)
    print_report(overall, by_category, records)

    arguments.results_dir.mkdir(parents=True, exist_ok=True)
    output = arguments.results_dir / f"generation_{label}.json"
    output.write_text(
        json.dumps(
            {
                "settings": {
                    "mode": arguments.mode,
                    "model": settings.llm_model,
                    "limit": arguments.limit if arguments.mode == "e2e" else None,
                    "temperature": arguments.temperature if arguments.mode == "oracle" else None,
                    "reference_date": day.isoformat(),
                    "run_at": datetime.now().isoformat(timespec="seconds"),
                },
                "overall": overall.model_dump(),
                "by_category": [s.model_dump() for s in by_category],
                "questions": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
