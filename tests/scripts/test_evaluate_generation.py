"""Tests of the generation evaluation runner (fake generator, fake API: no model)."""

from datetime import date

from peopleknowledge.evaluation.gold import GoldQuestion, Source
from peopleknowledge.generation import (
    BuiltPrompt,
    GeneratedAnswer,
    GenerationError,
    GenerationResult,
    Generator,
)
from scripts.evaluate_generation import api_runner, oracle_chunks, oracle_runner, run_all, summaries
from tests.api.test_answer_api import chunk

EXPECTED = Source(document_id="HR-005", section="4.2")
ACCEPTABLE = Source(document_id="HR-005", section="6.1")
RULES = {"probation": {"max_days": 2}}


def gold(**overrides) -> GoldQuestion:
    data = {
        "id": "Q001",
        "question": "Congés pendant mon essai ?",
        "expected_sources": [EXPECTED],
        "acceptable_sources": [ACCEPTABLE],
        "expected_answer": "2 jours au maximum.",
        "rule_refs": ["probation.max_days"],
        "category": "exception_elsewhere",
        "difficulty": "easy",
    }
    data.update(overrides)
    return GoldQuestion.model_validate(data)


CHUNKS = {
    EXPECTED: [chunk("4.2", "pendant l'essai : 2 jours au maximum")],
    ACCEPTABLE: [chunk("6.1", "report des jours")],
}


class ScriptedGenerator(Generator):
    model = "fake"

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.prompts: list[BuiltPrompt] = []

    def generate(self, prompt: BuiltPrompt) -> GenerationResult:
        self.prompts.append(prompt)
        if self.answer == "fail":
            raise GenerationError("invalid JSON")
        return GenerationResult(
            answer=GeneratedAnswer(status="complete", answer=self.answer, citations=["S1"]),
            model=self.model,
            elapsed_ms=10,
        )


def test_oracle_context_puts_expected_sources_first() -> None:
    selected = oracle_chunks(gold(), CHUNKS)

    assert [item.chunk.chunk_id for item in selected] == ["HR-005-v1.0#4.2", "HR-005-v1.0#6.1"]


def test_oracle_runner_evaluates_the_answer_of_the_perfect_context() -> None:
    generator = ScriptedGenerator("Oui, 2 jours au maximum.")
    run = oracle_runner(generator, CHUNKS, RULES, date(2026, 11, 4))

    record = run(gold())

    assert record["evaluation"]["passed"] is True
    assert record["citations"] == [EXPECTED.model_dump()]
    assert "2 jours au maximum" in generator.prompts[0].messages[1].content


def test_failures_are_recorded_and_counted() -> None:
    records = run_all(
        [gold(), gold(id="Q002")],
        oracle_runner(ScriptedGenerator("fail"), CHUNKS, RULES, date(2026, 11, 4)),
    )

    overall, by_category = summaries("t", records)
    assert [r["error"] for r in records] == ["invalid JSON", "invalid JSON"]
    assert overall.errors == 2 and overall.pass_rate == 0
    assert by_category[0].label == "exception_elsewhere"


def test_api_runner_sends_the_gold_id_and_reads_the_api_evaluation() -> None:
    sent: list[tuple[str, dict]] = []

    def fake_post(url: str, body: dict) -> dict:
        sent.append((url, body))
        return {
            "status": "complete",
            "answer": "2 jours.",
            "citations": [{"source": EXPECTED.model_dump()}],
            "retrieved_count": 5,
            "timing": {"total_ms": 1200},
            "evaluation": {"passed": True, "problems": []},
        }

    record = api_runner("http://api/", 5, date(2026, 11, 4), post_json=fake_post)(
        gold(category="history")
    )

    url, body = sent[0]
    assert url == "http://api/api/answer"
    assert body["gold_id"] == "Q001" and body["include_archived"] is True
    assert body["reference_date"] == "2026-11-04"
    assert record["evaluation"]["passed"] is True and record["elapsed_ms"] == 1200
