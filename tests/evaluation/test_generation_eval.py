"""Tests of the automatic checks of a generated answer (no model involved)."""

from peopleknowledge.evaluation.generation_eval import (
    evaluate_answer,
    leaks_prompt,
    repeats_question,
    summarize_generation,
    unsupported_numbers,
)
from peopleknowledge.evaluation.gold import GoldQuestion, Source

RULES = {"leave": {"sick": "certificat sous 48 h", "base": 24, "bonus": 2}}
SYSTEM = 'Return JSON like {"answer": "A supported answer."}'
PROMPT = "SOURCES: certificat sous 48 h. Droit de base 24 jours, +2 jours d'ancienneté."
EXPECTED = Source(document_id="HR-005", section="9")
ACCEPTABLE = Source(document_id="HR-001", section="6")
FORBIDDEN = Source(document_id="HR-029", section="3")


def gold(**overrides) -> GoldQuestion:
    data = {
        "id": "Q016",
        "question": "Je suis malade pendant mes congés ?",
        "expected_sources": [EXPECTED],
        "acceptable_sources": [ACCEPTABLE],
        "forbidden_sources": [FORBIDDEN],
        "expected_answer": "Oui, avec un certificat sous 48 h.",
        "rule_refs": ["leave.sick"],
        "category": "vocabulary",
        "difficulty": "easy",
    }
    data.update(overrides)
    return GoldQuestion.model_validate(data)


def run(answer: str, *, status="complete", cited=(EXPECTED,), question=None, text=None):
    question = question or gold()
    return evaluate_answer(
        question_text=text or question.question,
        status=status,
        answer=answer,
        cited=list(cited),
        prompt_user_text=PROMPT,
        system_prompt=SYSTEM,
        gold=question,
        rules=RULES,
    )


def test_a_correct_answer_passes_every_check() -> None:
    result = run("Oui : envoyez un certificat médical dans les 48 heures.")

    assert result.passed
    assert result.status_verdict == "ok"
    assert result.facts_found == ["48"] and result.fact_coverage == 1.0
    assert result.sources.recall == 1.0 and result.sources.precision == 1.0
    assert result.problems == []


def test_a_missing_key_fact_fails() -> None:
    result = run("Oui, avec un certificat médical.")

    assert not result.passed
    assert result.facts_missing == ["48"]
    assert any("48" in problem for problem in result.problems)


def test_the_placeholder_answer_of_the_prompt_is_detected() -> None:
    result = run("A supported answer.")

    assert result.prompt_leak and not result.passed


def test_a_forbidden_source_fails_and_off_gold_sources_lower_precision() -> None:
    other = Source(document_id="HR-006", section="2")
    result = run("Oui, sous 48 h.", cited=[EXPECTED, other, FORBIDDEN])

    assert not result.passed
    assert result.sources.forbidden_cited == [FORBIDDEN]
    assert result.sources.cited_off_gold == [other]
    assert result.sources.precision == 1 / 3


def test_unanswerable_questions_must_not_get_a_complete_status() -> None:
    question = gold(
        category="unanswerable",
        expected_sources=[],
        rule_refs=[],
        expected_answer="Non disponible.",
    )

    assert run("Je ne sais pas.", status="not_found", cited=[], question=question).passed
    wrong = run("Vous avez 3 jours.", status="complete", cited=[], question=question)
    assert wrong.status_verdict == "wrong" and not wrong.passed


def test_partial_is_weak_but_can_pass() -> None:
    result = run("Oui, sous 48 h.", status="partial")

    assert result.status_verdict == "weak" and result.passed


def test_numbers_absent_from_the_prompt_are_flagged_but_sums_are_allowed() -> None:
    assert unsupported_numbers("Vous avez 26 jours (24 + 2).", PROMPT) == []
    assert unsupported_numbers("Sous 73 h ou 5 jours.", PROMPT) == ["5", "73"]


def test_leak_and_repetition_helpers() -> None:
    assert leaks_prompt("a supported   answer", SYSTEM)
    assert not leaks_prompt("Oui.", SYSTEM)  # too short to be a meaningful leak
    assert repeats_question(
        "Je suis malade pendant mes congés ? Oui.", "Je suis malade pendant mes congés ?"
    )


def test_without_gold_only_generic_checks_run() -> None:
    result = evaluate_answer(
        question_text="Une question libre",
        status="complete",
        answer="Il faut 99 jours.",
        cited=[],
        prompt_user_text=PROMPT,
        system_prompt=SYSTEM,
    )

    assert result.passed is None and result.sources is None
    assert result.unsupported_numbers == ["99"]


def test_summary_counts_errors_as_failures() -> None:
    good = run("Oui, sous 48 h.")
    bad = run("Oui.")

    summary = summarize_generation("t", [good, bad, None], [100, 300])

    assert summary.questions == 3 and summary.errors == 1
    assert summary.pass_rate == round(1 / 3, 3)
    assert summary.fact_coverage == 0.5
    assert summary.latency_p50_ms in {100, 300}
