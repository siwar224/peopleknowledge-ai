"""Measure a generated answer: automatic checks, no LLM judge.

Some checks need a gold question (what the right answer is), others only need
what the model was given (the prompt):

With a gold question:
- status: an answerable question should get ``complete`` (``partial`` is
  weak); an unanswerable one must NOT get ``complete``;
- key facts: share of the question's numbers (evaluation/key_facts.py) found in
  the answer;
- citations: recall of the expected sources, precision (cited sources that are
  expected or acceptable), forbidden sources cited (e.g. the archived v1).

For any question:
- unsupported numbers: numbers of the answer that the prompt does not contain
  (nor the sum of two of its numbers, e.g. 24 + 2 = 26 days): almost always a
  hallucination in an HR answer. Limit: a wrong figure that happens to be such
  a sum (48 + 24 = 72 hours) is not caught; the key facts check covers it;
- prompt leak: the answer is text copied from the system prompt (the Q016 bug,
  "A supported answer.");
- repeated question: the answer starts by copying the question.

Nothing here calls a model, Qdrant or the generation package: values come in,
a verdict comes out, so the API and scripts/evaluate_generation.py share it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any, Literal

from pydantic import BaseModel, Field

from .gold import GoldQuestion, Source
from .key_facts import TermVocabulary, derive_key_facts, extract_numbers, fact_present

StatusVerdict = Literal["ok", "weak", "wrong"]

_MIN_LEAK_CHARS = 12  # shorter answers ("Oui.") naturally appear anywhere


class SourceCheck(BaseModel):
    expected: list[Source]
    expected_cited: list[Source]
    expected_missing: list[Source]
    cited_off_gold: list[Source]  # cited, but neither expected nor acceptable
    forbidden_cited: list[Source]
    recall: float | None  # None when the question has no expected source
    precision: float | None  # None when nothing is cited


class GenerationEvaluation(BaseModel):
    """Every check of one answer; gold-dependent fields are None without a gold question."""

    gold_id: str | None = None
    category: str | None = None
    expected_status: str | None = None
    status: str
    status_verdict: StatusVerdict | None = None
    key_facts: list[str] = Field(default_factory=list)
    facts_found: list[str] = Field(default_factory=list)
    facts_missing: list[str] = Field(default_factory=list)
    fact_coverage: float | None = None  # None when the question has no key fact
    sources: SourceCheck | None = None
    unsupported_numbers: list[str] = Field(default_factory=list)
    prompt_leak: bool = False
    repeats_question: bool = False
    passed: bool | None = None  # None without a gold question
    problems: list[str] = Field(default_factory=list)  # short French explanations for the UI


def _status_verdict(question: GoldQuestion, status: str) -> tuple[str, StatusVerdict]:
    if question.category == "unanswerable":
        return "not_found", "wrong" if status == "complete" else "ok"
    return "complete", {"complete": "ok", "partial": "weak"}.get(status, "wrong")  # type: ignore[return-value]


def _source_check(question: GoldQuestion, cited: list[Source]) -> SourceCheck:
    expected = list(question.expected_sources)
    allowed = {*expected, *question.acceptable_sources}
    forbidden = set(question.forbidden_sources)
    unique = list(dict.fromkeys(cited))
    expected_cited = [s for s in expected if s in unique]
    return SourceCheck(
        expected=expected,
        expected_cited=expected_cited,
        expected_missing=[s for s in expected if s not in unique],
        cited_off_gold=[s for s in unique if s not in allowed and s not in forbidden],
        forbidden_cited=[s for s in unique if s in forbidden],
        recall=len(expected_cited) / len(expected) if expected else None,
        precision=sum(s in allowed for s in unique) / len(unique) if unique else None,
    )


def _as_number(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None  # a clock time such as "8:30"


def unsupported_numbers(answer: str, prompt_text: str) -> list[str]:
    """Numbers of the answer that the prompt neither contains nor gives as a sum of two."""

    given = extract_numbers(prompt_text)
    values = [v for v in (_as_number(n) for n in given) if v is not None]
    computed = {a + b for a in values for b in values}
    result = []
    for number in sorted(extract_numbers(answer), key=lambda n: (len(n), n)):
        value = _as_number(number)
        if number in given or (value is not None and value in computed):
            continue
        result.append(number)
    return result


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def leaks_prompt(answer: str, system_prompt: str) -> bool:
    """The answer is a passage of the system prompt (e.g. its JSON example)."""

    squashed = _squash(answer).rstrip(".")
    return len(squashed) >= _MIN_LEAK_CHARS and squashed in _squash(system_prompt)


def repeats_question(answer: str, question: str) -> bool:
    question_text = _squash(question).rstrip("?").strip()
    return bool(question_text) and _squash(answer).startswith(question_text)


def evaluate_answer(
    *,
    question_text: str,
    status: str,
    answer: str,
    cited: list[Source],
    prompt_user_text: str,
    system_prompt: str,
    gold: GoldQuestion | None = None,
    rules: dict[str, Any] | None = None,
    vocabulary: TermVocabulary | None = None,
) -> GenerationEvaluation:
    """All checks of one answer. ``prompt_user_text`` = what the model was given."""

    result = GenerationEvaluation(
        status=status,
        unsupported_numbers=unsupported_numbers(answer, prompt_user_text),
        prompt_leak=leaks_prompt(answer, system_prompt),
        repeats_question=repeats_question(answer, question_text),
    )
    problems: list[str] = []
    if result.prompt_leak:
        problems.append("La réponse est un texte recopié du prompt système.")
    if result.repeats_question:
        problems.append("La réponse commence par répéter la question.")
    if result.unsupported_numbers:
        problems.append(
            "Chiffre(s) absent(s) des sources : " + ", ".join(result.unsupported_numbers) + "."
        )

    if gold is None:
        result.problems = problems
        return result

    expected_status, verdict = _status_verdict(gold, status)
    derived = derive_key_facts(gold.rule_refs, rules or {}, gold.not_required_facts, vocabulary)
    facts = [fact.value for fact in derived]
    found = [fact.value for fact in derived if fact_present(fact, answer)]
    missing = [fact.value for fact in derived if not fact_present(fact, answer)]
    sources = _source_check(gold, cited)

    if verdict == "wrong":
        problems.insert(0, f"Statut « {status} » au lieu de « {expected_status} ».")
    elif verdict == "weak":
        problems.append("Statut « partial » : le modèle estime sa réponse incomplète.")
    if missing:
        problems.append("Chiffre(s) clé(s) manquant(s) : " + ", ".join(missing) + ".")
    if sources.forbidden_cited:
        names = ", ".join(f"{s.document_id} §{s.section}" for s in sources.forbidden_cited)
        problems.append(f"Source interdite citée : {names}.")
    if sources.expected_missing:
        names = ", ".join(f"{s.document_id} §{s.section}" for s in sources.expected_missing)
        problems.append(f"Source attendue non citée : {names}.")

    passed = (
        verdict != "wrong"
        and not missing
        and not sources.forbidden_cited
        and not result.unsupported_numbers
        and not result.prompt_leak
    )
    return result.model_copy(
        update={
            "gold_id": gold.id,
            "category": gold.category,
            "expected_status": expected_status,
            "status_verdict": verdict,
            "key_facts": facts,
            "facts_found": found,
            "facts_missing": missing,
            "fact_coverage": len(found) / len(facts) if facts else None,
            "sources": sources,
            "passed": passed,
            "problems": problems,
        }
    )


def _mean(values: Iterable[float | None]) -> float | None:
    kept = [v for v in values if v is not None]
    return round(sum(kept) / len(kept), 3) if kept else None


def _percentile(values: list[int], share: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(share * (len(ordered) - 1)))]


class GenerationSummary(BaseModel):
    label: str
    questions: int
    errors: int  # generation failures (invalid JSON, server down...): counted as failed
    pass_rate: float
    status_ok: float
    status_wrong: int
    fact_coverage: float | None
    source_recall: float | None
    source_precision: float | None
    forbidden_cited: int
    unsupported_numbers: int  # answers with at least one
    prompt_leaks: int
    latency_p50_ms: int | None
    latency_p95_ms: int | None


def summarize_generation(
    label: str,
    evaluations: list[GenerationEvaluation | None],
    latencies_ms: list[int] | None = None,
) -> GenerationSummary:
    """``None`` in ``evaluations`` = a question whose generation failed."""

    done = [e for e in evaluations if e is not None]
    count = len(evaluations)
    return GenerationSummary(
        label=label,
        questions=count,
        errors=count - len(done),
        pass_rate=round(sum(bool(e.passed) for e in done) / count, 3) if count else 0.0,
        status_ok=round(sum(e.status_verdict == "ok" for e in done) / count, 3) if count else 0.0,
        status_wrong=sum(e.status_verdict == "wrong" for e in done),
        fact_coverage=_mean(e.fact_coverage for e in done),
        source_recall=_mean(e.sources.recall for e in done if e.sources),
        source_precision=_mean(e.sources.precision for e in done if e.sources),
        forbidden_cited=sum(bool(e.sources and e.sources.forbidden_cited) for e in done),
        unsupported_numbers=sum(bool(e.unsupported_numbers) for e in done),
        prompt_leaks=sum(e.prompt_leak for e in done),
        latency_p50_ms=_percentile(latencies_ms or [], 0.5),
        latency_p95_ms=_percentile(latencies_ms or [], 0.95),
    )
