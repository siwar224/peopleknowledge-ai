"""Measure the retrieval step against the gold questions.

For each question, the retriever returns chunks in rank order; each chunk is
mapped to its SOURCE (document + section / CSV row / card) and compared with
the question's expected, acceptable and forbidden sources.

Indicators (standard in information retrieval):

- hit@k: share of questions whose expected sources are ALL in the first k
  results. A multi_source question needs both of its sections.
- MRR (mean reciprocal rank): mean of 1 / rank of the first expected source
  (1.0 = always first, 0.5 = second on average, 0 when missing).
- useful@k: share of questions with an expected OR acceptable source in the
  first k results. hit@k is strict (the best section); useful@k tells whether
  the generation model would at least receive a passage holding the answer.
- forbidden: questions for which a forbidden source (e.g. the archived v1)
  came back in the results.

Unanswerable questions have no expected source: they are left out of hit@k and
MRR, and only their best score is reported (a low score could later be used to
detect "the corpus does not contain the answer").

Nothing here knows Qdrant or the embedding model: the retriever is a function,
so the same measure can compare models, chunk sizes or search strategies.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from .gold import GoldQuestion, Source

HIT_LEVELS = (1, 3, 5)


@dataclass(frozen=True)
class Retrieved:
    """One search result, reduced to what the measure needs."""

    source: Source
    chunk_id: str
    score: float


# (question text, include archived documents?, user context) -> results in rank order.
# The user context (department, job...) lets a retriever apply the profile option.
Retriever = Callable[[str, bool, dict], list[Retrieved]]


@dataclass
class QuestionResult:
    question: GoldQuestion
    include_archived: bool
    retrieved: list[Retrieved]
    expected_ranks: dict[Source, int | None] = field(default_factory=dict)  # None = not retrieved
    acceptable_ranks: dict[Source, int] = field(default_factory=dict)
    forbidden_ranks: dict[Source, int] = field(default_factory=dict)

    @property
    def answerable(self) -> bool:
        return bool(self.question.expected_sources)

    @property
    def first_rank(self) -> int | None:
        """Rank of the best-placed expected source."""

        ranks = [rank for rank in self.expected_ranks.values() if rank is not None]
        return min(ranks) if ranks else None

    @property
    def complete_rank(self) -> int | None:
        """Rank at which ALL expected sources have been seen (None if one is missing)."""

        ranks = list(self.expected_ranks.values())
        if not ranks or any(rank is None for rank in ranks):
            return None
        return max(ranks)  # type: ignore[type-var]

    def hit(self, k: int) -> bool:
        return self.complete_rank is not None and self.complete_rank <= k

    @property
    def useful_rank(self) -> int | None:
        """Rank of the first source that can answer: expected OR acceptable."""

        ranks = [r for r in self.expected_ranks.values() if r is not None]
        ranks += list(self.acceptable_ranks.values())
        return min(ranks) if ranks else None

    def useful(self, k: int) -> bool:
        return self.useful_rank is not None and self.useful_rank <= k

    @property
    def reciprocal_rank(self) -> float:
        return 1 / self.first_rank if self.first_rank else 0.0

    @property
    def top_score(self) -> float | None:
        return self.retrieved[0].score if self.retrieved else None


def first_ranks(retrieved: list[Retrieved]) -> dict[Source, int]:
    """Best rank of every source (a section split in two parts appears twice)."""

    ranks: dict[Source, int] = {}
    for rank, item in enumerate(retrieved, 1):
        ranks.setdefault(item.source, rank)
    return ranks


def needs_archived(question: GoldQuestion) -> bool:
    """Only explicit questions about the past search the archived documents."""

    return question.category == "history"


def evaluate_question(
    question: GoldQuestion, retrieved: list[Retrieved], include_archived: bool
) -> QuestionResult:
    ranks = first_ranks(retrieved)
    return QuestionResult(
        question=question,
        include_archived=include_archived,
        retrieved=retrieved,
        expected_ranks={source: ranks.get(source) for source in question.expected_sources},
        acceptable_ranks={s: ranks[s] for s in question.acceptable_sources if s in ranks},
        forbidden_ranks={s: ranks[s] for s in question.forbidden_sources if s in ranks},
    )


def evaluate(
    questions: Iterable[GoldQuestion],
    retriever: Retriever,
    archived: str = "history",
) -> list[QuestionResult]:
    """``archived``: "history" (default: only history questions search archived
    documents), "never" or "always" (stress test: does the old version come back?)."""

    if archived not in {"history", "never", "always"}:
        raise ValueError(f"archived must be history, never or always, not {archived!r}.")
    results = []
    for question in questions:
        include = archived == "always" or (archived == "history" and needs_archived(question))
        context = question.user_context.model_dump(exclude_none=True)
        retrieved = retriever(question.question, include, context)
        results.append(evaluate_question(question, retrieved, include))
    return results


@dataclass
class Summary:
    label: str
    questions: int  # answerable questions only
    hits: dict[int, float]  # k -> share of questions (strict: all expected sources)
    useful: dict[int, float]  # k -> share with an expected OR acceptable source in the top k
    mrr: float
    forbidden: int  # questions with at least one forbidden source retrieved


def summarize(label: str, results: list[QuestionResult]) -> Summary:
    answerable = [result for result in results if result.answerable]
    count = len(answerable)
    return Summary(
        label=label,
        questions=count,
        hits={k: sum(r.hit(k) for r in answerable) / count if count else 0.0 for k in HIT_LEVELS},
        useful={
            k: sum(r.useful(k) for r in answerable) / count if count else 0.0 for k in HIT_LEVELS
        },
        mrr=sum(r.reciprocal_rank for r in answerable) / count if count else 0.0,
        forbidden=sum(1 for r in results if r.forbidden_ranks),
    )


def summaries_by_category(results: list[QuestionResult]) -> list[Summary]:
    categories = sorted({r.question.category for r in results if r.answerable})
    return [
        summarize(category, [r for r in results if r.question.category == category])
        for category in categories
    ]
