"""Retrieval indicators (hit@k, MRR, forbidden sources), with a scripted fake retriever."""

import pytest
from app.rag.evaluation.gold import GoldQuestion, Source
from app.rag.evaluation.retrieval_eval import (
    Retrieved,
    evaluate,
    evaluate_question,
    summaries_by_category,
    summarize,
)


def source(text: str) -> Source:
    document_id, section = text.split(" ")
    return Source(document_id=document_id, section=section)


def gold(qid: str, expected: list[str], category: str = "direct", forbidden: tuple = ()) -> GoldQuestion:
    return GoldQuestion(
        id=qid,
        question=f"Question {qid} ?",
        expected_sources=[source(s) for s in expected],
        forbidden_sources=[source(s) for s in forbidden],
        expected_answer="Réponse.",
        category=category,
        difficulty="easy",
    )


def results(*sources: str) -> list[Retrieved]:
    return [
        Retrieved(source=source(s), chunk_id=f"chunk-{n}", score=1 - n / 10)
        for n, s in enumerate(sources)
    ]


def test_ranks_hits_and_reciprocal_rank() -> None:
    result = evaluate_question(
        gold("Q001", ["HR-005 4.2"]), results("HR-001 6.1", "HR-005 4.2", "HR-005 8"), False
    )

    assert result.expected_ranks == {source("HR-005 4.2"): 2}
    assert (result.first_rank, result.complete_rank) == (2, 2)
    assert [result.hit(k) for k in (1, 3, 5)] == [False, True, True]
    assert result.reciprocal_rank == 0.5


def test_missing_source_counts_as_zero() -> None:
    result = evaluate_question(gold("Q001", ["HR-005 4.2"]), results("HR-001 6.1"), False)

    assert result.complete_rank is None
    assert not result.hit(5)
    assert result.reciprocal_rank == 0.0


def test_multi_source_question_needs_all_its_sources() -> None:
    question = gold("Q019", ["HR-005 5.2", "HR-005 5.1"], category="multi_source")

    both = evaluate_question(question, results("HR-005 5.1", "HR-006 5", "HR-005 5.2"), False)
    assert (both.first_rank, both.complete_rank) == (1, 3)
    assert [both.hit(k) for k in (1, 3)] == [False, True]
    assert both.reciprocal_rank == 1.0  # MRR looks at the FIRST expected source

    one_only = evaluate_question(question, results("HR-005 5.1", "HR-006 5"), False)
    assert one_only.complete_rank is None and not one_only.hit(5)


def test_a_section_in_two_parts_keeps_its_best_rank() -> None:
    result = evaluate_question(
        gold("Q001", ["HR-001 18"]), results("HR-001 7", "HR-001 18", "HR-001 18"), False
    )

    assert result.expected_ranks == {source("HR-001 18"): 2}


def test_forbidden_sources_are_reported() -> None:
    question = gold("Q003", ["HR-011 6.3"], category="version_conflict", forbidden=("HR-029 6",))

    result = evaluate_question(question, results("HR-029 6", "HR-011 6.3"), True)

    assert result.forbidden_ranks == {source("HR-029 6"): 1}


def test_only_history_questions_search_archived_documents_by_default() -> None:
    calls: list[tuple[str, bool]] = []

    def retriever(question: str, include_archived: bool) -> list[Retrieved]:
        calls.append((question, include_archived))
        return []

    questions = [gold("Q001", ["HR-005 4.2"]), gold("Q004", ["HR-029 3.1"], category="history")]

    evaluate(questions, retriever)
    assert [archived for _, archived in calls] == [False, True]
    calls.clear()
    evaluate(questions, retriever, archived="always")
    assert [archived for _, archived in calls] == [True, True]
    calls.clear()
    evaluate(questions, retriever, archived="never")
    assert [archived for _, archived in calls] == [False, False]

    with pytest.raises(ValueError, match="archived must be"):
        evaluate(questions, retriever, archived="sometimes")


def test_summary_ignores_unanswerable_questions_and_groups_by_category() -> None:
    unanswerable = GoldQuestion(
        id="Q006", question="Quel salaire ?", expected_sources=[], expected_answer="Inconnu.",
        category="unanswerable", difficulty="medium",
    )
    evaluated = [
        evaluate_question(gold("Q001", ["HR-005 4.2"]), results("HR-005 4.2"), False),  # rank 1
        evaluate_question(gold("Q002", ["HR-005 6.1"]), results("HR-001 7", "HR-005 6.1"), False),  # 2
        evaluate_question(
            gold("Q003", ["HR-011 6.3"], category="version_conflict", forbidden=("HR-029 6",)),
            results("HR-029 6"),  # missing, and a forbidden source came back
            True,
        ),
        evaluate_question(unanswerable, results("HR-001 8.1"), False),
    ]

    overall = summarize("all", evaluated)
    assert overall.questions == 3  # the unanswerable one is left out
    assert overall.hits == {1: pytest.approx(1 / 3), 3: pytest.approx(2 / 3), 5: pytest.approx(2 / 3)}
    assert overall.mrr == pytest.approx((1 + 0.5 + 0) / 3)
    assert overall.forbidden == 1

    by_category = {s.label: s for s in summaries_by_category(evaluated)}
    assert set(by_category) == {"direct", "version_conflict"}
    assert by_category["direct"].hits[1] == 0.5
    assert evaluated[3].top_score == 1.0 and not evaluated[3].answerable


def test_api_retriever_asks_the_search_endpoint() -> None:
    from scripts.evaluate_retrieval import api_retriever

    calls: list[tuple[str, dict]] = []

    def fake_post(url: str, body: dict) -> dict:
        calls.append((url, body))
        return {
            "results": [
                {"source": {"document_id": "HR-005", "section": "4.2"}, "chunk_id": "HR-005-v1.2#4.2",
                 "score": 0.8, "rank": 1}
            ]
        }

    retrieve = api_retriever("http://localhost:8001/", limit=10, post_json=fake_post)
    found = retrieve("Congés pendant l'essai ?", True)

    assert calls == [
        ("http://localhost:8001/api/search",
         {"question": "Congés pendant l'essai ?", "limit": 10, "include_archived": True})
    ]
    assert found == [Retrieved(source=source("HR-005 4.2"), chunk_id="HR-005-v1.2#4.2", score=0.8)]


def test_useful_counts_an_acceptable_source_too() -> None:
    question = GoldQuestion(
        id="Q007", question="Combien de jours de congé ?",
        expected_sources=[source("HR-005 2.1")], acceptable_sources=[source("HR-001 6.1")],
        expected_answer="24 jours.", category="direct", difficulty="easy",
    )

    result = evaluate_question(question, results("HR-001 6.1", "HR-005 8", "HR-005 2.1"), False)

    assert not result.hit(1) and result.hit(3)  # strict: the expected section is third
    assert result.useful_rank == 1 and result.useful(1)  # but a passage with the answer is first
    summary = summarize("all", [result])
    assert summary.hits[1] == 0.0 and summary.useful[1] == 1.0
