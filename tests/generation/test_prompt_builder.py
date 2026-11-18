"""Tests for the stable instructions and dynamic RAG prompt.

They check the prompt's CONTRACT (message roles, block order, tags, status values,
JSON fields), not its exact wording, so the instructions can be reworded freely.
"""

import re
from datetime import date

import pytest

from peopleknowledge.evaluation.gold import UserContext
from peopleknowledge.generation import BuiltContext, build_prompt


def _block(message: str, tag: str) -> str:
    """Content between <tag> and </tag>, stripped (fails the test if the tag is missing)."""
    match = re.search(rf"<{tag}>\n(.*?)\n</{tag}>", message, flags=re.DOTALL)
    assert match, f"<{tag}> block missing"
    return match.group(1).strip()


def test_builds_one_system_message_then_one_user_message() -> None:
    prompt = build_prompt(
        "Puis-je poser des congés ?",
        BuiltContext(text="", sources=[]),
        reference_date=date(2026, 11, 4),
    )

    assert [message.role for message in prompt.messages] == ["system", "user"]
    system = prompt.messages[0].content
    # The three statuses the generator accepts, and the three JSON fields it parses.
    for status in ("complete", "partial", "not_found"):
        assert status in system
    # no example answer the model could copy (the Q016 bug: "A supported answer.")
    assert '"answer": "..."' in system
    for field in ("status", "answer", "citations"):
        assert f'"{field}"' in system
    assert "USER CONTEXT" in system and "SOURCES" in system


def test_system_message_does_not_depend_on_the_request() -> None:
    first = build_prompt("Question A ?", BuiltContext(text="", sources=[]),
                         reference_date=date(2026, 11, 4))
    second = build_prompt(
        "Question B ?",
        BuiltContext(
            text="--- SOURCE [S1] ---\nx\n--- END SOURCE [S1] ---",
            sources=[
                {
                    "source_id": "S1",
                    "rank": 1,
                    "score": 0.5,
                    "chunk_id": "HR-011-v2.0#3.3",
                    "source": {"document_id": "HR-011", "section": "3.3"},
                    "title": "Politique de télétravail",
                    "version": "2.0",
                    "lifecycle_status": "active",
                    "effective_from": "2026-09-01",
                }
            ],
        ),
        reference_date=date(2025, 1, 1),
        user_context=UserContext(department="sales"),
    )

    assert first.messages[0] == second.messages[0]


def test_dynamic_information_is_ordered_and_the_question_comes_after_the_sources() -> None:
    question = "Je suis en période d'essai. Est-ce que je peux poser des congés ?"
    source_text = "--- SOURCE [S1] ---\nContent:\nRègle test\n--- END SOURCE [S1] ---"
    prompt = build_prompt(
        question,
        BuiltContext(
            text=source_text,
            sources=[
                {
                    "source_id": "S1",
                    "rank": 1,
                    "score": 0.8,
                    "chunk_id": "HR-005-v1.2#4.2",
                    "source": {"document_id": "HR-005", "section": "4.2"},
                    "title": "Politique des congés annuels",
                    "version": "1.2",
                    "lifecycle_status": "active",
                    "effective_from": "2026-01-01",
                    "pages": [3],
                }
            ],
        ),
        reference_date=date(2026, 11, 4),
        user_context=UserContext(
            department="product",
            employment_type="cdi",
            in_probation=True,
        ),
    )

    user_message = prompt.messages[1].content
    positions = [
        user_message.index("2026-11-04"),
        user_message.index("<user_context>"),
        user_message.index("<retrieved_sources>"),
        user_message.index("<employee_question>"),
    ]
    assert positions == sorted(positions)
    assert _block(user_message, "retrieved_sources") == source_text
    assert _block(user_message, "employee_question") == question
    assert user_message.count(question) == 1
    assert prompt.allowed_source_ids == ["S1"]


def test_user_context_has_fixed_fields_and_explicit_missing_values() -> None:
    prompt = build_prompt(
        "Quelle règle s'applique ?",
        BuiltContext(text="", sources=[]),
        reference_date=date(2026, 11, 4),
        user_context=UserContext(department="engineering", in_probation=False),
    )

    user_message = _block(prompt.messages[1].content, "user_context")
    assert "department: engineering" in user_message
    assert "job_family: not provided" in user_message
    assert "employment_type: not provided" in user_message
    assert "in_probation: false" in user_message
    assert "seniority_years: not provided" in user_message


def test_empty_context_is_explicit() -> None:
    prompt = build_prompt(
        "Quel est mon salaire ?",
        BuiltContext(text="", sources=[]),
        reference_date=date(2026, 11, 4),
    )

    assert _block(prompt.messages[1].content, "retrieved_sources") == "No sources were retrieved."
    assert prompt.allowed_source_ids == []


def test_question_is_stripped_before_being_inserted() -> None:
    prompt = build_prompt(
        "  Puis-je télétravailler ?\n",
        BuiltContext(text="", sources=[]),
        reference_date=date(2026, 11, 4),
    )

    assert _block(prompt.messages[1].content, "employee_question") == "Puis-je télétravailler ?"


def test_building_the_same_prompt_is_deterministic() -> None:
    arguments = {
        "question": "Puis-je télétravailler ?",
        "context": BuiltContext(text="", sources=[]),
        "reference_date": date(2026, 11, 4),
    }

    assert build_prompt(**arguments) == build_prompt(**arguments)


def test_blank_question_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be blank"):
        build_prompt("  ", BuiltContext(text="", sources=[]), reference_date=date(2026, 11, 4))
