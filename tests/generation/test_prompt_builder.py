"""Tests for the stable instructions and dynamic RAG prompt."""

from datetime import date

import pytest
from app.rag.evaluation.gold import UserContext
from app.rag.generation import BuiltContext, build_prompt


def test_builds_one_system_message_then_one_user_message() -> None:
    prompt = build_prompt(
        "Puis-je poser des congés ?",
        BuiltContext(text="", sources=[]),
        reference_date=date(2026, 11, 4),
    )

    assert [message.role for message in prompt.messages] == ["system", "user"]
    assert "Use only the USER CONTEXT and SOURCES" in prompt.messages[0].content
    assert '"status": "complete"' in prompt.messages[0].content
    assert "must be exactly \"complete\", \"partial\", or \"not_found\"" in (
        prompt.messages[0].content
    )
    assert "application displays the cited sources" in prompt.messages[0].content


def test_dynamic_information_is_ordered_and_the_question_is_last() -> None:
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
    assert user_message.index("REFERENCE DATE:") < user_message.index("USER CONTEXT:")
    assert user_message.index("USER CONTEXT:") < user_message.index("SOURCES:")
    assert user_message.index("SOURCES:") < user_message.index("QUESTION:")
    assert source_text in user_message
    assert user_message.endswith(question)


def test_user_context_has_fixed_fields_and_explicit_missing_values() -> None:
    prompt = build_prompt(
        "Quelle règle s'applique ?",
        BuiltContext(text="", sources=[]),
        reference_date=date(2026, 11, 4),
        user_context=UserContext(department="engineering", in_probation=False),
    )

    user_message = prompt.messages[1].content
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

    assert "SOURCES:\nNo sources were retrieved.\n\nQUESTION:" in prompt.messages[1].content


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
