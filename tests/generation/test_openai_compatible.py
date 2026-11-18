"""Tests for structured generation through an OpenAI-compatible client."""

import json
from datetime import date
from types import SimpleNamespace

import pytest

from peopleknowledge.generation import (
    BuiltContext,
    GenerationError,
    OpenAICompatibleGenerator,
    build_prompt,
)


class FakeCompletions:
    def __init__(self, content: str | None, *, choices: bool = True) -> None:
        self.content = content
        self.with_choices = choices
        self.calls: list[dict] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        result = []
        if self.with_choices:
            result.append(SimpleNamespace(message=SimpleNamespace(content=self.content)))
        return SimpleNamespace(choices=result)


class FakeClient:
    def __init__(self, completions: FakeCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


def prompt_with_source() -> object:
    context = BuiltContext.model_validate(
        {
            "text": "--- SOURCE [S1] ---\nContent:\nTest evidence\n--- END SOURCE [S1] ---",
            "sources": [
                {
                    "source_id": "S1",
                    "rank": 1,
                    "score": 0.8,
                    "chunk_id": "HR-005-v1.2#4.2",
                    "source": {"document_id": "HR-005", "section": "4.2"},
                    "title": "Annual leave policy",
                    "version": "1.2",
                    "lifecycle_status": "active",
                    "effective_from": "2026-01-01",
                    "pages": [3],
                }
            ],
        }
    )
    return build_prompt(
        "Puis-je poser des congés ?",
        context,
        reference_date=date(2026, 11, 4),
    )


def generator_with(content: str | None, *, choices: bool = True) -> tuple:
    completions = FakeCompletions(content, choices=choices)
    generator = OpenAICompatibleGenerator(
        base_url="http://localhost:8080/v1/",
        model="qwen-test",
        temperature=0.1,
        max_tokens=300,
        client=FakeClient(completions),
    )
    return generator, completions


def test_sends_chat_messages_and_parses_a_grounded_json_answer() -> None:
    content = json.dumps(
        {
            "status": "complete",
            "answer": "Vous pouvez poser deux jours.",
            "citations": ["S1"],
        }
    )
    generator, completions = generator_with(content)

    result = generator.generate(prompt_with_source())

    assert result.answer.status == "complete"
    assert result.answer.citations == ["S1"]
    assert result.model == "qwen-test"
    assert result.elapsed_ms >= 0
    call = completions.calls[0]
    assert call["model"] == "qwen-test"
    assert [message["role"] for message in call["messages"]] == ["system", "user"]
    assert call["temperature"] == 0.1
    assert call["max_tokens"] == 300
    assert call["response_format"] == {"type": "json_object"}
    assert generator.base_url == "http://localhost:8080/v1"


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("not JSON", "invalid answer JSON"),
        (
            json.dumps(
                {
                    "status": "complete",
                    "answer": "Réponse sans citation.",
                    "citations": [],
                }
            ),
            "needs at least one citation",
        ),
    ],
)
def test_rejects_malformed_or_internally_inconsistent_answers(
    content: str, message: str
) -> None:
    generator, _ = generator_with(content)

    with pytest.raises(GenerationError, match=message):
        generator.generate(prompt_with_source())


def test_inline_labels_become_citations_and_unknown_labels_are_dropped() -> None:
    content = json.dumps(
        {"status": "complete", "answer": "Deux jours [S1].", "citations": ["S2", "S1"]}
    )
    generator, _ = generator_with(content)

    result = generator.generate(prompt_with_source())

    assert result.answer.answer == "Deux jours."
    assert result.answer.citations == ["S1"]
    assert result.dropped_citations == ["S2"]


def test_rejects_an_answer_citing_only_unavailable_sources() -> None:
    content = json.dumps(
        {
            "status": "complete",
            "answer": "Réponse inventée [S2].",
            "citations": ["S2"],
        }
    )
    generator, _ = generator_with(content)

    with pytest.raises(GenerationError, match="unavailable source"):
        generator.generate(prompt_with_source())


@pytest.mark.parametrize(
    ("content", "choices", "message"),
    [(None, True, "empty content"), ("ignored", False, "no choices")],
)
def test_rejects_missing_server_output(
    content: str | None, choices: bool, message: str
) -> None:
    generator, _ = generator_with(content, choices=choices)

    with pytest.raises(GenerationError, match=message):
        generator.generate(prompt_with_source())


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"base_url": " "}, "base_url"),
        ({"model": " "}, "model"),
        ({"temperature": 3}, "temperature"),
        ({"max_tokens": 0}, "max_tokens"),
        ({"timeout_seconds": 0}, "timeout_seconds"),
    ],
)
def test_rejects_invalid_generator_configuration(overrides: dict, message: str) -> None:
    arguments = {
        "base_url": "http://localhost:8080/v1",
        "model": "qwen-test",
        "client": FakeClient(FakeCompletions("{}")),
        **overrides,
    }

    with pytest.raises(ValueError, match=message):
        OpenAICompatibleGenerator(**arguments)
