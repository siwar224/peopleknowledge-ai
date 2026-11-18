"""Generate answers through an OpenAI-compatible server such as llama-server.

Citation handling is TOLERANT but safe: a small model sometimes cites a label
that was not in the prompt ("S7" with 5 sources) or writes "[S1]" in the text.
Unknown labels are dropped (and reported in ``dropped_citations``) instead of
turning a good answer into an HTTP 502; inline labels are removed from the text
and kept as citations. The safety rule stays: a complete or partial answer
that cites NO available source is still refused.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from openai import OpenAI, OpenAIError
from pydantic import ValidationError

from .base import GenerationError, Generator
from .models import BuiltPrompt, GeneratedAnswer, GenerationResult


class OpenAICompatibleGenerator(Generator):
    """Synchronous generator for the OpenAI-compatible Chat Completions API."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "local",
        temperature: float = 0.1,
        max_tokens: int = 300,
        timeout_seconds: float = 120.0,
        client: Any | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("base_url must not be blank.")
        if not model.strip():
            raise ValueError("model must not be blank.")
        if not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2.")
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1.")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")

        self.base_url = base_url.rstrip("/")
        self._model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = client or OpenAI(
            base_url=self.base_url,
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=0,
        )

    @property
    def model(self) -> str:
        return self._model

    def generate(self, prompt: BuiltPrompt) -> GenerationResult:
        started = time.perf_counter()
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[message.model_dump() for message in prompt.messages],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
        except OpenAIError as error:
            raise GenerationError(f"generation server request failed: {error}") from error

        if not completion.choices:
            raise GenerationError("generation server returned no choices.")
        content = completion.choices[0].message.content
        if not content or not content.strip():
            raise GenerationError("generation server returned empty content.")

        try:
            data = json.loads(content)
        except json.JSONDecodeError as error:
            raise GenerationError(
                f"generation server returned invalid answer JSON: {error}"
            ) from error
        if not isinstance(data, dict):
            raise GenerationError("generation server returned invalid answer JSON: not an object.")

        data, dropped = _clean_citations(data, prompt.allowed_source_ids)
        if dropped and not data.get("citations") and data.get("status") != "not_found":
            raise GenerationError(f"answer cites only unavailable source label(s): {dropped}.")
        try:
            answer = GeneratedAnswer.model_validate(data)
        except ValidationError as error:
            raise GenerationError(
                f"generation server returned invalid answer JSON: {error}"
            ) from error

        return GenerationResult(
            answer=answer,
            model=self.model,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            dropped_citations=dropped,
        )


_INLINE_LABEL = re.compile(r"\s*\[(S[1-9]\d*)\]")


def _clean_citations(data: dict, allowed: list[str]) -> tuple[dict, list[str]]:
    """Move inline [S1] labels to the citations list, drop labels not in the prompt."""

    answer = data.get("answer")
    citations = data.get("citations")
    if not isinstance(answer, str) or not isinstance(citations, list):
        return data, []  # left to the model validation, which explains the problem
    inline = _INLINE_LABEL.findall(answer)
    labels = [c for c in [*citations, *inline] if isinstance(c, str)]
    kept = [c for c in dict.fromkeys(labels) if c in allowed]
    dropped = sorted({c for c in labels if c not in allowed})
    cleaned = {**data, "answer": _INLINE_LABEL.sub("", answer).strip(), "citations": kept}
    if any(not isinstance(c, str) for c in citations):
        cleaned["citations"] = citations  # malformed: let the model validation report it
    return cleaned, dropped
