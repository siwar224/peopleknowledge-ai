"""Generate answers through an OpenAI-compatible server such as llama-server."""

from __future__ import annotations

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
            answer = GeneratedAnswer.model_validate_json(content)
        except ValidationError as error:
            raise GenerationError(
                f"generation server returned invalid answer JSON: {error}"
            ) from error

        unknown = sorted(set(answer.citations) - set(prompt.allowed_source_ids))
        if unknown:
            raise GenerationError(f"answer cites unavailable source label(s): {unknown}.")

        return GenerationResult(
            answer=answer,
            model=self.model,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )
