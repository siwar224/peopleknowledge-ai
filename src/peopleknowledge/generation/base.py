"""Provider-independent contract for language-model answer generation."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .models import BuiltPrompt, GenerationResult


class GenerationError(Exception):
    """The model call failed or returned an answer that cannot be trusted."""


class Generator(ABC):
    """Generate one validated answer from a prompt, regardless of the provider."""

    @property
    @abstractmethod
    def model(self) -> str:
        """Model identifier sent to the inference server."""

    @abstractmethod
    def generate(self, prompt: BuiltPrompt) -> GenerationResult:
        """Call the model and validate its structured answer."""
