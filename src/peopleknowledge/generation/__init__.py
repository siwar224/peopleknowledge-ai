"""Build grounded contexts and generate answers from retrieved chunks."""

from .base import GenerationError, Generator
from .context_builder import build_context
from .models import (
    BuiltContext,
    BuiltPrompt,
    ContextSource,
    GeneratedAnswer,
    GenerationResult,
    PromptMessage,
    RetrievedChunk,
)
from .openai_compatible import OpenAICompatibleGenerator
from .prompt_builder import build_prompt

__all__ = [
    "BuiltContext",
    "BuiltPrompt",
    "ContextSource",
    "GeneratedAnswer",
    "GenerationError",
    "GenerationResult",
    "Generator",
    "OpenAICompatibleGenerator",
    "PromptMessage",
    "RetrievedChunk",
    "build_context",
    "build_prompt",
]
