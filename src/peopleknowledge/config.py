"""Settings of the API, read from environment variables or from the project's .env file.

Every value has a default that works on the owner's machine with the local
embedded Qdrant, so that ``uvicorn`` can be started without any .env file.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Vector database. Empty QDRANT_URL = local embedded mode (the folder below).
    qdrant_path: Path = PROJECT_ROOT / "datasets" / "indexes" / "qdrant"
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_collection: str = "peopleknowledge"

    # Embedding model: must be the one used by scripts/index_chunks.py.
    embedding_model: str = "BAAI/bge-m3"
    # Hugging Face cache. Set it (e.g. D:\huggingface) when the system variable is not visible.
    hf_home: str | None = None
    # Never download a model from the API: a missing model is an error, not a 2 GB surprise.
    hf_hub_offline: bool = True

    # Answer generation through a local OpenAI-compatible llama.cpp server.
    llm_base_url: str = "http://localhost:8080/v1"
    llm_api_key: str = "local"
    llm_model: str = "qwen2.5-1.5b-instruct"
    llm_temperature: float = Field(default=0.1, ge=0, le=2)
    llm_max_tokens: int = Field(default=300, ge=1)
    llm_timeout_seconds: float = Field(default=120.0, gt=0)

    gold_path: Path = PROJECT_ROOT / "datasets" / "evaluation" / "rag_questions_gold.jsonl"
    company_profile_path: Path = PROJECT_ROOT / "datasets" / "metadata" / "company_profile.yaml"

    def apply_hugging_face_environment(self) -> None:
        """Must run BEFORE sentence-transformers is imported."""

        if self.hf_home:
            os.environ["HF_HOME"] = self.hf_home
        if self.hf_hub_offline:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")


@lru_cache
def get_settings() -> Settings:
    return Settings()
