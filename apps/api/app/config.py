"""Settings of the API, read from environment variables or from the project's .env file.

Every value has a default that works on the owner's machine with the local
embedded Qdrant, so that ``uvicorn`` can be started without any .env file.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


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

    gold_path: Path = PROJECT_ROOT / "datasets" / "evaluation" / "rag_questions_gold.jsonl"

    def apply_hugging_face_environment(self) -> None:
        """Must run BEFORE sentence-transformers is imported."""

        if self.hf_home:
            os.environ["HF_HOME"] = self.hf_home
        if self.hf_hub_offline:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")


@lru_cache
def get_settings() -> Settings:
    return Settings()
