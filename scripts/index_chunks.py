"""Embed the chunks with bge-m3 and store them in Qdrant.

Input: datasets/processed/chunks/*.jsonl (output of chunk_documents.py).
Output: the Qdrant collection "peopleknowledge", by default in LOCAL EMBEDDED
mode: a folder, datasets/indexes/qdrant/ (no server, nothing to install,
ignored by git and fully regenerable). ``--qdrant-url`` uses a Qdrant server
instead (local Docker or Qdrant Cloud): the rest of the code is the same.

Like the other runners, only what changed is redone: a chunk already in the
collection with the same text and the same model is not encoded again (the
slow part: several minutes for the whole corpus on a laptop CPU); points of
chunks that no longer exist are deleted.

The model is read from the Hugging Face cache (HF_HOME). It must be the same
model at indexing and search time: vectors of two models are not comparable.

Usage:
    python scripts/index_chunks.py
    python scripts/index_chunks.py --recreate          # rebuild everything (e.g. new model)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from pydantic import ValidationError
from qdrant_client import QdrantClient

from peopleknowledge.ingestion.chunking.models import Chunk
from peopleknowledge.retrieval.embeddings import Embedder, SentenceTransformerEmbedder
from peopleknowledge.retrieval.qdrant_index import IndexingError, ensure_collection, sync_chunks

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS_DIR = PROJECT_ROOT / "datasets" / "processed" / "chunks"
DEFAULT_QDRANT_PATH = PROJECT_ROOT / "datasets" / "indexes" / "qdrant"
DEFAULT_COLLECTION = "peopleknowledge"
DEFAULT_MODEL = "BAAI/bge-m3"


def load_chunks(chunks_dir: Path) -> list[Chunk]:
    paths = sorted(chunks_dir.glob("*.jsonl"))
    if not paths:
        raise ValueError(
            f"no chunk file in {chunks_dir} (run scripts/chunk_documents.py first)."
        )
    chunks: list[Chunk] = []
    for path in paths:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                chunks.append(Chunk.model_validate(json.loads(line)))
            except ValueError as error:  # JSON or validation error
                raise ValueError(f"{path.name} line {number}: {error}") from error
    return chunks


def open_client(
    qdrant_path: Path, qdrant_url: str | None, api_key: str | None
) -> QdrantClient:
    if qdrant_url:
        return QdrantClient(url=qdrant_url, api_key=api_key)
    qdrant_path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(qdrant_path))


def run(
    client: QdrantClient,
    collection: str,
    chunks_dir: Path,
    embedder: Embedder,
    recreate: bool = False,
) -> int:
    chunks = load_chunks(chunks_dir)
    ensure_collection(client, collection, embedder.dimension, recreate=recreate)
    started = time.perf_counter()
    report = sync_chunks(client, collection, chunks, embedder)
    elapsed = time.perf_counter() - started

    for chunk_id in report.deleted:
        print(f"  deleted   {chunk_id}  (no longer in the chunk files)")
    total = client.count(collection).count
    print(
        f"Indexing: new={len(report.new)}, updated={len(report.updated)}, "
        f"unchanged={len(report.unchanged)}, deleted={len(report.deleted)}; "
        f"{report.encoded} chunk(s) encoded in {elapsed:.1f} s."
    )
    print(f"Collection {collection!r}: {total} points, model {embedder.name}.")
    return 0


def main(argv: list[str] | None = None, embedder: Embedder | None = None) -> int:
    """``embedder`` replaces the real model (used by the tests)."""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS_DIR)
    parser.add_argument("--qdrant-path", type=Path, default=DEFAULT_QDRANT_PATH)
    parser.add_argument(
        "--qdrant-url", help="use a Qdrant server instead of the local folder"
    )
    parser.add_argument("--api-key", help="Qdrant Cloud API key (never commit it)")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--recreate", action="store_true", help="delete and rebuild the collection"
    )
    arguments = parser.parse_args(argv)

    if embedder is None:
        embedder = SentenceTransformerEmbedder(
            arguments.model, batch_size=arguments.batch_size
        )
    client = open_client(
        arguments.qdrant_path.resolve(), arguments.qdrant_url, arguments.api_key
    )
    try:
        return run(
            client,
            arguments.collection,
            arguments.chunks_dir.resolve(),
            embedder,
            arguments.recreate,
        )
    except (ValueError, ValidationError, IndexingError) as error:
        print(f"ERROR: {error}")
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
