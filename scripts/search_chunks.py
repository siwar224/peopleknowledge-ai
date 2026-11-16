"""Ask a question and print the closest chunks found in Qdrant (manual check).

Archived documents are left out by default, as the assistant will do;
``--archived`` includes them (for "what was the rule before?" questions).

The local embedded Qdrant folder can be opened by ONE program at a time:
close any other script using it first.

Usage:
    python scripts/search_chunks.py "Puis-je poser des congés pendant ma période d'essai ?"
    python scripts/search_chunks.py "Combien de jours de télétravail avant 2026 ?" --archived
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.rag.embeddings import SentenceTransformerEmbedder
from app.rag.indexing.qdrant_index import search
from qdrant_client import QdrantClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QDRANT_PATH = PROJECT_ROOT / "datasets" / "indexes" / "qdrant"


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--archived", action="store_true", help="include archived documents"
    )
    parser.add_argument("--qdrant-path", type=Path, default=DEFAULT_QDRANT_PATH)
    parser.add_argument("--collection", default="peopleknowledge")
    parser.add_argument("--model", default="BAAI/bge-m3")
    arguments = parser.parse_args(argv)

    if not arguments.qdrant_path.exists():
        print("ERROR: no index yet (run scripts/index_chunks.py first).")
        return 1
    client = QdrantClient(path=str(arguments.qdrant_path))
    try:
        embedder = SentenceTransformerEmbedder(arguments.model)
        results = search(
            client,
            arguments.collection,
            embedder,
            arguments.question,
            limit=arguments.limit,
            include_archived=arguments.archived,
        )
    finally:
        client.close()

    print(f"Question: {arguments.question}\n")
    for rank, point in enumerate(results, 1):
        chunk = point.payload or {}
        document = chunk.get("document", {})
        pages = ", ".join(map(str, chunk.get("pages") or [])) or "-"
        print(
            f"{rank}. score {point.score:.3f}  {chunk.get('chunk_id')}  "
            f"[{document.get('lifecycle_status')}]  p. {pages}"
        )
        print(f"   {chunk.get('prefix')}")
        print(f"   {chunk.get('content', '')[:220].replace(chr(10), ' / ')}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
