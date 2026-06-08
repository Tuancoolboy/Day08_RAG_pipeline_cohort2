"""Build the local vector database for the team RAG chatbot."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from team.src.vector_store import VECTOR_INDEX_PATH, build_vector_index  # noqa: E402


def main() -> None:
    index = build_vector_index()
    print(f"Built vector index: {VECTOR_INDEX_PATH}")
    print(f"Records: {index['record_count']}")
    print(f"Embedding model: {index['embedding_model']}")
    print(f"Embedding dim: {index['embedding_dim']}")


if __name__ == "__main__":
    main()

