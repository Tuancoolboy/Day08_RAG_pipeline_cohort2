"""Persisted local vector database for the team RAG demo.

The project requirements mention Weaviate, but a Weaviate server/API key is not
always available during local demos. This module provides a deterministic local
vector index that can be built, stored, searched, and evaluated without network
access. It keeps the same retrieval output shape used by the UI.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

from .retrieval import KNOWLEDGE_BASE_PATH, load_documents, tokenize


TEAM_DIR = Path(__file__).resolve().parents[1]
VECTOR_STORE_DIR = TEAM_DIR / "vector_store"
VECTOR_INDEX_PATH = VECTOR_STORE_DIR / "index.json"
EMBEDDING_DIM = 384
EMBEDDING_MODEL_NAME = "local-hashed-tfidf-384"
SCHEMA_VERSION = 1


def _hash_token(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def embed_text(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    """Embed text into a deterministic hashed bag-of-words vector."""
    vector = [0.0] * dim
    tokens = tokenize(text)
    if not tokens:
        return vector

    for token in tokens:
        hashed = _hash_token(token)
        index = hashed % dim
        sign = 1.0 if (hashed >> 8) % 2 == 0 else -1.0
        vector[index] += sign

    return _normalize(vector)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Cosine similarity for already normalized vectors."""
    if not left or not right:
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def _document_text(document: dict[str, Any]) -> str:
    return f"{document['title']} {document['source']} {document['content']}"


def build_vector_index(
    knowledge_base_path: Path = KNOWLEDGE_BASE_PATH,
    output_path: Path = VECTOR_INDEX_PATH,
) -> dict[str, Any]:
    """Build and persist the vector index from the team knowledge base."""
    documents = load_documents(knowledge_base_path)
    records = []

    for document in documents:
        text = _document_text(document)
        records.append(
            {
                "id": document["id"],
                "content": document["content"],
                "metadata": document["metadata"],
                "embedding": embed_text(text),
            }
        )

    index = {
        "schema_version": SCHEMA_VERSION,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_dim": EMBEDDING_DIM,
        "source_path": str(knowledge_base_path),
        "built_at_unix": int(time.time()),
        "record_count": len(records),
        "records": records,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return index


def load_vector_index(path: Path = VECTOR_INDEX_PATH) -> dict[str, Any]:
    """Load the persisted vector index, building it when missing."""
    if not path.exists():
        return build_vector_index(output_path=path)

    with path.open("r", encoding="utf-8") as f:
        index = json.load(f)

    if index.get("schema_version") != SCHEMA_VERSION:
        return build_vector_index(output_path=path)
    if index.get("embedding_dim") != EMBEDDING_DIM:
        return build_vector_index(output_path=path)

    return index


def vector_search(
    query: str,
    top_k: int = 5,
    index_path: Path = VECTOR_INDEX_PATH,
) -> list[dict[str, Any]]:
    """Search the local vector index and return UI-compatible sources."""
    if not query.strip():
        return []

    index = load_vector_index(index_path)
    query_embedding = embed_text(query)

    results = []
    for record in index["records"]:
        score = cosine_similarity(query_embedding, record["embedding"])
        results.append(
            {
                "content": record["content"],
                "score": round(max(score, 0.0), 4),
                "metadata": record["metadata"],
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def vector_store_status(path: Path = VECTOR_INDEX_PATH) -> dict[str, Any]:
    """Return index status for scripts, tests, and UI diagnostics."""
    if not path.exists():
        return {
            "exists": False,
            "path": str(path),
            "record_count": 0,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "embedding_dim": EMBEDDING_DIM,
        }

    index = load_vector_index(path)
    return {
        "exists": True,
        "path": str(path),
        "record_count": int(index.get("record_count", 0)),
        "embedding_model": index.get("embedding_model", EMBEDDING_MODEL_NAME),
        "embedding_dim": int(index.get("embedding_dim", EMBEDDING_DIM)),
        "built_at_unix": index.get("built_at_unix"),
    }

