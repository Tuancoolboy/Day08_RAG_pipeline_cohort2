"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""


import hashlib
import json
import math
import re
import sys
from pathlib import Path

INDEX_PATH = Path(__file__).parent.parent / "data" / "index" / "vector_store.json"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_local_index() -> dict:
    """Load the local vector index created by Task 4."""
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Index file not found: {INDEX_PATH}. Run task4_chunking_indexing.py first."
        )
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Calculate cosine similarity for two vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def hash_embedding(text: str, dim: int) -> list[float]:
    """Same deterministic fallback embedding used in Task 4."""
    vector = [0.0] * dim
    tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign

    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def embed_query(query: str, index: dict) -> list[float]:
    """Embed query using the same embedding approach stored in the index."""
    chunks = index.get("chunks", [])
    first_metadata = chunks[0].get("metadata", {}) if chunks else {}
    stored_model = first_metadata.get("embedding_model", index.get("embedding_model", ""))
    dim = int(first_metadata.get("embedding_dim", index.get("embedding_dim", 1024)))

    if stored_model.startswith("hash-fallback-for-"):
        return hash_embedding(query, dim)

    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(index.get("embedding_model", stored_model))
        return model.encode(query).tolist()
    except Exception:
        return hash_embedding(query, dim)


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    # TODO: Implement semantic search
    #
    # Bước 1: Embed query bằng cùng model ở Task 4
    # Bước 2: Query vector store (cosine similarity)
    # Bước 3: Return top_k results
    #
    # Ví dụ với Weaviate:
    # import weaviate
    # from sentence_transformers import SentenceTransformer
    #
    # model = SentenceTransformer("BAAI/bge-m3")
    # query_embedding = model.encode(query).tolist()
    #
    # client = weaviate.connect_to_local()
    # collection = client.collections.get("DrugLawDocs")
    #
    # results = collection.query.near_vector(
    #     near_vector=query_embedding,
    #     limit=top_k,
    #     return_metadata=MetadataQuery(distance=True)
    # )
    #
    # return [
    #     {
    #         "content": obj.properties["content"],
    #         "score": 1 - obj.metadata.distance,  # distance → similarity
    #         "metadata": {"source": obj.properties["source"], ...}
    #     }
    #     for obj in results.objects
    # ]
    if not query.strip():
        return []

    index = load_local_index()
    query_embedding = embed_query(query, index)

    results = []
    for chunk in index.get("chunks", []):
        score = cosine_similarity(query_embedding, chunk.get("embedding", []))
        results.append({
            "content": chunk.get("content", ""),
            "score": float(score),
            "metadata": chunk.get("metadata", {}),
        })

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
