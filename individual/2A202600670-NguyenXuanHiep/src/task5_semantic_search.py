"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.
    """
    import torch
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
    from pathlib import Path

    EMBEDDING_MODEL = "BAAI/bge-m3"
    CHROMA_DB_PATH  = Path(__file__).parent.parent / "data" / "vectorstore"
    COLLECTION_NAME = "drug_law_docs"

    # Bước 1: Embed query bằng cùng model Task 4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = SentenceTransformer(EMBEDDING_MODEL, device=device)

    query_embedding = model.encode(
        query,
        normalize_embeddings=True,  # giống Task 4
        convert_to_numpy=True,
    ).tolist()

    # Bước 2: Query ChromaDB
    client = chromadb.PersistentClient(
        path=str(CHROMA_DB_PATH),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection(COLLECTION_NAME)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # Bước 3: Format output — distance → similarity score
    # ChromaDB cosine distance: score = 1 - distance
    output = []
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, dist in zip(documents, metadatas, distances):
        output.append({
            "content" : doc,
            "score"   : round(1 - dist, 4),  # cosine similarity
            "metadata": meta,
        })

    # Sắp xếp descending theo score
    output.sort(key=lambda x: x["score"], reverse=True)

    return output


if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
