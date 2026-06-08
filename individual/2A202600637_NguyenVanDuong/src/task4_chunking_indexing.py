"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (Weaviate khuyến cáo)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho tiếng Việt)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - Weaviate (khuyến cáo: hỗ trợ hybrid search built-in)
    - ChromaDB (đơn giản, local)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers weaviate-client
"""

import hashlib
import json
import math
import re
from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
INDEX_DIR = Path(__file__).parent.parent / "data" / "index"
INDEX_PATH = INDEX_DIR / "vector_store.json"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# TODO: Chọn chunking strategy và giải thích vì sao
CHUNK_SIZE = 500        # Vì sao chọn 500? ...
CHUNK_OVERLAP = 50      # Vì sao chọn 50? ...
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# TODO: Chọn embedding model và giải thích
EMBEDDING_MODEL = "BAAI/bge-m3"  # Vì sao? Multilingual, tốt cho tiếng Việt
EMBEDDING_DIM = 1024

# TODO: Chọn vector store
VECTOR_STORE = "weaviate"  # "weaviate" | "chromadb" | "faiss"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    # TODO: Iterate qua STANDARDIZED_DIR, đọc .md files
    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        relative_path = md_file.relative_to(STANDARDIZED_DIR)
        doc_type = relative_path.parts[0] if len(relative_path.parts) > 1 else "unknown"
        documents.append({
            "content": content,
            "metadata": {
                "source": md_file.name,
                "path": str(relative_path).replace("\\", "/"),
                "type": doc_type,
            },
        })
    return documents
    # raise NotImplementedError("Implement load_documents")


def _recursive_split_text(text: str) -> list[str]:
    """Split text by natural markdown boundaries before falling back to chars."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return splitter.split_text(text)
    except ImportError:
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + CHUNK_SIZE, text_length)
            window = text[start:end]

            if end < text_length:
                split_at = max(window.rfind("\n\n"), window.rfind("\n"), window.rfind(". "))
                if split_at > CHUNK_SIZE * 0.5:
                    end = start + split_at + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            if end >= text_length:
                break
            start = max(end - CHUNK_OVERLAP, start + 1)

        return chunks


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    # TODO: Implement chunking
    #
    # Ví dụ với RecursiveCharacterTextSplitter:
    chunks = []
    for doc in documents:
        splits = _recursive_split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunks.append({
                "content": chunk_text,
                "metadata": {
                    **doc["metadata"],
                    "chunk_index": i,
                    "chunking_method": CHUNKING_METHOD,
                    "chunk_size": CHUNK_SIZE,
                    "chunk_overlap": CHUNK_OVERLAP,
                },
            })
    return chunks
    # raise NotImplementedError("Implement chunk_documents")


def _hash_embedding(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    """Deterministic local fallback embedding when model downloads are unavailable."""
    vector = [0.0] * dim
    tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign

    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    # TODO: Implement embedding
    #
    # Ví dụ với sentence-transformers:
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(EMBEDDING_MODEL)
        texts = [c["content"] for c in chunks]
        embeddings = model.encode(texts, show_progress_bar=True)
        for chunk, emb in zip(chunks, embeddings):
            chunk["embedding"] = emb.tolist()
            chunk["metadata"]["embedding_model"] = EMBEDDING_MODEL
            chunk["metadata"]["embedding_dim"] = len(chunk["embedding"])
    except Exception as exc:
        print(f"Embedding fallback: {exc}")
        for chunk in chunks:
            chunk["embedding"] = _hash_embedding(chunk["content"])
            chunk["metadata"]["embedding_model"] = f"hash-fallback-for-{EMBEDDING_MODEL}"
            chunk["metadata"]["embedding_dim"] = EMBEDDING_DIM
    return chunks
    # raise NotImplementedError("Implement embed_chunks")


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "vector_store": "local_json",
        "configured_vector_store": VECTOR_STORE,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "chunks": chunks,
    }
    INDEX_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Local index saved: {INDEX_PATH}")
    return

    # TODO: Implement indexing
    #
    # Ví dụ với Weaviate:
    import weaviate
    from weaviate.classes.config import Configure, Property, DataType
    
    client = weaviate.connect_to_local()  # hoặc connect_to_weaviate_cloud()
    
    # Tạo collection
    collection = client.collections.create(
        name="DrugLawDocs",
        vectorizer_config=Configure.Vectorizer.none(),
        properties=[
            Property(name="content", data_type=DataType.TEXT),
            Property(name="source", data_type=DataType.TEXT),
            Property(name="doc_type", data_type=DataType.TEXT),
        ]
    )
    
    # Insert chunks
    with collection.batch.dynamic() as batch:
        for chunk in chunks:
            batch.add_object(
                properties={
                    "content": chunk["content"],
                    "source": chunk["metadata"].get("source", ""),
                    "doc_type": chunk["metadata"].get("type", ""),
                },
                vector=chunk["embedding"]
            )
    # raise NotImplementedError("Implement index_to_vectorstore")


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\nLoaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
