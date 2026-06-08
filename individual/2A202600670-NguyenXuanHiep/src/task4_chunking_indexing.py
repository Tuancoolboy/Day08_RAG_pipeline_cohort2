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

from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

# =============================================================================
# CONFIGURATION
# =============================================================================

# Chunking: MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter
# - Vì file .md có cấu trúc heading rõ ràng (# ## ###)
# - Tách theo heading trước → giữ ngữ nghĩa từng section
# - Sau đó tách tiếp nếu section quá dài (chunk_size=500)
# - chunk_size=500: đủ ngữ nghĩa, không quá dài cho embedding
# - chunk_overlap=50: tránh mất context ở ranh giới chunk
CHUNK_SIZE     = 500
CHUNK_OVERLAP  = 50
CHUNKING_METHOD = "markdown_header"

# Embedding: BAAI/bge-m3
# - Multilingual, tốt nhất cho tiếng Việt hiện tại
# - 1024 dim, chạy được trên GTX 1650 (4GB VRAM)
# - Miễn phí, không cần API key
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM   = 1024

# Vector Store: ChromaDB
# - Local, không cần server/Docker
# - Đủ dùng cho dataset nhỏ (~100 chunks)
# - Dễ reset và debug
VECTOR_STORE     = "chromadb"
CHROMA_DB_PATH   = Path(__file__).parent.parent / "data" / "vectorstore"
COLLECTION_NAME  = "drug_law_docs"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            print(f"  ⚠ Bỏ qua file rỗng: {md_file.name}")
            continue

        # Xác định loại document dựa vào thư mục cha
        doc_type = "legal" if "legal" in str(md_file) else "news"

        documents.append({
            "content": content,
            "metadata": {
                "source"  : md_file.name,
                "filepath": str(md_file),
                "type"    : doc_type,
            },
        })
        print(f"  ✓ Loaded [{doc_type}]: {md_file.name}")

    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo 2 bước:
      1. MarkdownHeaderTextSplitter  → tách theo heading
      2. RecursiveCharacterTextSplitter → tách tiếp nếu section quá dài
    """
    from langchain_text_splitters import (
        MarkdownHeaderTextSplitter,
        RecursiveCharacterTextSplitter,
    )

    # Headers cần tách theo
    headers_to_split = [
        ("#",   "header_1"),
        ("##",  "header_2"),
        ("###", "header_3"),
    ]

    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split,
        strip_headers=False,  # giữ lại heading trong chunk
    )

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in documents:
        # Bước 1: tách theo markdown header
        md_chunks = md_splitter.split_text(doc["content"])

        for md_chunk in md_chunks:
            # md_chunk là LangChain Document object
            section_text = md_chunk.page_content.strip()
            if not section_text:
                continue

            # Bước 2: tách tiếp nếu section dài hơn CHUNK_SIZE
            if len(section_text) > CHUNK_SIZE:
                sub_splits = recursive_splitter.split_text(section_text)
            else:
                sub_splits = [section_text]

            for i, split_text in enumerate(sub_splits):
                if not split_text.strip():
                    continue

                # Gộp metadata: từ doc gốc + header metadata từ md_splitter
                chunk_metadata = {
                    **doc["metadata"],
                    **md_chunk.metadata,   # header_1, header_2, header_3
                    "chunk_index": i,
                    "chunk_total": len(sub_splits),
                }

                chunks.append({
                    "content" : split_text.strip(),
                    "metadata": chunk_metadata,
                })

    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng BAAI/bge-m3.
    Tự động dùng GPU (GTX 1650) nếu có, fallback CPU.
    """
    import torch
    from sentence_transformers import SentenceTransformer

    # Detect device
    if torch.cuda.is_available():
        device = "cuda"
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  🎮 GPU: {torch.cuda.get_device_name(0)} ({vram:.1f} GB VRAM)")
    else:
        device = "cpu"
        print("  💻 Dùng CPU (không tìm thấy GPU)")

    print(f"  📥 Loading model: {EMBEDDING_MODEL} ...")
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)

    texts = [c["content"] for c in chunks]

    # Batch size nhỏ cho GTX 1650 (4GB VRAM)
    batch_size = 16 if device == "cuda" else 8

    print(f"  🔄 Embedding {len(texts)} chunks (batch_size={batch_size}) ...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # chuẩn hoá L2 → cosine similarity tốt hơn
        convert_to_numpy=True,
    )

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()

    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào ChromaDB local.
    Tự động xoá collection cũ nếu đã tồn tại (idempotent).
    """
    import chromadb
    from chromadb.config import Settings

    CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)

    # Khởi tạo ChromaDB persistent client
    client = chromadb.PersistentClient(
        path=str(CHROMA_DB_PATH),
        settings=Settings(anonymized_telemetry=False),
    )

    # Xoá collection cũ nếu tồn tại (để chạy lại idempotent)
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"  🗑 Đã xoá collection cũ: {COLLECTION_NAME}")

    # Tạo collection mới
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine similarity
    )
    print(f"  ✓ Tạo collection: {COLLECTION_NAME}")

    # Insert theo batch (tránh memory spike)
    BATCH_SIZE = 50
    total = len(chunks)

    for start in range(0, total, BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]

        collection.add(
            ids         = [f"chunk_{start + j}" for j in range(len(batch))],
            embeddings  = [c["embedding"]        for c in batch],
            documents   = [c["content"]          for c in batch],
            metadatas   = [c["metadata"]         for c in batch],
        )
        print(f"  📦 Indexed {min(start + BATCH_SIZE, total)}/{total} chunks")

    # Kiểm tra lại
    count = collection.count()
    print(f"\n  ✅ ChromaDB sẵn sàng: {count} chunks tại {CHROMA_DB_PATH}")


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking : {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Store    : {VECTOR_STORE} → {CHROMA_DB_PATH}")
    print("=" * 50)

    # Bước 1: Load
    print("\n[1/4] Loading documents ...")
    docs = load_documents()
    print(f"  → {len(docs)} documents")

    if not docs:
        print("⚠ Không có file nào trong data/standardized/ — chạy task3 trước!")
        return

    # Bước 2: Chunk
    print("\n[2/4] Chunking ...")
    chunks = chunk_documents(docs)
    print(f"  → {len(chunks)} chunks")

    # Bước 3: Embed
    print("\n[3/4] Embedding ...")
    chunks = embed_chunks(chunks)
    print(f"  → {len(chunks)} chunks embedded")

    # Bước 4: Index
    print("\n[4/4] Indexing to ChromaDB ...")
    index_to_vectorstore(chunks)

    print("\n" + "=" * 50)
    print("✓ Pipeline hoàn thành!")
    print(f"✓ Vector store: {CHROMA_DB_PATH}")


if __name__ == "__main__":
    run_pipeline()