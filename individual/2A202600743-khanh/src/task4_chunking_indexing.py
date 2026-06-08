"""Task 4 - Chunk Markdown documents and build a local JSONL index."""

import json
import re
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
STANDARDIZED_DIR = PROJECT_DIR / "data" / "standardized"
INDEX_DIR = PROJECT_DIR / "data" / "index"
CHUNKS_PATH = INDEX_DIR / "chunks.jsonl"

# Recursive character chunking keeps legal/news text simple and robust locally.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
CHUNKING_METHOD = "recursive"
EMBEDDING_MODEL = "local-tfidf-hash"
EMBEDDING_DIM = 512
VECTOR_STORE = "local-jsonl"


def load_documents() -> list[dict]:
    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8", errors="replace").strip()
        if not content:
            continue
        doc_type = md_file.parent.name
        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "path": str(md_file),
                    "type": doc_type,
                },
            }
        )
    return documents


def _split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""
            step = max(1, chunk_size - overlap)
            for start in range(0, len(paragraph), step):
                piece = paragraph[start : start + chunk_size].strip()
                if piece:
                    chunks.append(piece)
            continue
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            chunks.append(current.strip())
            prefix = current[-overlap:].strip() if overlap and current else ""
            current = f"{prefix}\n\n{paragraph}".strip() if prefix else paragraph
    if current:
        chunks.append(current.strip())
    return chunks


def chunk_documents(documents: list[dict]) -> list[dict]:
    chunks = []
    for doc in documents:
        for index, chunk_text in enumerate(_split_text(doc["content"])):
            chunks.append(
                {
                    "content": chunk_text,
                    "metadata": {**doc["metadata"], "chunk_index": index},
                }
            )
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    # The search modules compute sparse/vector scores on demand from text.
    for chunk in chunks:
        chunk["embedding_model"] = EMBEDDING_MODEL
    return chunks


def index_to_vectorstore(chunks: list[dict]) -> Path:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with CHUNKS_PATH.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    return CHUNKS_PATH


def load_index() -> list[dict]:
    if not CHUNKS_PATH.exists():
        run_pipeline()
    if not CHUNKS_PATH.exists():
        return []
    return [json.loads(line) for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_pipeline() -> list[dict]:
    docs = load_documents()
    chunks = embed_chunks(chunk_documents(docs))
    index_to_vectorstore(chunks)
    print(f"Indexed {len(chunks)} chunks to {CHUNKS_PATH}")
    return chunks


if __name__ == "__main__":
    run_pipeline()
