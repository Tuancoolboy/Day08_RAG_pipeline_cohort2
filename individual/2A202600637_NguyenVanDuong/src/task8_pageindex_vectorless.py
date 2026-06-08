"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API
"""

import os
import json
import re
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
INDEX_PATH = Path(__file__).parent.parent / "data" / "index" / "vector_store.json"
PAGEINDEX_LOCAL_MANIFEST = Path(__file__).parent.parent / "data" / "index" / "pageindex_manifest.json"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def tokenize(text: str) -> list[str]:
    """Tokenize text for local vectorless fallback search."""
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def load_chunks() -> list[dict]:
    """Load chunks from Task 4's local index."""
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Index file not found: {INDEX_PATH}. Run task4_chunking_indexing.py first."
        )

    index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    return [
        {
            "content": chunk.get("content", ""),
            "metadata": chunk.get("metadata", {}),
        }
        for chunk in index.get("chunks", [])
        if chunk.get("content", "").strip()
    ]


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    # TODO: Implement upload
    #
    # Tham khảo: https://github.com/VectifyAI/PageIndex
    #
    # from pageindex import PageIndex
    #
    # pi = PageIndex(api_key=PAGEINDEX_API_KEY)
    #
    # for md_file in STANDARDIZED_DIR.rglob("*.md"):
    #     content = md_file.read_text(encoding="utf-8")
    #     pi.upload(
    #         content=content,
    #         metadata={"filename": md_file.name, "type": md_file.parent.name}
    #     )
    #     print(f"  ✓ Uploaded: {md_file.name}")
    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        relative_path = md_file.relative_to(STANDARDIZED_DIR)
        documents.append({
            "content": content,
            "metadata": {
                "filename": md_file.name,
                "path": str(relative_path).replace("\\", "/"),
                "type": md_file.parent.name,
            },
        })

    PAGEINDEX_LOCAL_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    PAGEINDEX_LOCAL_MANIFEST.write_text(
        json.dumps(
            {
                "mode": "local_vectorless_fallback",
                "pageindex_api_key_present": bool(PAGEINDEX_API_KEY),
                "document_count": len(documents),
                "documents": [
                    {
                        "metadata": doc["metadata"],
                        "char_count": len(doc["content"]),
                    }
                    for doc in documents
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Prepared {len(documents)} documents for PageIndex fallback.")
    print(f"  Local manifest saved: {PAGEINDEX_LOCAL_MANIFEST}")
    return documents


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    # TODO: Implement PageIndex query
    #
    # from pageindex import PageIndex
    #
    # pi = PageIndex(api_key=PAGEINDEX_API_KEY)
    # results = pi.query(query=query, top_k=top_k)
    #
    # return [
    #     {
    #         "content": r.text,
    #         "score": r.score,
    #         "metadata": r.metadata,
    #         "source": "pageindex"
    #     }
    #     for r in results
    # ]
    if not query.strip():
        return []

    query_terms = tokenize(query)
    if not query_terms:
        return []

    chunks = load_chunks()
    query_term_set = set(query_terms)
    scored = []

    for chunk in chunks:
        content = chunk["content"]
        doc_terms = tokenize(content)
        if not doc_terms:
            continue

        doc_term_set = set(doc_terms)
        overlap = query_term_set & doc_term_set
        if not overlap:
            continue

        coverage = len(overlap) / len(query_term_set)
        density = sum(doc_terms.count(term) for term in overlap) / max(len(doc_terms), 1)
        score = coverage + density

        scored.append({
            "content": content,
            "score": float(score),
            "metadata": {**chunk.get("metadata", {}), "pageindex_mode": "local_fallback"},
            "source": "pageindex",
        })

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("PAGEINDEX_API_KEY is not set; using local vectorless fallback.")
        print("Register at https://pageindex.ai/ if you want hosted PageIndex.")

    print("Preparing documents...")
    upload_documents()

    print("\nTest query:")
    results = pageindex_search("hinh phat su dung ma tuy", top_k=3)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
    sys.exit(0)

    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
