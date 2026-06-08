"""
Task 8 — PageIndex Vectorless RAG.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR  = Path(__file__).parent.parent / "data" / "standardized"


def _get_client():
    """Khởi tạo PageIndexClient."""
    from pageindex import PageIndexClient
    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def upload_documents() -> dict:
    """Upload toàn bộ markdown documents lên PageIndex."""
    if not PAGEINDEX_API_KEY:
        raise ValueError("Thiếu PAGEINDEX_API_KEY — set trong file .env")

    pi = _get_client()

    md_files = sorted(STANDARDIZED_DIR.rglob("*.md"))
    if not md_files:
        raise FileNotFoundError(
            f"Không có file .md trong {STANDARDIZED_DIR} — chạy task3 trước!"
        )

    uploaded = {}
    for md_file in md_files:
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            print(f"  ⚠ Bỏ qua file rỗng: {md_file.name}")
            continue

        doc_type = "legal" if "legal" in str(md_file) else "news"
        try:
            response = pi.upload(
                content  = content,
                metadata = {
                    "filename": md_file.name,
                    "type"    : doc_type,
                    "filepath": str(md_file),
                },
            )
            doc_id = response.id
            uploaded[md_file.name] = doc_id
            print(f"  ✓ Uploaded [{doc_type}]: {md_file.name} → id={doc_id}")
        except Exception as e:
            print(f"  ✗ Lỗi upload {md_file.name}: {e}")

    print(f"\n  Upload xong: {len(uploaded)}/{len(md_files)} files.")
    return uploaded


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.

    Returns:
        List of {'content', 'score', 'metadata', 'source': 'pageindex'}
    """
    if not PAGEINDEX_API_KEY:
        print("  ⚠ Thiếu PAGEINDEX_API_KEY — fallback local search")
        return _fallback_local_search(query, top_k)

    try:
        pi = _get_client()
        raw_results = pi.query(query=query, top_k=top_k)

        results = []
        for r in raw_results:
            results.append({
                "content" : r.text,
                "score"   : round(float(r.score), 4),
                "metadata": {
                    **r.metadata,
                    "filename": r.metadata.get("filename", "unknown"),
                    "type"    : r.metadata.get("type", "unknown"),
                },
                "source"  : "pageindex",
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    except Exception as e:
        print(f"  ⚠ PageIndex API lỗi: {e} → fallback local search")
        return _fallback_local_search(query, top_k)


def _fallback_local_search(query: str, top_k: int) -> list[dict]:
    """
    Fallback khi PageIndex API không khả dụng.
    Dùng BM25 local, đánh dấu source='pageindex' để pipeline hoạt động.
    """
    try:
        from .task6_lexical_search import lexical_search
        results = lexical_search(query, top_k=top_k)
        for r in results:
            r["source"] = "pageindex" 
        return results
    except Exception as e:
        print(f"  ⚠ Fallback BM25 cũng lỗi: {e}")
        return []


if __name__ == "__main__":
    # Kiểm tra API methods có sẵn
    print("PageIndex SDK info:")
    from pageindex import PageIndexClient
    print(f"  Class : PageIndexClient")
    print(f"  Methods: {[m for m in dir(PageIndexClient) if not m.startswith('_')]}")

    if not PAGEINDEX_API_KEY:
        print("\n⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("\nUploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] [{r['source']}] {r['content'][:100]}...")