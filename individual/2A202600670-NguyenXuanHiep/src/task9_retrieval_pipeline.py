"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả (RRF hoặc weighted fusion)
    3. Rerank
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results
"""

import asyncio
from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_THRESHOLD = 0.3
DEFAULT_TOP_K   = 5
RERANK_METHOD   = "rrf"


def retrieve(
    query          : str,
    top_k          : int   = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking  : bool  = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search  → dense_results
          ├→ Lexical Search   → sparse_results
          ├→ Merge (RRF)      → merged_results
          ├→ Rerank (RRF)     → reranked_results
          └→ score < threshold → PageIndex fallback
    """

    # ------------------------------------------------------------------
    # Step 1: Song song chạy semantic + lexical
    # ------------------------------------------------------------------
    print(f"  [1/4] Semantic search ...")
    try:
        dense_results = semantic_search(query, top_k=top_k * 2)
    except Exception as e:
        print(f"  ⚠ Semantic search lỗi: {e}")
        dense_results = []

    print(f"  [1/4] Lexical search ...")
    try:
        sparse_results = lexical_search(query, top_k=top_k * 2)
    except Exception as e:
        print(f"  ⚠ Lexical search lỗi: {e}")
        sparse_results = []

    print(f"  → Dense: {len(dense_results)} | Sparse: {len(sparse_results)}")

    # ------------------------------------------------------------------
    # Step 2: Merge bằng RRF
    # ------------------------------------------------------------------
    print(f"  [2/4] Merging (RRF) ...")

    # Lọc bỏ list rỗng trước khi đưa vào RRF
    ranked_lists = [r for r in [dense_results, sparse_results] if r]

    if not ranked_lists:
        print("  ⚠ Cả semantic và lexical đều rỗng → fallback PageIndex")
        return _pageindex_fallback(query, top_k)

    merged = rerank_rrf(ranked_lists, top_k=top_k * 2, k=60)

    # Đánh dấu nguồn hybrid
    for item in merged:
        item["source"] = "hybrid"

    print(f"  → Merged: {len(merged)} chunks")

    # ------------------------------------------------------------------
    # Step 3: Rerank
    # ------------------------------------------------------------------
    if use_reranking and merged:
        print(f"  [3/4] Reranking (method={RERANK_METHOD}) ...")
        try:
            final_results = rerank(
                query        = query,
                candidates   = merged,
                top_k        = top_k,
                method       = RERANK_METHOD,
                ranked_lists = [dense_results, sparse_results],  # cho RRF
            )
        except Exception as e:
            print(f"  ⚠ Rerank lỗi: {e} → dùng merged trực tiếp")
            final_results = merged[:top_k]
    else:
        print(f"  [3/4] Bỏ qua reranking.")
        final_results = merged[:top_k]

    print(f"  → After rerank: {len(final_results)} chunks")

    # ------------------------------------------------------------------
    # Step 4: Kiểm tra threshold → fallback PageIndex
    # ------------------------------------------------------------------
    best_score = final_results[0]["score"] if final_results else 0.0

    print(f"  [4/4] Best score: {best_score:.4f} | Threshold: {score_threshold}")

    if not final_results or best_score < score_threshold:
        print(
            f"  ⚠ Score {best_score:.4f} < threshold {score_threshold} "
            f"→ Fallback sang PageIndex"
        )
        return _pageindex_fallback(query, top_k)

    return final_results[:top_k]


def _pageindex_fallback(query: str, top_k: int) -> list[dict]:
    """Fallback sang PageIndex khi hybrid search không đủ tốt."""
    try:
        results = pageindex_search(query, top_k=top_k)
        print(f"  ✓ PageIndex fallback: {len(results)} kết quả")
        return results
    except Exception as e:
        print(f"  ✗ PageIndex fallback cũng lỗi: {e}")
        return []


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý năm 2024",
        "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện",
    ]

    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        print("-" * 60)
        try:
            results = retrieve(q, top_k=3, use_reranking=True)

            if not results:
                print("  Không có kết quả.")
                continue

            for i, r in enumerate(results, 1):
                source = r.get("source", "unknown")
                score  = r.get("score", 0)
                meta   = r.get("metadata", {})
                print(
                    f"  {i}. [{score:.4f}] [{source}] "
                    f"[{meta.get('type', '?')}] "
                    f"{r['content'][:100].strip()}..."
                )
        except Exception as e:
            print(f"  ✗ Lỗi: {e}")