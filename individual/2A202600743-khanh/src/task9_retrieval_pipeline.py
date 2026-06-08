"""Task 9 - Hybrid retrieval pipeline with fallback logic."""

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search

SCORE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    dense_results = semantic_search(query, top_k=top_k * 2)
    sparse_results = lexical_search(query, top_k=top_k * 2)
    merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 3)
    for item in merged:
        item["source"] = "hybrid"

    final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD) if use_reranking else merged[:top_k]
    for item in final_results:
        item["source"] = "hybrid"

    if not final_results or final_results[0].get("score", 0.0) < score_threshold:
        fallback = pageindex_search(query, top_k=top_k)
        if fallback:
            return fallback[:top_k]
    return final_results[:top_k]


if __name__ == "__main__":
    for result in retrieve("hinh phat ma tuy", top_k=3):
        print(f"[{result['score']:.3f}] [{result['source']}] {result['content'][:100]}...")
