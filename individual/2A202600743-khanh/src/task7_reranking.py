"""Task 7 - Reranking with local RRF and lexical overlap scoring."""

from .task5_semantic_search import tokenize


def rerank_cross_encoder(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """Approximate cross-encoder rerank with query/content token overlap."""
    query_terms = set(tokenize(query))
    reranked = []
    for rank, candidate in enumerate(candidates, 1):
        doc_terms = set(tokenize(candidate.get("content", "")))
        overlap = len(query_terms & doc_terms) / len(query_terms) if query_terms else 0.0
        base_score = float(candidate.get("score", 0.0))
        score = 0.7 * overlap + 0.3 * base_score + 1.0 / (1000 + rank)
        item = candidate.copy()
        item["score"] = float(score)
        reranked.append(item)
    reranked.sort(key=lambda item: item["score"], reverse=True)
    return reranked[:top_k]


def rerank_mmr(query_embedding: list[float], candidates: list[dict], top_k: int = 5, lambda_param: float = 0.7) -> list[dict]:
    return sorted(candidates, key=lambda item: item.get("score", 0), reverse=True)[:top_k]


def rerank_rrf(ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60) -> list[dict]:
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}
    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item.get("metadata", {}).get("path", "") + "|" + item.get("content", "")
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            items[key] = item
    results = []
    for key, score in sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:top_k]:
        item = items[key].copy()
        item["score"] = float(score)
        results.append(item)
    return results


def rerank(query: str, candidates: list[dict], top_k: int = 5, method: str = "cross_encoder") -> list[dict]:
    if method == "rrf":
        return rerank_rrf([candidates], top_k=top_k)
    if method == "mmr":
        return rerank_mmr([], candidates, top_k=top_k)
    return rerank_cross_encoder(query, candidates, top_k=top_k)


if __name__ == "__main__":
    sample = [{"content": "Toi tang tru trai phep chat ma tuy", "score": 0.5, "metadata": {}}]
    print(rerank("hinh phat ma tuy", sample))
