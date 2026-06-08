"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.
"""

import hashlib
import math
import re
import sys
from typing import Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def tokenize(text: str) -> set[str]:
    """Tokenize Vietnamese/ASCII text into lowercase terms."""
    return set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def lexical_overlap_score(query: str, document: str) -> float:
    """Small local relevance proxy for cross-encoder fallback."""
    query_terms = tokenize(query)
    doc_terms = tokenize(document)
    if not query_terms or not doc_terms:
        return 0.0
    return len(query_terms & doc_terms) / len(query_terms)


def content_key(item: dict) -> str:
    """Stable key for de-duplicating chunks across rankers."""
    metadata = item.get("metadata", {})
    source = metadata.get("source", "")
    chunk_index = metadata.get("chunk_index", "")
    if source or chunk_index != "":
        return f"{source}:{chunk_index}"
    return hashlib.sha256(item.get("content", "").encode("utf-8")).hexdigest()


def cosine_sim(vec_a: list[float], vec_b: list[float]) -> float:
    """Cosine similarity helper for MMR."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    # TODO: Implement cross-encoder reranking
    #
    # Option A: Jina Reranker API
    # import requests
    # response = requests.post(
    #     "https://api.jina.ai/v1/rerank",
    #     headers={"Authorization": f"Bearer {JINA_API_KEY}"},
    #     json={
    #         "model": "jina-reranker-v2-base-multilingual",
    #         "query": query,
    #         "documents": [c["content"] for c in candidates],
    #         "top_n": top_k
    #     }
    # )
    # reranked = response.json()["results"]
    # return [
    #     {**candidates[r["index"]], "score": r["relevance_score"]}
    #     for r in reranked
    # ]
    #
    # Option B: Local model (Qwen3-Reranker)
    # from transformers import AutoModelForSequenceClassification, AutoTokenizer
    # ...
    if not candidates:
        return []

    max_original_score = max(abs(float(c.get("score", 0.0))) for c in candidates) or 1.0
    reranked = []

    for candidate in candidates:
        original_score = float(candidate.get("score", 0.0)) / max_original_score
        overlap = lexical_overlap_score(query, candidate.get("content", ""))
        rerank_score = 0.65 * overlap + 0.35 * original_score

        item = candidate.copy()
        item["score"] = float(rerank_score)
        item["rerank_score"] = float(rerank_score)
        item["original_score"] = float(candidate.get("score", 0.0))
        item["rerank_method"] = "local_overlap_fallback"
        reranked.append(item)

    reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
    return reranked[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    # TODO: Implement MMR
    #
    # selected = []
    # remaining = list(range(len(candidates)))
    #
    # for _ in range(min(top_k, len(candidates))):
    #     best_idx = None
    #     best_score = float('-inf')
    #
    #     for idx in remaining:
    #         # Relevance to query
    #         relevance = cosine_sim(query_embedding, candidates[idx]["embedding"])
    #
    #         # Max similarity to already selected
    #         max_sim_to_selected = 0
    #         for sel_idx in selected:
    #             sim = cosine_sim(candidates[idx]["embedding"], candidates[sel_idx]["embedding"])
    #             max_sim_to_selected = max(max_sim_to_selected, sim)
    #
    #         # MMR score
    #         mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected
    #
    #         if mmr_score > best_score:
    #             best_score = mmr_score
    #             best_idx = idx
    #
    #     selected.append(best_idx)
    #     remaining.remove(best_idx)
    #
    # return [candidates[i] for i in selected]
    if not candidates:
        return []

    selected: list[int] = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx: Optional[int] = None
        best_score = float("-inf")

        for idx in remaining:
            candidate_embedding = candidates[idx].get("embedding", [])
            relevance = cosine_sim(query_embedding, candidate_embedding)
            if relevance == 0.0:
                relevance = float(candidates[idx].get("score", 0.0))

            max_sim_to_selected = 0.0
            for selected_idx in selected:
                sim = cosine_sim(
                    candidate_embedding,
                    candidates[selected_idx].get("embedding", []),
                )
                max_sim_to_selected = max(max_sim_to_selected, sim)

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is None:
            break

        selected.append(best_idx)
        remaining.remove(best_idx)

    results = []
    for idx in selected:
        item = candidates[idx].copy()
        item["score"] = float(item.get("score", 0.0))
        item["mmr_selected"] = True
        item["rerank_method"] = "mmr"
        results.append(item)
    return results


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    # TODO: Implement RRF
    #
    # rrf_scores = {}  # content -> score
    # content_map = {}  # content -> full dict
    #
    # for ranked_list in ranked_lists:
    #     for rank, item in enumerate(ranked_list, 1):
    #         key = item["content"]
    #         rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k + rank)
    #         content_map[key] = item
    #
    # # Sort by RRF score
    # sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    #
    # results = []
    # for content, score in sorted_items[:top_k]:
    #     item = content_map[content].copy()
    #     item["score"] = score
    #     results.append(item)
    #
    # return results
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}
    original_scores: dict[str, list[float]] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = content_key(item)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map.setdefault(key, item)
            original_scores.setdefault(key, []).append(float(item.get("score", 0.0)))

    sorted_keys = sorted(rrf_scores, key=lambda key: rrf_scores[key], reverse=True)

    results = []
    for key in sorted_keys[:top_k]:
        item = content_map[key].copy()
        item["score"] = float(rrf_scores[key])
        item["rrf_score"] = float(rrf_scores[key])
        item["original_scores"] = original_scores.get(key, [])
        item["rerank_method"] = "rrf"
        results.append(item)

    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking

    Returns:
        List of top_k reranked candidates.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        # Cần query_embedding - embed query trước
        query_embedding = candidates[0].get("embedding", []) if candidates else []
        return rerank_mmr(query_embedding, candidates, top_k)
    elif method == "rrf":
        # RRF cần nhiều ranked lists - gọi riêng
        return rerank_rrf([candidates], top_k)
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Test with dummy data
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
