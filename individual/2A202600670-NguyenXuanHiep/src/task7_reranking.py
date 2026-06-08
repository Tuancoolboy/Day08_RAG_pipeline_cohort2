"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.
"""

import os
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Helper: cosine similarity
# ---------------------------------------------------------------------------

def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Tính cosine similarity giữa 2 vectors."""
    a, b  = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Cross-encoder: Jina Reranker API (multilingual, tốt cho tiếng Việt)
# ---------------------------------------------------------------------------

def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank bằng Jina Reranker v2 API (multilingual).

    Cơ chế: Cross-encoder nhận cặp (query, document) cùng lúc,
    attention có thể cross giữa 2 văn bản → chính xác hơn bi-encoder
    nhưng chậm hơn (không precompute được).
    """
    import requests

    JINA_API_KEY = os.getenv("JINA_API_KEY", "")
    if not JINA_API_KEY:
        raise ValueError("Thiếu JINA_API_KEY — set environment variable trước.")

    response = requests.post(
        "https://api.jina.ai/v1/rerank",
        headers={
            "Authorization": f"Bearer {JINA_API_KEY}",
            "Content-Type" : "application/json",
        },
        json={
            "model"    : "jina-reranker-v2-base-multilingual",
            "query"    : query,
            "documents": [c["content"] for c in candidates],
            "top_n"    : top_k,
        },
        timeout=30,
    )
    response.raise_for_status()
    reranked = response.json()["results"]

    return [
        {
            **candidates[r["index"]],
            "rerank_score": round(r["relevance_score"], 4),
            "score"       : round(r["relevance_score"], 4),
        }
        for r in reranked
    ]


# ---------------------------------------------------------------------------
# MMR — Maximal Marginal Relevance
# ---------------------------------------------------------------------------

def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Cơ chế:
      - λ gần 1.0 → ưu tiên relevance (giống semantic search thuần)
      - λ gần 0.0 → ưu tiên diversity (tránh kết quả trùng lặp)
      - λ=0.7: cân bằng, vừa relevant vừa đa dạng

    Yêu cầu: mỗi candidate phải có key 'embedding'.
    """
    if not candidates:
        return []

    # Kiểm tra embedding tồn tại
    if "embedding" not in candidates[0]:
        raise ValueError(
            "Candidates thiếu 'embedding' — "
            "gọi embed_chunks() trước hoặc dùng method='rrf'."
        )

    selected_indices  = []
    remaining_indices = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx   = None
        best_score = float("-inf")

        for idx in remaining_indices:
            # Relevance: similarity với query
            relevance = _cosine_sim(query_embedding, candidates[idx]["embedding"])

            # Diversity: max similarity với các doc đã chọn
            if selected_indices:
                max_sim = max(
                    _cosine_sim(
                        candidates[idx]["embedding"],
                        candidates[sel]["embedding"],
                    )
                    for sel in selected_indices
                )
            else:
                max_sim = 0.0

            # MMR score
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx   = idx

        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

    return [
        {**candidates[i], "score": round(
            _cosine_sim(query_embedding, candidates[i]["embedding"]), 4
        )}
        for i in selected_indices
    ]


# ---------------------------------------------------------------------------
# RRF — Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    RRF(d) = Σ 1 / (k + rank_r(d))

    Cơ chế:
      - Gộp kết quả từ nhiều ranker (semantic + lexical)
      - Mỗi document nhận điểm = tổng 1/(k + rank) từ tất cả ranker
      - k=60: hằng số làm mượt, giảm ảnh hưởng của rank 1 quá cao
      - Không cần normalize score giữa các ranker → robust

    Ví dụ:
      Semantic rank 1  → 1/(60+1) = 0.0164
      Lexical  rank 3  → 1/(60+3) = 0.0159
      RRF score        = 0.0164 + 0.0159 = 0.0323
    """
    rrf_scores  = {}   # content → rrf score
    content_map = {}   # content → full dict (giữ metadata)

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key]  = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map[key] = item

    # Sort descending theo RRF score
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item          = content_map[content].copy()
        item["score"] = round(score, 6)
        results.append(item)

    return results


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "rrf",            # "cross_encoder" | "mmr" | "rrf"
    query_embedding: Optional[list[float]] = None,
    ranked_lists   : Optional[list[list[dict]]] = None,
    lambda_param   : float = 0.7,
    rrf_k          : int   = 60,
) -> list[dict]:
    """
    Unified reranking interface.

    method="rrf"          → cần ranked_lists (list of ranked results)
    method="mmr"          → cần query_embedding + candidates có 'embedding'
    method="cross_encoder"→ cần JINA_API_KEY trong env
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)

    elif method == "mmr":
        if query_embedding is None:
            # Tự embed query nếu không truyền vào
            import torch
            from sentence_transformers import SentenceTransformer
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model  = SentenceTransformer("BAAI/bge-m3", device=device)
            query_embedding = model.encode(
                query,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).tolist()
        return rerank_mmr(query_embedding, candidates, top_k, lambda_param)

    elif method == "rrf":
        if ranked_lists is None:
            ranked_lists = [candidates]
        return rerank_rrf(ranked_lists, top_k, rrf_k)

    else:
        raise ValueError(f"Unknown method: {method}. Chọn: cross_encoder | mmr | rrf")


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Dummy candidates để test RRF (không cần API key)
    semantic_results = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.92, "metadata": {"source": "legal"}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ",     "score": 0.85, "metadata": {"source": "legal"}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý",           "score": 0.71, "metadata": {"source": "news"}},
    ]
    lexical_results = [
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ",     "score": 8.5,  "metadata": {"source": "legal"}},
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 7.2,  "metadata": {"source": "legal"}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý",           "score": 3.1,  "metadata": {"source": "news"}},
    ]

    print("=" * 50)
    print("Test RRF (Reciprocal Rank Fusion)")
    print("=" * 50)
    rrf_results = rerank(
        query        = "hình phạt tàng trữ ma tuý",
        candidates   = [],
        top_k        = 3,
        method       = "rrf",
        ranked_lists = [semantic_results, lexical_results],
    )
    for i, r in enumerate(rrf_results, 1):
        print(f"[{i}] RRF score: {r['score']:.6f} | {r['content']}")