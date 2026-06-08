"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

import json
import re
import sys
from pathlib import Path

INDEX_PATH = Path(__file__).parent.parent / "data" / "index" / "vector_store.json"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def tokenize(text: str) -> list[str]:
    """
    Tokenize text for BM25.

    Regex keeps Vietnamese words and numbers while removing punctuation/noisy
    markdown symbols, which works well for this small legal/news corpus.
    """
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def load_corpus() -> list[dict]:
    """Load chunks from the local index created by Task 4."""
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Index file not found: {INDEX_PATH}. Run task4_chunking_indexing.py first."
        )

    index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    corpus = []
    for chunk in index.get("chunks", []):
        content = chunk.get("content", "").strip()
        if not content:
            continue
        corpus.append({
            "content": content,
            "metadata": chunk.get("metadata", {}),
        })
    return corpus

# TODO: Load corpus từ data/standardized/ hoặc từ vector store
CORPUS: list[dict] = load_corpus()  # List of {'content': str, 'metadata': dict}


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    # TODO: Implement BM25 index
    #
    # from rank_bm25 import BM25Okapi
    #
    # # Tokenize - cho tiếng Việt nên dùng underthesea hoặc đơn giản split()
    # tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
    # bm25 = BM25Okapi(tokenized_corpus)
    # return bm25
    from rank_bm25 import BM25Okapi

    tokenized_corpus = [tokenize(doc["content"]) for doc in corpus]
    return BM25Okapi(tokenized_corpus)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    # TODO: Implement lexical search
    #
    # tokenized_query = query.lower().split()
    # scores = bm25.get_scores(tokenized_query)
    #
    # # Get top_k indices
    # import numpy as np
    # top_indices = np.argsort(scores)[::-1][:top_k]
    #
    # results = []
    # for idx in top_indices:
    #     if scores[idx] > 0:
    #         results.append({
    #             "content": CORPUS[idx]["content"],
    #             "score": float(scores[idx]),
    #             "metadata": CORPUS[idx]["metadata"]
    #         })
    # return results
    if not query.strip() or not CORPUS:
        return []

    tokenized_query = tokenize(query)
    if not tokenized_query:
        return []

    bm25 = build_bm25_index(CORPUS)
    scores = bm25.get_scores(tokenized_query)
    ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)

    results = []
    for idx in ranked_indices[:top_k]:
        score = float(scores[idx])
        if score <= 0:
            continue
        results.append({
            "content": CORPUS[idx]["content"],
            "score": score,
            "metadata": CORPUS[idx]["metadata"],
        })
    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
