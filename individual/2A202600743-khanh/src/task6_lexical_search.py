"""Task 6 - BM25 lexical search implemented with the Python standard library."""

import math
from collections import Counter

from .task4_chunking_indexing import load_index
from .task5_semantic_search import tokenize

K1 = 1.5
B = 0.75


def build_bm25_index(corpus: list[dict]) -> dict:
    tokenized = [tokenize(doc["content"]) for doc in corpus]
    doc_freq = Counter()
    for tokens in tokenized:
        doc_freq.update(set(tokens))
    avg_len = sum(len(tokens) for tokens in tokenized) / len(tokenized) if tokenized else 0
    return {
        "tokenized": tokenized,
        "term_counts": [Counter(tokens) for tokens in tokenized],
        "doc_freq": doc_freq,
        "avg_len": avg_len,
        "n_docs": len(corpus),
    }


def _bm25_score(query_tokens: list[str], doc_index: int, index: dict) -> float:
    score = 0.0
    doc_len = len(index["tokenized"][doc_index])
    if not doc_len or not index["avg_len"]:
        return 0.0
    term_counts = index["term_counts"][doc_index]
    for token in query_tokens:
        tf = term_counts.get(token, 0)
        if not tf:
            continue
        df = index["doc_freq"].get(token, 0)
        idf = math.log(1 + (index["n_docs"] - df + 0.5) / (df + 0.5))
        denominator = tf + K1 * (1 - B + B * doc_len / index["avg_len"])
        score += idf * (tf * (K1 + 1)) / denominator
    return score


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    corpus = load_index()
    if not corpus:
        return []
    index = build_bm25_index(corpus)
    query_tokens = tokenize(query)
    scored = []
    for idx, doc in enumerate(corpus):
        score = _bm25_score(query_tokens, idx, index)
        if score > 0:
            scored.append(
                {
                    "content": doc["content"],
                    "score": float(score),
                    "metadata": doc.get("metadata", {}),
                }
            )
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


if __name__ == "__main__":
    for result in lexical_search("ma tuy chat cam", top_k=5):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
