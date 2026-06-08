"""Task 5 - Local semantic-style retrieval over the JSONL chunk index."""

import math
import re
from collections import Counter

from .task4_chunking_indexing import load_index


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE)


def _char_ngrams(text: str, n: int = 3) -> Counter:
    cleaned = re.sub(r"\s+", " ", text.lower())
    return Counter(cleaned[i : i + n] for i in range(max(0, len(cleaned) - n + 1)))


def _cosine(left: Counter, right: Counter) -> float:
    common = set(left) & set(right)
    numerator = sum(left[key] * right[key] for key in common)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Return top chunks by character n-gram cosine similarity."""
    corpus = load_index()
    query_vector = _char_ngrams(query)
    results = []
    for chunk in corpus:
        score = _cosine(query_vector, _char_ngrams(chunk["content"]))
        if score <= 0:
            continue
        results.append(
            {
                "content": chunk["content"],
                "score": float(score),
                "metadata": chunk.get("metadata", {}),
            }
        )
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    for result in semantic_search("hinh phat ma tuy", top_k=5):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
