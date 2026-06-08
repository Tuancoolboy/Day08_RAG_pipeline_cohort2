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

import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


# ---------------------------------------------------------------------------
# Load corpus từ data/standardized/
# ---------------------------------------------------------------------------

def _load_corpus() -> list[dict]:
    """Đọc toàn bộ .md files từ data/standardized/ làm corpus."""
    corpus = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue
        doc_type = "legal" if "legal" in str(md_file) else "news"
        corpus.append({
            "content" : content,
            "metadata": {
                "source"  : md_file.name,
                "filepath": str(md_file),
                "type"    : doc_type,
            },
        })
    return corpus


CORPUS: list[dict] = _load_corpus()


# ---------------------------------------------------------------------------
# Tokenizer tiếng Việt đơn giản
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """
    Tokenize tiếng Việt:
      - Lowercase
      - Tách theo khoảng trắng (syllable-level, đủ dùng cho BM25)
      - Bỏ ký tự đặc biệt
    Nâng cao: thay bằng underthesea.word_tokenize() để tách từ ghép.
    """
    import re
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)   # bỏ dấu câu
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


# ---------------------------------------------------------------------------
# Build BM25 index
# ---------------------------------------------------------------------------

def build_bm25_index(corpus: list[dict]) -> BM25Okapi:
    """
    Xây dựng BM25 index từ corpus.
    Dùng BM25Okapi: k1=1.5, b=0.75 (mặc định tốt cho văn bản tiếng Việt).
    """
    tokenized_corpus = [_tokenize(doc["content"]) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)
    return bm25


# ---------------------------------------------------------------------------
# Lexical Search
# ---------------------------------------------------------------------------

def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
        Sorted by score descending.
    """
    if not CORPUS:
        raise ValueError("CORPUS rỗng — kiểm tra data/standardized/ có file .md không.")

    bm25 = build_bm25_index(CORPUS)

    # Tokenize query
    tokenized_query = _tokenize(query)

    # Tính BM25 score cho toàn bộ corpus
    scores = bm25.get_scores(tokenized_query)  # numpy array

    # Lấy top_k indices, sắp xếp descending
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] <= 0:
            continue  # bỏ qua doc không liên quan
        results.append({
            "content" : CORPUS[idx]["content"],
            "score"   : round(float(scores[idx]), 4),
            "metadata": CORPUS[idx]["metadata"],
        })

    return results


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Corpus size: {len(CORPUS)} documents\n")

    query   = "Điều 248 tàng trữ trái phép chất ma tuý"
    results = lexical_search(query, top_k=5)

    print(f"Query: {query}")
    print(f"Top {len(results)} results:\n")
    for i, r in enumerate(results, 1):
        print(f"[{i}] Score: {r['score']:.4f} | Source: {r['metadata']['source']}")
        print(f"     {r['content'][:150].strip()}...")
        print()