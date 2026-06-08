"""Lightweight hybrid retrieval for the team RAG chatbot."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


TEAM_DIR = Path(__file__).resolve().parents[1]
KNOWLEDGE_BASE_PATH = TEAM_DIR / "data" / "knowledge_base.json"


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 5
    mode: str = "hybrid"
    use_reranking: bool = True


def tokenize(text: str) -> list[str]:
    """Tokenize Vietnamese/ASCII text with a small stopword filter."""
    normalized = unicodedata.normalize("NFD", text.lower())
    normalized = "".join(char for char in normalized if unicodedata.category(char) != "Mn")
    normalized = normalized.replace("đ", "d")
    tokens = re.findall(r"\w+", normalized, flags=re.UNICODE)
    stopwords = {
        "la",
        "là",
        "cua",
        "của",
        "va",
        "và",
        "ve",
        "về",
        "theo",
        "cho",
        "trong",
        "nhung",
        "những",
        "cac",
        "các",
        "gi",
        "gì",
        "nao",
        "nào",
        "co",
        "có",
        "duoc",
        "được",
    }
    return [token for token in tokens if token not in stopwords and len(token) > 1]


def load_documents(path: Path = KNOWLEDGE_BASE_PATH) -> list[dict[str, Any]]:
    """Load team knowledge base documents."""
    with path.open("r", encoding="utf-8") as f:
        documents = json.load(f)

    for doc in documents:
        doc.setdefault("metadata", {})
        doc["metadata"].update(
            {
                "id": doc["id"],
                "title": doc["title"],
                "source": doc["source"],
                "year": doc["year"],
                "doc_type": doc["doc_type"],
            }
        )
    return documents


def _document_frequency(documents: list[dict[str, Any]]) -> Counter[str]:
    df: Counter[str] = Counter()
    for doc in documents:
        df.update(set(tokenize(f"{doc['title']} {doc['content']} {doc['source']}")))
    return df


def _idf(term: str, total_docs: int, df: Counter[str]) -> float:
    return math.log((total_docs + 1) / (df.get(term, 0) + 1)) + 1.0


def _lexical_score(query_tokens: list[str], doc_text: str, total_docs: int, df: Counter[str]) -> float:
    if not query_tokens:
        return 0.0

    doc_tokens = tokenize(doc_text)
    doc_counts = Counter(doc_tokens)
    max_tf = max(doc_counts.values(), default=1)

    score = 0.0
    possible = 0.0
    for token in query_tokens:
        weight = _idf(token, total_docs, df)
        possible += weight
        score += (doc_counts.get(token, 0) / max_tf) * weight

    coverage = len(set(query_tokens) & set(doc_tokens)) / max(len(set(query_tokens)), 1)
    return min((score / possible) * 0.75 + coverage * 0.25, 1.0)


def _semantic_score(query: str, doc_text: str) -> float:
    query_norm = " ".join(tokenize(query))
    doc_norm = " ".join(tokenize(doc_text))
    if not query_norm or not doc_norm:
        return 0.0

    sequence_score = SequenceMatcher(None, query_norm, doc_norm[:1200]).ratio()
    query_terms = set(query_norm.split())
    doc_terms = set(doc_norm.split())
    jaccard = len(query_terms & doc_terms) / max(len(query_terms | doc_terms), 1)
    return min(sequence_score * 0.45 + jaccard * 0.55, 1.0)


def _normalize(results: list[dict[str, Any]], key: str) -> None:
    max_score = max((item[key] for item in results), default=0.0)
    if max_score <= 0:
        return
    for item in results:
        item[key] = item[key] / max_score


def _rank_documents(query: str, documents: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    query_tokens = tokenize(query)
    df = _document_frequency(documents)
    total_docs = len(documents)
    ranked: list[dict[str, Any]] = []
    vector_scores: dict[str, float] = {}

    if mode in {"vector", "hybrid_vector"}:
        from .vector_store import vector_search

        vector_results = vector_search(query, top_k=len(documents))
        vector_scores = {
            item["metadata"]["id"]: float(item["score"])
            for item in vector_results
        }

    for doc in documents:
        doc_text = f"{doc['title']} {doc['source']} {doc['content']}"
        lexical = _lexical_score(query_tokens, doc_text, total_docs, df)
        semantic = _semantic_score(query, doc_text)
        vector = vector_scores.get(doc["id"], 0.0)

        if mode == "lexical_only":
            score = lexical
        elif mode == "dense_only":
            score = semantic
        elif mode == "vector":
            score = vector
        elif mode == "hybrid_vector":
            score = lexical * 0.35 + semantic * 0.15 + vector * 0.50
        else:
            score = lexical * 0.65 + semantic * 0.35

        ranked.append(
            {
                "content": doc["content"],
                "score": score,
                "metadata": doc["metadata"],
                "_lexical_score": lexical,
                "_semantic_score": semantic,
                "_vector_score": vector,
            }
        )

    _normalize(ranked, "score")
    return ranked


def _rerank(query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    query_terms = set(tokenize(query))
    query_numbers = {term for term in query_terms if term.isdigit()}
    reranked = []
    for item in candidates:
        metadata = item["metadata"]
        title_terms = set(tokenize(metadata.get("title", "")))
        source_terms = set(tokenize(metadata.get("source", "")))
        metadata_terms = title_terms | source_terms
        title_bonus = len(query_terms & title_terms) / max(len(query_terms), 1)
        source_bonus = len(query_terms & source_terms) / max(len(query_terms), 1)
        number_bonus = 0.25 if query_numbers and query_numbers & metadata_terms else 0.0
        rerank_score = item["score"] * 0.75 + title_bonus * 0.1 + source_bonus * 0.08 + number_bonus
        reranked.append({**item, "score": min(rerank_score, 1.0)})

    return sorted(reranked, key=lambda item: item["score"], reverse=True)


def retrieve(
    query: str,
    top_k: int = 5,
    mode: str = "hybrid",
    use_reranking: bool = True,
) -> list[dict[str, Any]]:
    """Retrieve relevant documents for a user query."""
    if not query.strip():
        return []

    documents = load_documents()
    candidates = _rank_documents(query, documents, mode=mode)
    candidates = sorted(candidates, key=lambda item: item["score"], reverse=True)

    if use_reranking:
        candidates = _rerank(query, candidates[: max(top_k * 3, top_k)])

    output = []
    for item in candidates[:top_k]:
        output.append(
            {
                "content": item["content"],
                "score": round(float(item["score"]), 4),
                "metadata": item["metadata"],
            }
        )
    return output
