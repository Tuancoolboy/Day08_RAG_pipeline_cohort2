"""Offline evaluation pipeline for the team RAG chatbot.

This script avoids API calls so the group can run it during demo without
spending credits. It measures retrieval and answer quality with deterministic
token-overlap metrics, then exports a markdown report.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency is declared in requirements.txt
    def load_dotenv(*args, **kwargs):
        return False


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
load_dotenv(ROOT_DIR / ".env")

from team.src.pipeline import answer_question  # noqa: E402
from team.src.retrieval import tokenize  # noqa: E402


EVAL_DIR = Path(__file__).resolve().parent
GOLDEN_DATASET_PATH = EVAL_DIR / "golden_dataset.json"
RESULTS_PATH = EVAL_DIR / "results.md"
USE_OPENAI_FOR_EVAL = os.getenv("TEAM_EVAL_USE_OPENAI", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


CONFIGS = {
    "hybrid_vector_rerank": {
        "retrieval_mode": "hybrid_vector",
        "use_reranking": True,
    },
    "vector_only": {
        "retrieval_mode": "vector",
        "use_reranking": False,
    },
    "hybrid_rerank": {
        "retrieval_mode": "hybrid",
        "use_reranking": True,
    },
    "dense_only": {
        "retrieval_mode": "dense_only",
        "use_reranking": False,
    },
}


def load_golden_dataset() -> list[dict[str, str]]:
    """Load golden dataset from JSON file."""
    with GOLDEN_DATASET_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _token_set(text: str) -> set[str]:
    return set(tokenize(text))


def token_overlap(reference: str, candidate: str) -> float:
    """Return token recall against a reference string."""
    reference_tokens = _token_set(reference)
    if not reference_tokens:
        return 0.0
    candidate_tokens = _token_set(candidate)
    return len(reference_tokens & candidate_tokens) / len(reference_tokens)


def context_recall(expected_context: str, sources: list[dict[str, Any]]) -> float:
    """Measure whether retrieved sources include the expected source/context."""
    combined_context = " ".join(
        f"{source.get('metadata', {}).get('source', '')} "
        f"{source.get('metadata', {}).get('title', '')} "
        f"{source.get('content', '')}"
        for source in sources
    )
    return token_overlap(expected_context, combined_context)


def context_precision(question: str, sources: list[dict[str, Any]]) -> float:
    """Average retrieved source score weighted by rough question overlap."""
    if not sources:
        return 0.0

    scores = []
    for source in sources:
        overlap = token_overlap(question, source.get("content", ""))
        retrieval_score = float(source.get("score", 0.0))
        scores.append((overlap + retrieval_score) / 2)
    return statistics.mean(scores)


def evaluate_config(
    golden_dataset: list[dict[str, str]],
    retrieval_mode: str,
    use_reranking: bool,
) -> dict[str, Any]:
    """Run evaluation for one retrieval config."""
    rows = []
    for item in golden_dataset:
        result = answer_question(
            item["question"],
            top_k=5,
            use_openai=USE_OPENAI_FOR_EVAL,
            retrieval_mode=retrieval_mode,
            use_reranking=use_reranking,
        )
        answer = result["answer"]
        sources = result["sources"]

        row = {
            "question": item["question"],
            "answer": answer,
            "expected_answer": item["expected_answer"],
            "expected_context": item["expected_context"],
            "faithfulness": token_overlap(answer, " ".join(s["content"] for s in sources)),
            "answer_relevance": token_overlap(item["expected_answer"], answer),
            "context_recall": context_recall(item["expected_context"], sources),
            "context_precision": context_precision(item["question"], sources),
            "top_source": sources[0]["metadata"]["source"] if sources else "",
        }
        rows.append(row)

    aggregate = {
        "faithfulness": statistics.mean(row["faithfulness"] for row in rows),
        "answer_relevance": statistics.mean(row["answer_relevance"] for row in rows),
        "context_recall": statistics.mean(row["context_recall"] for row in rows),
        "context_precision": statistics.mean(row["context_precision"] for row in rows),
    }

    return {"aggregate": aggregate, "rows": rows}


def compare_configs(golden_dataset: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    """Compare at least two retrieval/generation configs."""
    results = {}
    for config_name, params in CONFIGS.items():
        results[config_name] = evaluate_config(golden_dataset, **params)
    return results


def _format_score(value: float) -> str:
    return f"{value:.3f}"


def _worst_rows(rows: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            row["answer_relevance"] + row["context_recall"] + row["context_precision"]
        )
        / 3,
    )[:limit]


def export_results(results: dict[str, dict[str, Any]]) -> None:
    """Export evaluation report to results.md."""
    lines = [
        "# RAG Evaluation Results",
        "",
        "## Method",
        "",
        "Evaluation uses OpenAI generation when `TEAM_EVAL_USE_OPENAI=1`; otherwise it uses local fallback generation.",
        "Vector configs search the persisted local vector database at `team/vector_store/index.json`.",
        "",
        "## Overall Scores",
        "",
        "| Config | Faithfulness | Answer Relevance | Context Recall | Context Precision |",
        "|--------|--------------|------------------|----------------|-------------------|",
    ]

    for config_name, result in results.items():
        aggregate = result["aggregate"]
        lines.append(
            "| "
            f"{config_name} | "
            f"{_format_score(aggregate['faithfulness'])} | "
            f"{_format_score(aggregate['answer_relevance'])} | "
            f"{_format_score(aggregate['context_recall'])} | "
            f"{_format_score(aggregate['context_precision'])} |"
        )

    best_config = max(
        results.items(),
        key=lambda item: statistics.mean(item[1]["aggregate"].values()),
    )[0]

    lines.extend(
        [
            "",
            "## A/B Comparison",
            "",
            f"Best overall config: `{best_config}`.",
            "",
            "Compared configs:",
            "",
            "- `hybrid_vector_rerank`: persisted vector database + lexical/semantic signals + reranking.",
            "- `vector_only`: persisted vector database only.",
            "- `hybrid_rerank`: hybrid lexical/semantic retrieval with reranking.",
            "- `dense_only`: semantic-style retrieval without reranking.",
            "",
            "## Worst Performers",
            "",
            "| Question | Top Source | Answer Relevance | Context Recall | Context Precision |",
            "|----------|------------|------------------|----------------|-------------------|",
        ]
    )

    for row in _worst_rows(results[best_config]["rows"]):
        question = row["question"].replace("|", " ")
        source = row["top_source"].replace("|", " ")
        lines.append(
            "| "
            f"{question} | "
            f"{source} | "
            f"{_format_score(row['answer_relevance'])} | "
            f"{_format_score(row['context_recall'])} | "
            f"{_format_score(row['context_precision'])} |"
        )

    lines.extend(
        [
            "",
            "## Recommendations",
            "",
            "- Add more real legal documents and crawled news articles to increase context coverage.",
            "- Replace local hashed embeddings with sentence-transformers or Weaviate Cloud when deployment is available.",
            "- Keep vector or hybrid-vector retrieval enabled for demo because it improves source ordering for legal queries.",
            "- Review low-scoring questions and add more granular chunks for specific articles.",
            "",
        ]
    )

    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} test cases")

    results = compare_configs(golden_dataset)
    export_results(results)

    print(f"Exported results to {RESULTS_PATH}")
    for config_name, result in results.items():
        aggregate = result["aggregate"]
        print(
            f"{config_name}: "
            f"faithfulness={aggregate['faithfulness']:.3f}, "
            f"answer_relevance={aggregate['answer_relevance']:.3f}, "
            f"context_recall={aggregate['context_recall']:.3f}, "
            f"context_precision={aggregate['context_precision']:.3f}"
        )


if __name__ == "__main__":
    main()
