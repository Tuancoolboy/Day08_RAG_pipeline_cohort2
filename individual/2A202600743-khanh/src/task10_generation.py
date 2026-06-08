"""Task 10 - Generation with source citations."""

import os
import re

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> None:
        return None

from .task9_retrieval_pipeline import retrieve

load_dotenv()

# Use 5 chunks because it usually gives enough legal/news evidence for citation
# while keeping the prompt short enough to reduce lost-in-the-middle effects.
TOP_K = 5

# Use nucleus sampling at 0.9 so the model can phrase Vietnamese answers
# naturally, but still stays constrained by the retrieved context.
TOP_P = 0.9

# Keep temperature low for factual RAG answers; citations matter more than
# creative wording in this task.
TEMPERATURE = 0.3
USE_OPENAI_API = os.getenv("USE_OPENAI_API", "false").lower() in {"1", "true", "yes"}
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = """Answer in Vietnamese using only provided context.
Every factual statement needs a bracketed citation. If evidence is missing,
say: I cannot verify this information."""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    if len(chunks) <= 2:
        return chunks
    front = chunks[::2]
    back = list(reversed(chunks[1::2]))
    return front + back


def _source_label(chunk: dict, index: int) -> str:
    metadata = chunk.get("metadata", {})
    source = metadata.get("source") or metadata.get("path") or f"Source {index}"
    return source.replace(".md", "").replace(".pdf", "")


def format_context(chunks: list[dict]) -> str:
    parts = []
    for index, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        parts.append(
            f"[Document {index} | Source: {_source_label(chunk, index)} | "
            f"Type: {metadata.get('type', 'unknown')} | Score: {chunk.get('score', 0):.3f}]\n"
            f"{chunk.get('content', '')}"
        )
    return "\n\n---\n\n".join(parts)


def _pick_evidence_sentences(chunks: list[dict], limit: int = 4) -> list[str]:
    evidence = []
    for index, chunk in enumerate(chunks, 1):
        label = _source_label(chunk, index)
        sentences = re.split(r"(?<=[.!?])\s+|\n+", chunk.get("content", ""))
        for sentence in sentences:
            sentence = " ".join(sentence.split())
            if len(sentence) < 40:
                continue
            evidence.append(f"{sentence} [{label}]")
            break
        if len(evidence) >= limit:
            break
    return evidence


def _local_generate(reordered_chunks: list[dict]) -> str:
    evidence = _pick_evidence_sentences(reordered_chunks)
    if not evidence:
        return "I cannot verify this information."
    return " ".join(evidence)


def _openai_generate(query: str, context: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("sk-xxx"):
        raise RuntimeError("OPENAI_API_KEY is missing or still uses the placeholder value")

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            },
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    return response.choices[0].message.content or "I cannot verify this information."


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    chunks = retrieve(query, top_k=top_k)
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    generation_source = "local_fallback"
    if USE_OPENAI_API:
        try:
            answer = _openai_generate(query, context)
            generation_source = f"openai:{OPENAI_MODEL}"
        except Exception as exc:
            print(f"OpenAI generation unavailable; using local fallback: {exc}")
            answer = _local_generate(reordered)
    else:
        answer = _local_generate(reordered)

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "none") if chunks else "none",
        "context": context,
        "generation_config": {
            "top_k": top_k,
            "top_p": TOP_P,
            "temperature": TEMPERATURE,
            "generation_source": generation_source,
            "model": OPENAI_MODEL if generation_source.startswith("openai:") else None,
        },
    }


if __name__ == "__main__":
    print(generate_with_citation("hinh phat tang tru ma tuy")["answer"])
