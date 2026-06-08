"""OpenAI-backed answer generation for the team RAG app."""

from __future__ import annotations

from typing import Any

from .config import ConfigError, OpenAISettings, load_openai_settings


SYSTEM_PROMPT = """You are a RAG assistant for Vietnamese legal and news documents.
Answer only from the provided context.
For every factual claim, include a citation close to the claim.
Use citation format [Source, Year] when metadata is available.
If the context is insufficient, say: I cannot verify this information."""


def _source_label(metadata: dict[str, Any]) -> str:
    source = metadata.get("source") or metadata.get("title") or "Unknown source"
    year = metadata.get("year") or metadata.get("date") or "n.d."
    return f"{source}, {year}"


def format_context(sources: list[dict[str, Any]]) -> str:
    """Format retrieved chunks for the model prompt."""
    if not sources:
        return "No context was retrieved."

    sections = []
    for index, source in enumerate(sources, start=1):
        content = str(source.get("content", "")).strip()
        metadata = source.get("metadata") or {}
        label = _source_label(metadata)
        sections.append(f"[{index}] Citation: [{label}]\nContent:\n{content}")

    return "\n\n".join(sections)


def build_messages(query: str, sources: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Build chat-completion messages for answer generation."""
    context = format_context(sources)
    user_prompt = (
        "Question:\n"
        f"{query.strip()}\n\n"
        "Retrieved context:\n"
        f"{context}\n\n"
        "Answer with citations. Do not guess."
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _error_response(message: str, sources: list[dict[str, Any]], error: str) -> dict[str, Any]:
    return {
        "answer": message,
        "sources": sources,
        "metadata": {"error": error},
    }


def generate_with_openai(
    query: str,
    sources: list[dict[str, Any]] | None = None,
    settings: OpenAISettings | None = None,
) -> dict[str, Any]:
    """
    Generate an answer from retrieved sources using OpenAI.

    Returns:
        {
            "answer": str,
            "sources": list[dict],
            "metadata": dict,
        }
    """
    safe_sources = sources or []

    try:
        runtime_settings = settings or load_openai_settings()
    except ConfigError as exc:
        return _error_response(str(exc), safe_sources, "config_error")

    try:
        from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
    except ImportError:
        return _error_response(
            "OpenAI package is not installed. Run: pip install -r requirements.txt",
            safe_sources,
            "missing_dependency",
        )

    client = OpenAI(
        api_key=runtime_settings.api_key,
        timeout=runtime_settings.timeout_seconds,
    )

    try:
        response = client.chat.completions.create(
            model=runtime_settings.model,
            messages=build_messages(query, safe_sources),
            temperature=0.2,
            top_p=0.9,
        )
    except APITimeoutError:
        return _error_response(
            "OpenAI request timed out. Please retry or increase OPENAI_TIMEOUT_SECONDS.",
            safe_sources,
            "timeout",
        )
    except APIConnectionError:
        return _error_response(
            "Cannot connect to OpenAI API. Please check network connection.",
            safe_sources,
            "connection_error",
        )
    except APIStatusError as exc:
        return _error_response(
            f"OpenAI API returned an error: HTTP {exc.status_code}.",
            safe_sources,
            "api_status_error",
        )

    answer = ""
    if response.choices:
        answer = (response.choices[0].message.content or "").strip()

    if not answer:
        answer = "I cannot verify this information."

    return {
        "answer": answer,
        "sources": safe_sources,
        "metadata": {"model": runtime_settings.model},
    }


def generate_with_citation(query: str, context_chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Compatibility wrapper used by evaluation and future UI code."""
    return generate_with_openai(query=query, sources=context_chunks)

