"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "I cannot verify this information"
"""

import os
import re
import sys
from dotenv import load_dotenv

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:
    from .task9_retrieval_pipeline import retrieve
except ImportError:
    from task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ evidence mà không quá dài gây lost in the middle
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: đủ diverse nhưng không quá random
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.3 vì: RAG cần factual, ít sáng tạo
TEMPERATURE = 0.3


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Answer the following question comprehensively in Vietnamese.
For every statement of fact or claim, immediately insert a citation in brackets
linking to the specific source (e.g., [Luật Phòng chống ma tuý 2021, Điều 3]
or [VnExpress, 2024]).

If the information is not explicitly stated in the provided context or knowledge
base, state 'Tôi không thể xác minh thông tin này từ nguồn hiện có' rather than
guessing.

Rules:
- Only use information from the provided context
- Every factual claim MUST have a citation
- If context is insufficient, say so clearly
- Structure your answer with clear paragraphs"""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA.
    Strategy: đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]
    (best first, worst in middle, second-best last)

    Args:
        chunks: List sorted by score descending (from retrieval)

    Returns:
        List reordered để maximize LLM attention.
    """
    # TODO: Implement reordering
    #
    # if len(chunks) <= 2:
    #     return chunks
    #
    # # Split into first half (important → đầu) and second half (important → cuối)
    # reordered = []
    # for i in range(0, len(chunks), 2):
    #     reordered.append(chunks[i])  # Odd positions go first
    # for i in range(len(chunks) - 1 - (len(chunks) % 2 == 0), 0, -2):
    #     reordered.append(chunks[i])  # Even positions go last (reversed)
    #
    # return reordered
    if len(chunks) <= 2:
        return chunks

    reordered = []
    for i in range(0, len(chunks), 2):
        reordered.append(chunks[i])

    last_even_index = len(chunks) - 1 if len(chunks) % 2 == 0 else len(chunks) - 2
    for i in range(last_even_index, 0, -2):
        reordered.append(chunks[i])

    return reordered


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string.
    """
    # TODO: Implement context formatting
    #
    # context_parts = []
    # for i, chunk in enumerate(chunks, 1):
    #     source = chunk.get("metadata", {}).get("source", f"Source {i}")
    #     doc_type = chunk.get("metadata", {}).get("type", "unknown")
    #     context_parts.append(
    #         f"[Document {i} | Source: {source} | Type: {doc_type}]\n"
    #         f"{chunk['content']}\n"
    #     )
    # return "\n---\n".join(context_parts)
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source") or metadata.get("filename") or f"Source {i}"
        doc_type = metadata.get("type", "unknown")
        chunk_index = metadata.get("chunk_index", "n/a")
        score = float(chunk.get("score", 0.0))

        context_parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type} | "
            f"Chunk: {chunk_index} | Score: {score:.3f}]\n"
            f"{chunk.get('content', '')}\n"
        )

    return "\n---\n".join(context_parts)


def citation_label(chunk: dict) -> str:
    """Build a compact citation label from chunk metadata."""
    metadata = chunk.get("metadata", {})
    source = metadata.get("source") or metadata.get("filename") or "Nguon khong ro"
    chunk_index = metadata.get("chunk_index")
    if chunk_index is not None:
        return f"{source}, chunk {chunk_index}"
    return source


def split_sentences(text: str) -> list[str]:
    """Small sentence splitter for local extractive fallback."""
    sentences = re.split(r"(?<=[.!?。])\s+|\n+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def token_set(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def local_answer_with_citation(query: str, chunks: list[dict]) -> str:
    """
    Extractive fallback when no LLM API key is available.

    It only states sentences found in retrieved context and cites each sentence.
    """
    query_terms = token_set(query)
    evidence = []

    for chunk in chunks:
        label = citation_label(chunk)
        for sentence in split_sentences(chunk.get("content", "")):
            sentence_terms = token_set(sentence)
            overlap = len(query_terms & sentence_terms)
            if overlap > 0:
                evidence.append((overlap, sentence, label))

    evidence.sort(key=lambda item: item[0], reverse=True)

    if not evidence:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    answer_lines = []
    seen = set()
    for _, sentence, label in evidence[:5]:
        normalized = sentence.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        answer_lines.append(f"- {sentence} [{label}]")

    if not answer_lines:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    return "\n".join(answer_lines)


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call LLM
        6. Return answer + sources

    Args:
        query: Câu hỏi của user

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng
            'retrieval_source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    # TODO: Implement generation pipeline
    #
    # # Step 1: Retrieve
    # chunks = retrieve(query, top_k=top_k)
    #
    # # Step 2: Reorder
    # reordered = reorder_for_llm(chunks)
    #
    # # Step 3: Format context
    # context = format_context(reordered)
    #
    # # Step 4: Build prompt
    # user_message = f"""Context:\n{context}\n\n---\n\nQuestion: {query}"""
    #
    # # Step 5: Call LLM
    # from openai import OpenAI
    # client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    #
    # response = client.chat.completions.create(
    #     model="gpt-4o-mini",
    #     messages=[
    #         {"role": "system", "content": SYSTEM_PROMPT},
    #         {"role": "user", "content": user_message}
    #     ],
    #     temperature=TEMPERATURE,
    #     top_p=TOP_P,
    # )
    #
    # answer = response.choices[0].message.content
    #
    # # Step 6: Return
    # return {
    #     "answer": answer,
    #     "sources": chunks,
    #     "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none"
    # }
    chunks = retrieve(query, top_k=top_k)
    if not chunks:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
        }

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
            answer = response.choices[0].message.content
        except Exception as exc:
            answer = (
                f"Không gọi được LLM ({exc}). Dưới đây là câu trả lời trích xuất từ context:\n"
                + local_answer_with_citation(query, reordered)
            )
    else:
        answer = local_answer_with_citation(query, reordered)

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none",
    }


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
