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
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from .task9_retrieval_pipeline import retrieve

# =============================================================================
# CONFIGURATION
# =============================================================================

TOP_K       = 5
TOP_P       = 0.9
TEMPERATURE = 0.3

GROQ_API_KEY   = os.getenv("GROQ_API_KEY",   "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

GROQ_MODELS = [
    "llama-3.1-8b-instant", 
    "llama3-8b-8192",  
    "llama-3.3-70b-versatile", 
    "mixtral-8x7b-32768", 
]

SYSTEM_PROMPT = """Bạn là trợ lý pháp lý chuyên về luật phòng chống ma tuý Việt Nam.
Hãy trả lời câu hỏi bằng tiếng Việt, dựa HOÀN TOÀN vào context được cung cấp.

Quy tắc bắt buộc:
1. Mọi thông tin đều PHẢI có citation dạng [Document X | tên nguồn]
2. Nếu context không đủ → trả lời: "Tôi không thể xác minh thông tin này từ nguồn hiện có."
3. KHÔNG bịa đặt thông tin ngoài context
4. Cấu trúc câu trả lời rõ ràng, có đoạn văn
"""


# =============================================================================
# DOCUMENT REORDERING
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Tránh lost-in-the-middle: [1,2,3,4,5] → [1,3,5,4,2]"""
    if len(chunks) <= 2:
        return chunks
    return chunks[::2] + chunks[1::2][::-1]


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """Format chunks thành context string có source label."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        parts.append(
            f"[Document {i} | {meta.get('source', f'Source {i}')} "
            f"| type={meta.get('type','unknown')} "
            f"| score={chunk.get('score', 0.0):.4f}]\n"
            f"{chunk['content'].strip()}"
        )
    return "\n\n---\n\n".join(parts)


# =============================================================================
# LLM CALLERS
# =============================================================================

def _call_groq(prompt: str) -> str:
    """Gọi Groq API với retry + sleep."""
    from groq import Groq

    client     = Groq(api_key=GROQ_API_KEY)
    last_error = None

    for model in GROQ_MODELS:
        # Mỗi model thử tối đa 3 lần
        for attempt in range(3):
            try:
                print(f"    [Groq] {model} (attempt {attempt+1}) ...")
                resp = client.chat.completions.create(
                    model       = model,
                    messages    = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                    temperature = TEMPERATURE,
                    top_p       = TOP_P,
                    max_tokens  = 512, 
                )
                print(f"    [Groq] ✓ {model} OK")
                return resp.choices[0].message.content.strip()

            except Exception as e:
                err = str(e)

                # Model bị xoá → bỏ qua, không retry
                if "decommissioned" in err or "not found" in err or "404" in err:
                    print(f"    [Groq] ✗ {model} decommissioned → skip")
                    break

                # Rate limit → sleep rồi retry
                if any(x in err for x in ["429", "rate", "quota", "limit"]):
                    wait = 10 * (attempt + 1) 
                    print(f"    [Groq] ⚠ rate limit → sleep {wait}s ...")
                    time.sleep(wait)
                    last_error = e
                    continue

                # Lỗi khác → raise ngay
                raise

    raise RuntimeError(f"Tất cả Groq models thất bại: {last_error}")


def _call_gemini(prompt: str) -> str:
    """Gọi Gemini API."""
    import google.generativeai as genai

    genai.configure(api_key=GEMINI_API_KEY)

    # Lấy model list thực tế
    try:
        available = [
            m.name.replace("models/", "")
            for m in genai.list_models()
            if "generateContent" in m.supported_generation_methods
        ]
    except Exception:
        available = []

    preferred = [
        "gemini-2.0-flash-lite", "gemini-2.0-flash",
        "gemini-1.5-flash-latest", "gemini-1.5-flash-8b",
    ]
    models = [m for m in preferred if m in available] or preferred
    last_error = None

    for model_name in models:
        try:
            print(f"    [Gemini] Trying: {model_name} ...")
            m    = genai.GenerativeModel(
                model_name         = model_name,
                system_instruction = SYSTEM_PROMPT,
                generation_config  = genai.GenerationConfig(
                    temperature=TEMPERATURE, top_p=TOP_P, max_output_tokens=2048
                ),
            )
            resp = m.generate_content(prompt)
            print(f"    [Gemini] ✓ {model_name} OK")
            return resp.text.strip()
        except Exception as e:
            err = str(e)
            if any(x in err for x in ["429", "404", "quota", "not found"]):
                print(f"    [Gemini] ⚠ {model_name}: skip → {err[:60]}")
                last_error = e
                continue
            raise

    raise RuntimeError(f"Tất cả Gemini models thất bại: {last_error}")


def _call_llm(prompt: str) -> str:
    """Groq trước → Gemini fallback."""
    if GROQ_API_KEY:
        try:
            return _call_groq(prompt)
        except Exception as e:
            print(f"  ⚠ Groq thất bại: {e}")

    if GEMINI_API_KEY:
        return _call_gemini(prompt)

    raise ValueError("Cần GROQ_API_KEY hoặc GEMINI_API_KEY trong .env")


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """End-to-end RAG generation có citation."""
    print(f"  [Debug] GROQ_API_KEY  : {'SET ✓' if GROQ_API_KEY   else 'EMPTY ✗'}")
    print(f"  [Debug] GEMINI_API_KEY: {'SET ✓' if GEMINI_API_KEY else 'EMPTY ✗'}")

    # Step 1: Retrieve
    print(f"  [1/4] Retrieving top {top_k} chunks ...")
    chunks = retrieve(query, top_k=top_k)

    if not chunks:
        return {
            "answer"          : "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources"         : [],
            "retrieval_source": "none",
        }

    retrieval_source = chunks[0].get("source", "hybrid")

    # Step 2: Reorder
    reordered = reorder_for_llm(chunks)

    # Step 3: Format
    context = format_context(reordered)
    user_message = (
        f"Context:\n\n{context}\n\n{'─'*60}\n\nCâu hỏi: {query}"
    )

    # Step 4: Call LLM
    print(f"  [4/4] Calling LLM ...")
    answer = _call_llm(user_message)
    print(f"  ✓ Generated {len(answer)} ký tự.")

    return {
        "answer"          : answer,
        "sources"         : chunks,
        "retrieval_source": retrieval_source,
    }


if __name__ == "__main__":
    result = generate_with_citation("Hình phạt tàng trữ ma tuý?")
    print(f"\nA:\n{result['answer']}")