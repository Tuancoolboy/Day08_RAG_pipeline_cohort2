"""Task 8 - PageIndex vectorless retrieval.

Set PAGEINDEX_USE_API=true plus PAGEINDEX_API_KEY and PAGEINDEX_DOC_IDS in .env
to query PageIndex Cloud. Without those values, the local fallback keeps tests and
offline demos runnable.
"""

import os
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> None:
        return None

from .task4_chunking_indexing import load_index
from .task5_semantic_search import tokenize

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
PAGEINDEX_USE_API = os.getenv("PAGEINDEX_USE_API", "false").lower() in {"1", "true", "yes"}
PAGEINDEX_DOC_IDS = [doc_id.strip() for doc_id in os.getenv("PAGEINDEX_DOC_IDS", "").split(",") if doc_id.strip()]
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
LANDING_LEGAL_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
PAGEINDEX_API_BASE = "https://api.pageindex.ai"


def _headers() -> dict[str, str]:
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("PAGEINDEX_API_KEY is missing")
    return {"api_key": PAGEINDEX_API_KEY}


def upload_documents() -> list[dict]:
    """Upload local PDF legal documents to PageIndex Cloud and return doc records."""
    if not PAGEINDEX_API_KEY:
        return [{"local_file": str(path)} for path in sorted(STANDARDIZED_DIR.rglob("*.md"))]

    import requests

    uploaded = []
    for pdf_file in sorted(LANDING_LEGAL_DIR.glob("*.pdf")):
        with pdf_file.open("rb") as file:
            response = requests.post(
                f"{PAGEINDEX_API_BASE}/doc/",
                headers=_headers(),
                files={"file": file},
                data={"if_retrieval": "true"},
                timeout=120,
            )
        response.raise_for_status()
        record = response.json()
        record["filename"] = pdf_file.name
        uploaded.append(record)
    return uploaded


def wait_until_ready(doc_id: str, timeout_seconds: int = 600, poll_seconds: int = 10) -> dict:
    """Poll PageIndex until a document is completed and retrieval_ready."""
    import requests

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        response = requests.get(
            f"{PAGEINDEX_API_BASE}/doc/{doc_id}/?type=tree",
            headers=_headers(),
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("status") == "completed" and data.get("retrieval_ready"):
            return data
        time.sleep(poll_seconds)
    raise TimeoutError(f"PageIndex document is not ready after {timeout_seconds}s: {doc_id}")


def _pageindex_api_search(query: str, top_k: int = 5) -> list[dict]:
    """Use the legacy PageIndex Retrieval API for processed doc IDs."""
    import requests

    def iter_relevant_contents(value):
        if isinstance(value, dict):
            yield value
        elif isinstance(value, str):
            yield {"relevant_content": value}
        elif isinstance(value, list):
            for item in value:
                yield from iter_relevant_contents(item)

    results = []
    for doc_id in PAGEINDEX_DOC_IDS:
        payload = {"doc_id": doc_id, "query": query, "thinking": False}
        response = requests.post(
            f"{PAGEINDEX_API_BASE}/retrieval/",
            headers=_headers(),
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        retrieval_id = response.json()["retrieval_id"]

        for _ in range(30):
            status_response = requests.get(
                f"{PAGEINDEX_API_BASE}/retrieval/{retrieval_id}/",
                headers=_headers(),
                timeout=60,
            )
            status_response.raise_for_status()
            data = status_response.json()
            if data.get("status") == "completed":
                for node in data.get("retrieved_nodes", []):
                    for content in iter_relevant_contents(node.get("relevant_contents", [])):
                        text = content.get("relevant_content", "")
                        if text:
                            results.append(
                                {
                                    "content": text,
                                    "score": 1.0,
                                    "metadata": {
                                        "doc_id": doc_id,
                                        "node_id": node.get("node_id"),
                                        "title": node.get("title"),
                                        "page_index": content.get("page_index"),
                                    },
                                    "source": "pageindex",
                                }
                            )
                break
            time.sleep(2)

    return results[:top_k]


def _local_pageindex_fallback(query: str, top_k: int = 5) -> list[dict]:
    """Offline vectorless fallback using section/token overlap over local chunks."""
    query_terms = set(tokenize(query))
    results = []
    for chunk in load_index():
        doc_terms = set(tokenize(chunk["content"]))
        overlap = len(query_terms & doc_terms)
        score = overlap / len(query_terms) if query_terms else 0.0
        if score <= 0:
            continue
        results.append(
            {
                "content": chunk["content"],
                "score": float(score),
                "metadata": chunk.get("metadata", {}),
                "source": "pageindex",
            }
        )
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Vectorless retrieval using PageIndex API when available, else local fallback."""
    if PAGEINDEX_USE_API and PAGEINDEX_API_KEY and PAGEINDEX_DOC_IDS:
        try:
            results = _pageindex_api_search(query, top_k=top_k)
            if results:
                return results
        except Exception as exc:
            print(f"PageIndex API unavailable; using local fallback: {exc}")

    return _local_pageindex_fallback(query, top_k=top_k)


if __name__ == "__main__":
    for result in pageindex_search("ma tuy", top_k=3):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
