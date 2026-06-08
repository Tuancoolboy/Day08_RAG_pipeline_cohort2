"""Task 1 - Collect legal documents about narcotics and controlled substances.

The downloadable document list is kept in ``data/sources/legal_documents.json``
so the downloader is reusable and the source provenance is easy to audit.
"""

import json
from pathlib import Path
from urllib.parse import unquote
from urllib.request import Request, urlopen

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data" / "landing" / "legal"
SOURCE_MANIFEST = PROJECT_DIR / "data" / "sources" / "legal_documents.json"
REPORT_PATH = DATA_DIR / "download_report.json"
MIN_FILE_SIZE_BYTES = 1024


class DownloadError(RuntimeError):
    """Raised when a legal document cannot be downloaded or validated."""


def setup_directory() -> None:
    """Create data/landing/legal/ if it does not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Directory ready: {DATA_DIR}")


def load_sources(manifest_path: Path = SOURCE_MANIFEST) -> list[dict]:
    """Load document source definitions from the JSON manifest."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing source manifest: {manifest_path}")

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    documents = data.get("documents", [])
    if not documents:
        raise DownloadError(f"No documents configured in {manifest_path}")
    return documents


def _assert_safe_filename(filename: str) -> None:
    if Path(filename).name != filename:
        raise ValueError(f"Filename must not include directories: {filename}")
    if Path(filename).suffix.lower() not in {".pdf", ".doc", ".docx"}:
        raise ValueError(f"Unsupported legal document extension: {filename}")


def download_file(document: dict, output_dir: Path = DATA_DIR) -> dict:
    """Download one configured legal document and return a small audit record."""
    filename = document["filename"]
    url = document["download_url"]
    _assert_safe_filename(filename)

    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename

    request = Request(url, headers={"User-Agent": "Day08-RAG-Pipeline/1.0"})
    with urlopen(request, timeout=60) as response:
        content = response.read()
        content_type = response.headers.get("content-type", "")
    if len(content) <= MIN_FILE_SIZE_BYTES:
        raise DownloadError(f"{filename} is too small: {len(content)} bytes")

    filepath.write_bytes(content)
    return {
        "title": document.get("title", filename),
        "number": document.get("number"),
        "filename": filename,
        "bytes": len(content),
        "content_type": content_type,
        "download_url": unquote(url),
        "source_page": document.get("source_page"),
        "saved_to": str(filepath),
    }


def download_all(manifest_path: Path = SOURCE_MANIFEST) -> list[dict]:
    """Download every legal document listed in the manifest."""
    setup_directory()
    report = []
    for document in load_sources(manifest_path):
        record = download_file(document)
        report.append(record)
        print(f"Downloaded {record['filename']} ({record['bytes']} bytes)")

    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report written: {REPORT_PATH}")
    return report


if __name__ == "__main__":
    download_all()
