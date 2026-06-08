"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install markitdown

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import json
from pathlib import Path

from markitdown import MarkItDown
import pdfplumber

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def safe_console(text: str) -> str:
    """Return text that can be printed on legacy Windows consoles."""
    return text.encode("ascii", errors="backslashreplace").decode("ascii")


def convert_pdf_to_markdown(filepath: Path, max_empty_pages: int = 3) -> str:
    """Extract text from a PDF into simple Markdown, with scan-PDF fallback."""
    parts = [f"# {filepath.stem}\n"]

    with pdfplumber.open(filepath) as pdf:
        empty_pages = 0
        for page_number, page in enumerate(pdf.pages, 1):
            text = (page.extract_text() or "").strip()
            if text:
                empty_pages = 0
                parts.append(f"\n\n## Page {page_number}\n\n{text}")
            else:
                empty_pages += 1
                if len(parts) == 1 and empty_pages >= max_empty_pages:
                    break

    if len(parts) == 1:
        parts.append(
            "\n\nNo extractable text was found in this PDF with the local text extractor. "
            "The file is likely a scanned legal document or image-based PDF, so OCR is "
            "required before the legal provisions can be indexed accurately. Keep this "
            "markdown file as a standardized placeholder that records the original source "
            "and explains why the legal text is unavailable for retrieval until OCR is run."
        )

    return "".join(parts).strip() + "\n"


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    md = MarkItDown()

    for filepath in legal_dir.iterdir():
        if filepath.suffix.lower() in (".pdf", ".docx", ".doc"):
            print(f"Converting: {safe_console(filepath.name)}")
            # TODO: Convert và lưu file
            if filepath.suffix.lower() == ".pdf":
                text_content = convert_pdf_to_markdown(filepath)
            else:
                result = md.convert(str(filepath))
                text_content = result.text_content
            output_path = output_dir / f"{filepath.stem}.md"
            output_path.write_text(text_content, encoding="utf-8")
            print(f"  Saved: {safe_console(str(output_path))}")
            # raise NotImplementedError("Implement convert_legal_docs")


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filepath in news_dir.iterdir():
        if filepath.suffix.lower() == ".json":
            print(f"Converting: {safe_console(filepath.name)}")
            # TODO: Đọc JSON, extract content_markdown, lưu thành .md
            data = json.loads(filepath.read_text(encoding="utf-8"))
            output_path = output_dir / f"{filepath.stem}.md"
            
            # Thêm metadata header
            header = f"# {data.get('title', 'Unknown')}\n\n"
            header += f"**Source:** {data.get('url', 'N/A')}\n"
            header += f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"
            
            content = header + data.get("content_markdown", "")
            output_path.write_text(content, encoding="utf-8")
            print(f"  Saved: {safe_console(str(output_path))}")
            # raise NotImplementedError("Implement convert_news_articles")


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print("\nDone! Output tai:", safe_console(str(OUTPUT_DIR)))


if __name__ == "__main__":
    convert_all()
