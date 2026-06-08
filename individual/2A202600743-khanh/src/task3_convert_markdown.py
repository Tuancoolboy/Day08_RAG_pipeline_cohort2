"""Task 3 - Convert landing files to Markdown."""

import json
import shutil
import subprocess
from pathlib import Path

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def _pdf_to_text(filepath: Path) -> str:
    pdftotext = shutil.which("pdftotext")
    if not pdftotext:
        return f"# {filepath.stem}\n\nPDF file saved at `{filepath}`.\n"

    result = subprocess.run(
        [pdftotext, "-layout", "-enc", "UTF-8", str(filepath), "-"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout


def convert_legal_docs() -> list[Path]:
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for filepath in sorted(legal_dir.iterdir()):
        if filepath.suffix.lower() not in {".pdf", ".docx", ".doc"}:
            continue
        text = _pdf_to_text(filepath) if filepath.suffix.lower() == ".pdf" else filepath.read_text(
            encoding="utf-8", errors="replace"
        )
        output_path = output_dir / f"{filepath.stem}.md"
        content = f"# {filepath.stem}\n\n**Source file:** {filepath.name}\n\n---\n\n{text.strip()}\n"
        output_path.write_text(content, encoding="utf-8")
        print(f"Saved {output_path}")
        written.append(output_path)
    return written


def convert_news_articles() -> list[Path]:
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for filepath in sorted(news_dir.glob("*.json")):
        data = json.loads(filepath.read_text(encoding="utf-8"))
        output_path = output_dir / f"{filepath.stem}.md"
        header = [
            f"# {data.get('title', filepath.stem)}",
            "",
            f"**Source:** {data.get('url', 'N/A')}",
            f"**Source name:** {data.get('source_name', 'N/A')}",
            f"**Crawled:** {data.get('date_crawled', 'N/A')}",
            "",
            "---",
            "",
        ]
        content = "\n".join(header) + data.get("content_markdown", "")
        output_path.write_text(content, encoding="utf-8")
        print(f"Saved {output_path}")
        written.append(output_path)
    return written


def convert_all() -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return convert_legal_docs() + convert_news_articles()


if __name__ == "__main__":
    convert_all()
