"""Task 2 - Crawl news articles from configured source URLs."""

import json
import re
import gzip
import unicodedata
import zlib
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data" / "landing" / "news"
SOURCE_MANIFEST = PROJECT_DIR / "data" / "sources" / "news_articles.json"


class ArticleHTMLParser(HTMLParser):
    """Small article-oriented HTML text extractor using only stdlib."""

    def __init__(self) -> None:
        super().__init__()
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self._tag_stack: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        self._tag_stack.append(tag)
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1
        if self._tag_stack:
            self._tag_stack.pop()

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if not text or self._skip_depth:
            return
        current_tag = self._tag_stack[-1] if self._tag_stack else ""
        if current_tag == "title":
            self.title_parts.append(text)
        if current_tag in {"h1", "h2", "p", "li"} and len(text) > 20:
            self.text_parts.append(text)

    @property
    def title(self) -> str:
        return " ".join(self.title_parts).strip()

    @property
    def markdown(self) -> str:
        paragraphs = []
        seen = set()
        for part in self.text_parts:
            key = part.lower()
            if key in seen:
                continue
            seen.add(key)
            paragraphs.append(part)
        return "\n\n".join(paragraphs)


def setup_directory() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_article_sources(manifest_path: Path = SOURCE_MANIFEST) -> list[dict]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    articles = data.get("articles", [])
    if not articles:
        raise ValueError(f"No articles configured in {manifest_path}")
    return articles


def slugify(text: str, fallback: str) -> str:
    text = text.replace("Đ", "D").replace("đ", "d")
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_text.lower()).strip("-")
    return slug[:80] or fallback


def fetch_url(url: str) -> str:
    request = Request(
        url,
        headers={
            "User-Agent": "Day08-RAG-Pipeline/1.0",
            "Accept-Encoding": "gzip, deflate",
        },
    )
    with urlopen(request, timeout=60) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        raw = response.read()
        encoding = response.headers.get("Content-Encoding", "").lower()
        if "gzip" in encoding or raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        elif "deflate" in encoding:
            raw = zlib.decompress(raw)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode(charset, errors="replace")


def crawl_article(source: dict) -> dict:
    html = fetch_url(source["url"])
    parser = ArticleHTMLParser()
    parser.feed(html)
    title = source.get("title") or parser.title or source["url"]
    content = parser.markdown
    if len(content) < 500:
        content = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()
    return {
        "url": source["url"],
        "title": title,
        "date_crawled": datetime.now(timezone.utc).isoformat(),
        "source_name": source.get("source_name", ""),
        "content_markdown": content,
    }


def crawl_all() -> list[dict]:
    setup_directory()
    results = []
    for index, source in enumerate(load_article_sources(), 1):
        article = crawl_article(source)
        source_slug = slugify(article.get("source_name", ""), f"source-{index}")
        title_slug = slugify(article["title"], f"article-{index}")
        filename = f"{index:02d}-{source_slug}-{title_slug}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved {filepath}")
        results.append(article)
    return results


if __name__ == "__main__":
    crawl_all()
