"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


ARTICLE_URLS = [
    "https://ngoisao.vnexpress.net/nhung-nghe-si-viet-nga-ngua-vi-ma-tuy-4816068.html",
    "https://ngoisao.vnexpress.net/anh-em-ca-si-chi-dan-ru-nhieu-nguoi-choi-ma-tuy-nhu-the-nao-4929875.html",
    "https://baobacninhtv.vn/dien-vien-hai-bi-tam-giu-vi-lien-quan-ma-tuy.bbg",
    "https://vtcnews.vn/dong-thai-cua-miu-le-sau-khi-bi-dieu-tra-lien-quan-ma-tuy-ar1017516.html",
    "https://vietnamnet.vn/nguoi-mau-an-tay-bi-giu-de-dieu-tra-lien-quan-tiec-ma-tuy-2340576.html",
]


def _extract_title_from_markdown(markdown: str) -> str:
    if not markdown:
        return ""
    for line in markdown.splitlines():
        line = line.strip().lstrip("#").strip()
        if line:
            return line
    return ""


async def crawl_article(url: str) -> dict:
    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler(headless=True, verbose=False) as crawler:
        result = await crawler.arun(
            url=url,
            wait_for="body",
            page_timeout=30000,
            word_count_threshold=10,
            excluded_tags=["nav", "footer", "header", "aside", "script", "style"],
            exclude_external_links=True,
        )

        if not result.success:
            print(f"  ✗ Failed: {url} — {result.error_message}")
            return {
                "url": url,
                "title": "CRAWL_FAILED",
                "date_crawled": datetime.now().isoformat(),
                "content_markdown": "",
                "error": result.error_message,
            }

        title = (
            result.metadata.get("og:title")
            or result.metadata.get("title")
            or _extract_title_from_markdown(result.markdown)
            or "Unknown"
        )

        return {
            "url": url,
            "title": title.strip(),
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": result.markdown,
        }


async def crawl_all():
    setup_directory()
    success_count = 0

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        try:
            article = await crawl_article(url)

            filename = f"article_{i:02d}.json"
            filepath = DATA_DIR / filename
            filepath.write_text(
                json.dumps(article, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            if article.get("title") != "CRAWL_FAILED":
                success_count += 1
                print(f"  ✓ Saved : {filepath}")
                print(f"  ✓ Title : {article['title']}")
            else:
                print(f"  ✗ Saved with error: {filepath}")

        except Exception as e:
            print(f"  ✗ Exception: {e}")

        await asyncio.sleep(1)

    print(f"\n{'='*50}")
    print(f"Hoàn thành: {success_count}/{len(ARTICLE_URLS)} bài crawl thành công.")
    print(f"Dữ liệu lưu tại: {DATA_DIR.resolve()}")


if __name__ == "__main__":
    # (Playwright cần subprocess, SelectorEventLoop không hỗ trợ)
    if sys.platform == "win32":
        loop = asyncio.ProactorEventLoop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(crawl_all())
        finally:
            loop.close()
    else:
        asyncio.run(crawl_all())