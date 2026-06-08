"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.
"""

import json
import re
from pathlib import Path

from markitdown import MarkItDown

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR  = Path(__file__).parent.parent / "data" / "standardized"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_markdown(text: str) -> str:
    """
    Loại bỏ các phần rác thường xuất hiện sau khi crawl:
      - Các dòng chỉ chứa link điều hướng / menu
      - Nhiều dòng trắng liên tiếp
      - Các dòng quảng cáo lặp lại
    """
    lines = text.splitlines()
    cleaned = []
    nav_keywords = [
        "Liên hệ quảng cáo", "Showbiz", "Thời cuộc", "Hoàng gia",
        "Chuyện lạ", "Hình sự", "Làm đẹp", "Lối sống", "Thời trang",
        "Ăn chơi", "Trắc nghiệm", "Podcasts", "Cung Hoàng Đạo",
        "Thương trường", "Du lịch", "Ẩm thực", "Trang chủ",
        "Đăng nhập", "Đăng ký", "Tìm kiếm", "Quảng cáo",
        "utm_source", "utm_medium", "utm_campaign",
    ]

    blank_count = 0
    for line in lines:
        stripped = line.strip()

        # Bỏ dòng trắng liên tiếp (giữ tối đa 1)
        if not stripped:
            blank_count += 1
            if blank_count <= 1:
                cleaned.append("")
            continue
        blank_count = 0

        # Bỏ dòng chứa từ khoá điều hướng / quảng cáo
        if any(kw in stripped for kw in nav_keywords):
            continue

        # Bỏ dòng chỉ toàn link markdown kiểu [text](url) không có nội dung thực
        # nhưng GIỮ lại nếu dòng có nội dung văn bản thực sự
        only_links = re.fullmatch(r"(\[.*?\]\(.*?\)\s*)+", stripped)
        if only_links:
            continue

        cleaned.append(line)

    return "\n".join(cleaned).strip()


def _extract_main_content(markdown: str) -> str:
    """
    Cố gắng tìm phần nội dung chính của bài báo bằng cách:
    - Bỏ qua tất cả trước heading đầu tiên (thường là menu/nav)
    - Dừng lại trước phần 'Xem thêm' / 'Bình luận' / footer
    """
    lines = markdown.splitlines()

    # Tìm vị trí heading đầu tiên (# ...) — thường là tiêu đề bài báo
    start_idx = 0
    for i, line in enumerate(lines):
        if re.match(r"^#{1,3}\s+\S", line.strip()):
            start_idx = i
            break

    # Tìm vị trí bắt đầu phần footer/comment để cắt bỏ
    stop_keywords = [
        "Bình luận", "Xem thêm", "Tin liên quan",
        "Có thể bạn quan tâm", "Tags:", "© 20",
    ]
    stop_idx = len(lines)
    for i in range(start_idx + 5, len(lines)):  # bỏ qua 5 dòng đầu sau heading
        stripped = lines[i].strip()
        if any(stripped.startswith(kw) for kw in stop_keywords):
            stop_idx = i
            break

    main_lines = lines[start_idx:stop_idx]
    return "\n".join(main_lines).strip()


# ---------------------------------------------------------------------------
# Convert functions
# ---------------------------------------------------------------------------

def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not legal_dir.exists():
        print(f"  ⚠ Thư mục không tồn tại: {legal_dir}")
        return

    md = MarkItDown()
    files = [
        f for f in legal_dir.iterdir()
        if f.is_file() and f.suffix.lower() in (".pdf", ".docx", ".doc")
    ]

    if not files:
        print(f"  ⚠ Không có file PDF/DOCX trong: {legal_dir}")
        return

    success = 0
    for filepath in sorted(files):
        print(f"  Converting: {filepath.name}")
        try:
            result = md.convert(str(filepath))

            content = result.text_content or ""
            if not content.strip():
                print(f"  ⚠ Nội dung rỗng: {filepath.name}")
                continue

            # Thêm YAML frontmatter
            from datetime import datetime
            frontmatter = "\n".join([
                "---",
                f"source_file: {filepath.name}",
                f"converted_at: {datetime.now().isoformat()}",
                "category: legal",
                "---", "", "",
            ])

            output_path = output_dir / f"{filepath.stem}.md"
            output_path.write_text(frontmatter + content, encoding="utf-8")
            success += 1
            print(f"  ✓ Saved: {output_path}")

        except Exception as e:
            print(f"  ✗ Lỗi: {filepath.name} — {e}")

    print(f"\n  Legal: {success}/{len(files)} file thành công.")


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not news_dir.exists():
        print(f"  ⚠ Thư mục không tồn tại: {news_dir}")
        return

    files = [
        f for f in news_dir.iterdir()
        if f.is_file() and f.suffix.lower() == ".json"
    ]

    if not files:
        print(f"  ⚠ Không có file JSON trong: {news_dir}")
        return

    success = 0
    for filepath in sorted(files):
        print(f"  Converting: {filepath.name}")
        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))

            # Bỏ qua bài bị lỗi crawl
            if data.get("title") == "CRAWL_FAILED":
                print(f"  ⚠ Bỏ qua (crawl failed): {filepath.name}")
                continue

            raw_content = data.get("content_markdown", "")
            if not raw_content.strip():
                print(f"  ⚠ Nội dung rỗng: {filepath.name}")
                continue

            main_content = _extract_main_content(raw_content)
            main_content = _clean_markdown(main_content)

            # Header metadata
            title   = data.get("title", "Unknown").strip()
            url     = data.get("url", "N/A")
            crawled = data.get("date_crawled", "N/A")

            header = "\n".join([
                f"# {title}",
                "",
                f"**Source:** {url}",
                f"**Crawled:** {crawled}",
                "",
                "---",
                "", "",
            ])

            output_path = output_dir / f"{filepath.stem}.md"
            output_path.write_text(header + main_content, encoding="utf-8")
            success += 1
            print(f"  ✓ Saved : {output_path}")
            print(f"  ✓ Title : {title}")

        except json.JSONDecodeError as e:
            print(f"  ✗ JSON lỗi {filepath.name}: {e}")
        except Exception as e:
            print(f"  ✗ Lỗi: {filepath.name} — {e}")

    print(f"\n  News: {success}/{len(files)} file thành công.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def convert_all():
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print(f"\n✓ Done! Output tại: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    convert_all()