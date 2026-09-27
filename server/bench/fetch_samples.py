"""Download a small set of comic pages into server/samples/ for local testing.

    python -m bench.fetch_samples              # 5 pages per chapter listed in samples/urls.json
    python -m bench.fetch_samples --pages 8

mycomic.com's chapter pages sit behind Cloudflare bot protection, so this script
does not scrape them. Instead, collect the page-image URLs while viewing a chapter
normally in the browser, and put them in samples/urls.json:

    {"<series id>": {"<chapter id>": ["<page 1 url>", "<page 2 url>", ...]}}

To get a chapter's list, open the chapter and run this in the DevTools console
(it copies the JSON array to the clipboard):

    copy(JSON.stringify([...document.querySelectorAll('img.page')].map(i => i.dataset.src || i.src)))

The image host (biccam.com) only needs the site's Referer, which this script sends.
Pages are spread across each chapter so the set covers different layouts. Existing
files are skipped. Also writes samples/manifest.json and, if missing, a
samples/truth.json skeleton to fill in by hand (correct bubble text, in reading
order) for the OCR benchmark.

samples/ is gitignored: these pages are for local testing only.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx

SITE = "https://mycomic.com"
# The image host returns 403 without this Referer.
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/128 Safari/537.36",
    "Referer": f"{SITE}/",
}
SAMPLES = Path(__file__).resolve().parent.parent / "samples"
TRUTH_PAGES = 5  # how many pages get a truth.json entry to fill in


def spread(items: list, n: int) -> list:
    """n items evenly spaced across the list (all of them if n >= len)."""
    if n >= len(items):
        return list(items)
    step = (len(items) - 1) / max(n - 1, 1)
    return [items[round(i * step)] for i in range(n)]


def write_truth_skeleton(manifest: dict, pages_per_chapter: int) -> None:
    truth_path = SAMPLES / "truth.json"
    if truth_path.exists():
        return
    # Take one page per chapter in turn, so the truth set covers several chapters.
    # Skip each chapter's first and last picked page: those are usually title or
    # credits pages with little dialogue.
    by_chapter: dict[str, list[str]] = {}
    for rel in sorted(manifest):
        by_chapter.setdefault(manifest[rel]["chapter"], []).append(rel)
    order = [rels[i] for i in range(1, pages_per_chapter - 1) for rels in by_chapter.values() if i < len(rels) - 1]
    truth = {
        "_format": "page path -> list of bubbles, each the exact Chinese text in reading order",
        "pages": {rel: [] for rel in order[:TRUTH_PAGES]},
    }
    truth_path.write_text(json.dumps(truth, indent=2, ensure_ascii=False), "utf-8")
    print(f"wrote samples/truth.json skeleton ({TRUTH_PAGES} pages to fill in)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pages", type=int, default=5, help="pages per chapter")
    ap.add_argument("--delay", type=float, default=0.5, help="seconds between downloads")
    args = ap.parse_args()

    urls_path = SAMPLES / "urls.json"
    if not urls_path.exists():
        raise SystemExit(f"Missing {urls_path}. See this script's docstring for how to collect it.")
    urls: dict[str, dict[str, list[str]]] = json.loads(urls_path.read_text("utf-8"))

    manifest_path = SAMPLES / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8")) if manifest_path.exists() else {}

    with httpx.Client(headers=HEADERS, timeout=30, follow_redirects=True) as client:
        for series, chapters in urls.items():
            for chapter, page_urls in chapters.items():
                numbered = list(enumerate(page_urls, start=1))
                for num, url in spread(numbered, args.pages):
                    rel = f"{series}/{chapter}/{num:03d}{Path(url).suffix}"
                    dest = SAMPLES / rel
                    if dest.exists():
                        print(f"skip  {rel}")
                        continue
                    resp = client.get(url)
                    resp.raise_for_status()
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(resp.content)
                    manifest[rel] = {"url": url, "series": series, "chapter": chapter, "page": num}
                    print(f"saved {rel} ({len(resp.content) // 1024} KB)")
                    time.sleep(args.delay)

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), "utf-8")
    write_truth_skeleton(manifest, args.pages)
    print(f"{len(manifest)} pages in {SAMPLES}")


if __name__ == "__main__":
    main()
