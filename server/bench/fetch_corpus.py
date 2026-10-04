"""Download whole chapters into server/samples/corpus/ for testing at scale.

    python -m bench.fetch_corpus

Reads every chapter from samples/urls.json ({series: {chapter: [urls]}}) and
samples/corpus_urls_compact.json ({series: {chapter: "<file> <file> ..."}}, the
image file names without .jpg, which keeps a list collected in the browser
small). Saves pages as samples/corpus/<series>/<chapter>/<NNN>.jpg and writes
samples/corpus/manifest.json. Existing files are skipped. samples/ is
gitignored: local testing only.

To collect a chapter list in the browser, open the series page and fetch each
chapter's HTML from there (the browser has the site's session); the page images
are every https://biccam.com/chapters/<chapter>/<page>-<hash>.jpg in it.
"""

from __future__ import annotations

import argparse
import json
import time

import httpx

from bench.fetch_samples import HEADERS, SAMPLES

CORPUS = SAMPLES / "corpus"
IMAGE_HOST = "https://biccam.com/chapters"


def chapters() -> dict[str, dict[str, list[str]]]:
    out: dict[str, dict[str, list[str]]] = {}
    full = SAMPLES / "urls.json"
    if full.exists():
        for series, chs in json.loads(full.read_text("utf-8")).items():
            out.setdefault(series, {}).update(chs)
    compact = SAMPLES / "corpus_urls_compact.json"
    if compact.exists():
        for series, chs in json.loads(compact.read_text("utf-8")).items():
            for ch, files in chs.items():
                out.setdefault(series, {})[ch] = [f"{IMAGE_HOST}/{ch}/{f}.jpg" for f in files.split()]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--delay", type=float, default=0.2, help="seconds between downloads")
    args = ap.parse_args()

    manifest, failed, fetched = {}, [], 0
    with httpx.Client(headers=HEADERS, timeout=60, follow_redirects=True) as client:
        for series, chs in chapters().items():
            for ch, urls in chs.items():
                for i, url in enumerate(urls, 1):
                    rel = f"{series}/{ch}/{i:03d}.jpg"
                    dest = CORPUS / rel
                    if not dest.exists():
                        r = client.get(url)
                        if r.status_code != 200:
                            failed.append((rel, url, r.status_code))
                            continue
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        dest.write_bytes(r.content)
                        fetched += 1
                        time.sleep(args.delay)
                    manifest[rel] = {"url": url, "series": series, "chapter": ch, "page": i}
                print(f"{series}/{ch}: {len(urls)} pages", flush=True)
    CORPUS.mkdir(parents=True, exist_ok=True)
    (CORPUS / "manifest.json").write_text(json.dumps(manifest, indent=1), "utf-8")
    print(f"{len(manifest)} pages ({fetched} downloaded now), {len(failed)} failed")
    for f in failed:
        print("  failed:", *f)


if __name__ == "__main__":
    main()
