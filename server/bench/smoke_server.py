"""Drive a running server (python -m atx.app) the way the extension will.

    python -m bench.smoke_server --tier quick [--concurrency 2] [--chapters 229697]

Sends each sample chapter's pages in order, `concurrency` at a time, each naming
the previous page (prev_url) so the accurate tier gets context. Prints per-page
server timings and a summary; run it twice to see the cache.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def chapters(only: list[str] | None) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for path in sorted(SAMPLES.rglob("*.jpg")):
        ch = f"{path.parent.parent.name}/{path.parent.name}"
        if not only or path.parent.name in only:
            out.setdefault(ch, []).append(path)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--tier", default="quick")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--chapters", nargs="*")
    ap.add_argument("--show", action="store_true", help="print every translation")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    client = httpx.Client(base_url=args.url, timeout=300)
    t = time.perf_counter()
    warm = client.post("/warm", data={"tier": args.tier}).json()
    print(f"warm {args.tier} -> {warm['resolved']}: {warm['load_ms']} ms load, "
          f"{(time.perf_counter() - t) * 1000:.0f} ms round trip; llm: {warm['components']['llm']}")

    def send(path: Path, prev: Path | None) -> tuple[Path, dict, float]:
        t0 = time.perf_counter()
        r = client.post("/translate", files={"image": (path.name, path.read_bytes(), "image/jpeg")},
                        data={"tier": args.tier, "page_url": path.as_posix(),
                              "prev_url": prev.as_posix() if prev else ""})
        r.raise_for_status()
        return path, r.json(), (time.perf_counter() - t0) * 1000

    wall, rows = time.perf_counter(), []
    with ThreadPoolExecutor(args.concurrency) as pool:
        for ch, pages in chapters(args.chapters).items():
            jobs = [pool.submit(send, p, pages[i - 1] if i else None) for i, p in enumerate(pages)]
            for job in jobs:
                path, res, rtt = job.result()
                rows.append((res, rtt))
                ms = res["ms"]
                print(f"{ch}/{path.name}: {len(res['regions']):2d} bubbles  "
                      f"{'cached' if res['cached'] else 'ocr ' + str(ms.get('ocr')) + ' translate ' + str(ms.get('translate'))}"
                      f"  server {ms['total']} ms, round trip {rtt:.0f} ms")
                if args.show:
                    for reg in res["regions"]:
                        print(f"    {reg['src']}\n      -> {reg['dst']}")
    wall = time.perf_counter() - wall
    rtts = [r for _, r in rows]
    print(f"\n{len(rows)} pages in {wall:.1f}s ({wall / len(rows) * 1000:.0f} ms/page with {args.concurrency} "
          f"in flight); round trip mean {statistics.mean(rtts):.0f} ms, max {max(rtts):.0f} ms")


if __name__ == "__main__":
    main()
