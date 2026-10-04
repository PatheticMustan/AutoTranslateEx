"""OCR every corpus page once (with the bubble detector, as the server does) and
save the bubbles to samples/corpus/ocr.json:

    {"<series>/<chapter>/<NNN>.jpg": [{"text", "box", "vertical", "frame"}, ...]}

    python -m bench.corpus_ocr [--device dml]

Pages already in ocr.json are skipped, so it can be stopped and resumed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from atx.ocr import Ocr
from atx.pipeline import load_detector, ocr_page
from bench.fetch_corpus import CORPUS

OUT = CORPUS / "ocr.json"


def load() -> dict[str, list[dict]]:
    return json.loads(OUT.read_text("utf-8")) if OUT.exists() else {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu", choices=["cpu", "dml"])
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    pages = sorted(json.loads((CORPUS / "manifest.json").read_text("utf-8")))
    done = load()
    ocr, det = Ocr(device=args.device), load_detector()
    t0, n = time.perf_counter(), 0
    for rel in pages:
        if rel in done:
            continue
        bubbles, _ = ocr_page(ocr, CORPUS / rel, det)
        done[rel] = [{"text": b.text, "box": list(b.box), "vertical": b.vertical,
                      "frame": list(b.frame) if b.frame else None} for b in bubbles]
        n += 1
        if n % 50 == 0:
            OUT.write_text(json.dumps(done, ensure_ascii=False), "utf-8")
            print(f"{len(done)}/{len(pages)} pages, {(time.perf_counter() - t0) / n * 1000:.0f} ms/page", flush=True)
    OUT.write_text(json.dumps(done, ensure_ascii=False), "utf-8")
    print(f"done: {len(done)} pages, {sum(len(v) for v in done.values())} bubbles")


if __name__ == "__main__":
    main()
