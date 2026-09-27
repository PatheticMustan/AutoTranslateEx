"""Pre-fill empty pages in samples/truth.json with the best local OCR, for review.

    python -m bench.prefill_truth [--models v6-medium]

Writes each page's bubbles (reading order) into truth.json and an annotated copy
of the page to bench/out/truth/<page>.png with numbered bubbles, so the text can
be checked against the image and corrected. Pages that already have bubbles are
left alone. Truth files are then verified with the highest-quality model
available on the dev machine (see PLAN.md, Phase 0).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from atx.ocr import MODEL_SETS, Ocr
from atx.pipeline import draw_debug, ocr_page

SERVER = Path(__file__).resolve().parent.parent
SAMPLES = SERVER / "samples"
OUT = SERVER / "bench" / "out" / "truth"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="v6-medium", choices=list(MODEL_SETS))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    truth_path = SAMPLES / "truth.json"
    truth = json.loads(truth_path.read_text("utf-8"))
    ocr = Ocr(args.models, det_side=0)  # full resolution: accuracy over speed here
    OUT.mkdir(parents=True, exist_ok=True)

    for rel, bubbles in truth["pages"].items():
        if bubbles:
            print(f"skip  {rel} ({len(bubbles)} bubbles already)")
            continue
        found, _ = ocr_page(ocr, SAMPLES / rel)
        truth["pages"][rel] = [b.text for b in found]
        debug = OUT / (rel.replace("/", "_").rsplit(".", 1)[0] + ".png")
        draw_debug(SAMPLES / rel, found, debug)
        print(f"fill  {rel}: {len(found)} bubbles -> {debug.relative_to(SERVER)}")

    truth_path.write_text(json.dumps(truth, indent=2, ensure_ascii=False), "utf-8")


if __name__ == "__main__":
    main()
