"""Page image -> speech bubbles. Phase 3 adds translation on top of this.

Debug CLI:
    python -m atx.pipeline samples/20001/229697/008.jpg --debug out.png [--models v6-tiny] [--device dml]

Prints each bubble (index, box, direction, text) and, with --debug, saves a copy
of the page with line boxes (thin) and numbered bubble boxes (thick).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from atx.grouping import Bubble, group_lines
from atx.ocr import DEFAULT_DET_SIDE, DEFAULT_MODEL_SET, MODEL_SETS, Ocr, OcrResult


def ocr_page(ocr: Ocr, image) -> tuple[list[Bubble], OcrResult]:
    result = ocr(image)
    return group_lines(result.lines), result


def draw_debug(image_path: Path, bubbles: list[Bubble], out: Path) -> None:
    from PIL import Image, ImageDraw

    img = Image.open(image_path).convert("RGB")
    d = ImageDraw.Draw(img)
    for i, b in enumerate(bubbles, 1):
        for line in b.lines:
            d.rectangle(line.box, outline=(0, 160, 255), width=1)
        x0, y0, x1, y1 = b.box
        color = (230, 40, 40) if b.vertical else (40, 170, 40)
        d.rectangle((x0 - 3, y0 - 3, x1 + 3, y1 + 3), outline=color, width=3)
        d.rectangle((x0 - 3, y0 - 22, x0 + 22, y0 - 3), fill=color)
        d.text((x0 + 2, y0 - 20), str(i), fill=(255, 255, 255), font_size=16)
    img.save(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", type=Path)
    ap.add_argument("--models", default=DEFAULT_MODEL_SET, choices=list(MODEL_SETS))
    ap.add_argument("--device", default="cpu", choices=["cpu", "dml"])
    ap.add_argument("--det-side", type=int, default=DEFAULT_DET_SIDE,
                    help="detection input: longest side in px (0 = RapidOCR default)")
    ap.add_argument("--debug", type=Path, help="save an annotated copy of the page here")
    ap.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252

    ocr = Ocr(args.models, args.device, det_side=args.det_side)
    bubbles, result = ocr_page(ocr, args.image)

    if args.json:
        print(json.dumps([{"box": b.box, "vertical": b.vertical, "text": b.text} for b in bubbles],
                         ensure_ascii=False, indent=2))
    else:
        t = result.seconds
        print(f"{len(result.lines)} lines -> {len(bubbles)} bubbles  "
              f"(det {t['det'] * 1000:.0f} ms, rec {t['rec'] * 1000:.0f} ms, total {t['total'] * 1000:.0f} ms)")
        for i, b in enumerate(bubbles, 1):
            print(f"{i:2d} {'V' if b.vertical else 'H'} {b.box}  {b.text}")
    if args.debug:
        draw_debug(args.image, bubbles, args.debug)
        print(f"saved {args.debug}")


if __name__ == "__main__":
    main()
