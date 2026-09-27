"""Benchmark OCR model sets for accuracy (vs samples/truth.json), speed and memory.

    python -m bench.bench_ocr                          # default grid
    python -m bench.bench_ocr --models v6-tiny v6-small --sides 720 960 --devices cpu dml

Each configuration (model set x detection size x device) runs in its own
subprocess over all sample pages: 2 warm-up runs, then the median of --reps runs
per page. Reports:
  CER      character error rate on the truth pages, ignoring punctuation and
           whitespace; missed bubbles count as fully wrong
  recall   truth bubbles matched by a found bubble (normalized distance <= 0.5)
  extra    found regions that match no truth bubble (watermarks, stray marks)
  ms/page  mean over all sample pages of the per-page median; p90 across pages
  load s   model load time;  peak MB  peak resident memory of the worker

Results go to bench/out/ocr_<timestamp>.json (per-page details) and
bench/out/ocr_latest.md (the table).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

SERVER = Path(__file__).resolve().parent.parent
SAMPLES = SERVER / "samples"
OUT = SERVER / "bench" / "out"

DEFAULT_GRID = (
    # (model set, detection side, device). v6-medium is left out: ~30-50 s per page
    # on CPU, far outside any budget (it's still available for prefill_truth).
    [(m, 960, "cpu") for m in ["v6-tiny", "v6-small-tiny", "v6-small", "v5-mobile", "v5-server", "cht-v3"]]
    + [(m, s, "cpu") for m in ["v6-tiny", "v6-small"] for s in (0, 720)]
    + [(m, 960, "dml") for m in ["v6-tiny", "v6-small"]]
)
MATCH_THRESHOLD = 0.5


# ---------- accuracy ----------

def normalize(text: str) -> str:
    return re.sub(r"[\W_]", "", text)


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def score_page(truth: list[str], found: list[str]) -> dict:
    """Greedily match truth bubbles to found bubbles by normalized edit distance."""
    t = [normalize(s) for s in truth]
    f = [normalize(s) for s in found]
    pairs = sorted(
        (edit_distance(ti, fj) / max(len(ti), 1), i, j)
        for i, ti in enumerate(t) for j, fj in enumerate(f)
    )
    matched_t, matched_f, errors = set(), set(), 0
    for dist, i, j in pairs:
        if dist > MATCH_THRESHOLD or i in matched_t or j in matched_f:
            continue
        matched_t.add(i)
        matched_f.add(j)
        errors += edit_distance(t[i], f[j])
    errors += sum(len(t[i]) for i in range(len(t)) if i not in matched_t)
    return {
        "chars": sum(len(s) for s in t),
        "errors": errors,
        "truth_bubbles": len(t),
        "matched": len(matched_t),
        "extra": len(f) - len(matched_f),
    }


# ---------- worker (one configuration, in its own process) ----------

def peak_mb() -> float:
    import psutil
    mem = psutil.Process().memory_info()
    return getattr(mem, "peak_wset", mem.rss) / 2**20


def run_worker(model_set: str, side: int, device: str, reps: int) -> dict:
    from atx.grouping import group_lines
    from atx.ocr import Ocr

    pages = sorted(str(p.relative_to(SAMPLES)).replace("\\", "/") for p in SAMPLES.rglob("*.jpg"))
    t0 = time.perf_counter()
    ocr = Ocr(model_set, device, det_side=side)
    load_s = time.perf_counter() - t0

    first = SAMPLES / pages[0]
    for _ in range(2):
        ocr(first)

    per_page = {}
    for rel in pages:
        runs = [ocr(SAMPLES / rel) for _ in range(reps)]
        per_page[rel] = {
            "ms": statistics.median(r.seconds["total"] for r in runs) * 1000,
            "det_ms": statistics.median(r.seconds["det"] for r in runs) * 1000,
            "rec_ms": statistics.median(r.seconds["rec"] for r in runs) * 1000,
            "bubbles": [b.text for b in group_lines(runs[0].lines)],
        }
    return {"model_set": model_set, "side": side, "device": device,
            "load_s": load_s, "peak_mb": peak_mb(), "pages": per_page}


# ---------- driver ----------

def summarize(result: dict, truth: dict[str, list[str]]) -> dict:
    ms = [p["ms"] for p in result["pages"].values()]
    scores = [score_page(truth[rel], result["pages"][rel]["bubbles"]) for rel in truth]
    chars = sum(s["chars"] for s in scores)
    return {
        "config": f"{result['model_set']} @{result['side'] or 'default'} {result['device']}",
        "cer": sum(s["errors"] for s in scores) / chars,
        "recall": sum(s["matched"] for s in scores) / sum(s["truth_bubbles"] for s in scores),
        "extra": sum(s["extra"] for s in scores),
        "ms": statistics.mean(ms),
        "p90": sorted(ms)[max(0, round(0.9 * len(ms)) - 1)],
        "det_ms": statistics.mean(p["det_ms"] for p in result["pages"].values()),
        "load_s": result["load_s"],
        "peak_mb": result["peak_mb"],
    }


def table(rows: list[dict]) -> str:
    head = "| config | CER | recall | extra | ms/page | p90 | det ms | load s | peak MB |\n|---|---|---|---|---|---|---|---|---|\n"
    return head + "\n".join(
        f"| {r['config']} | {r['cer']:.1%} | {r['recall']:.0%} | {r['extra']} | {r['ms']:.0f} | "
        f"{r['p90']:.0f} | {r['det_ms']:.0f} | {r['load_s']:.1f} | {r['peak_mb']:.0f} |"
        for r in rows
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--sides", nargs="+", type=int, default=[960])
    ap.add_argument("--devices", nargs="+", default=["cpu"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--worker", nargs=3, metavar=("MODELS", "SIDE", "DEVICE"), help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        m, side, dev = args.worker
        print(json.dumps(run_worker(m, int(side), dev, args.reps)))
        return

    sys.stdout.reconfigure(encoding="utf-8")
    truth = json.loads((SAMPLES / "truth.json").read_text("utf-8"))["pages"]
    grid = ([(m, s, d) for m in args.models for s in args.sides for d in args.devices]
            if args.models else DEFAULT_GRID)

    results, rows = [], []
    for m, side, dev in grid:
        print(f"running {m} @{side or 'default'} {dev} ...", flush=True)
        proc = subprocess.run(
            [sys.executable, "-m", "bench.bench_ocr", "--worker", m, str(side), dev, "--reps", str(args.reps)],
            cwd=SERVER, capture_output=True, text=True, encoding="utf-8",
        )
        if proc.returncode != 0:
            print(f"  failed:\n{proc.stderr[-2000:]}")
            continue
        result = json.loads(proc.stdout.strip().splitlines()[-1])
        results.append(result)
        rows.append(summarize(result, truth))
        r = rows[-1]
        print(f"  CER {r['cer']:.1%}  recall {r['recall']:.0%}  {r['ms']:.0f} ms/page  {r['peak_mb']:.0f} MB")

    rows.sort(key=lambda r: (round(r["cer"], 3), r["ms"]))
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (OUT / f"ocr_{stamp}.json").write_text(json.dumps({"summary": rows, "results": results}, ensure_ascii=False, indent=1), "utf-8")
    md = table(rows)
    (OUT / "ocr_latest.md").write_text(md + "\n", "utf-8")
    print("\n" + md)


if __name__ == "__main__":
    main()
