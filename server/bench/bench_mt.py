"""Compare translation tiers on the OCR'd sample pages.

    python -m bench.bench_mt                         # all tiers
    python -m bench.bench_mt --tiers very_quick quick

Bubbles come from the default OCR (cached in bench/out/mt_input.json). Pages are
translated in reading order, so the accurate tiers get the previous page of the
same chapter as context. Each tier: load, one warm-up page, then every page.

Writes bench/out/mt_report.html (summary + every page's image with source and
all translations side by side, for judging quality by eye) and
bench/out/mt_<timestamp>.json. Both stay local (bench/out is gitignored).
"""

from __future__ import annotations

import argparse
import html
import json
import statistics
import sys
import time
from pathlib import Path

import psutil

from atx import device as devmod
from atx import translators

SERVER = Path(__file__).resolve().parent.parent
SAMPLES = SERVER / "samples"
OUT = SERVER / "bench" / "out"
INPUT = OUT / "mt_input.json"

# name -> (tier, parallel override); quick-serial shows what batching buys.
CONFIGS = {
    "very_quick": ("very_quick", None),
    "quick": ("quick", None),
    "quick-serial": ("quick", 1),
    "accurate-2b": ("accurate-2b", None),
    "accurate-4b": ("accurate", None),
}


def load_pages() -> dict[str, list[str]]:
    """page -> bubble texts, in chapter/page order; pages without text are skipped."""
    if INPUT.exists():
        return json.loads(INPUT.read_text("utf-8"))
    from atx.ocr import Ocr
    from atx.pipeline import ocr_page

    ocr = Ocr()
    pages = {}
    for path in sorted(SAMPLES.rglob("*.jpg")):
        bubbles, _ = ocr_page(ocr, path)
        if bubbles:
            pages[str(path.relative_to(SAMPLES)).replace("\\", "/")] = [b.text for b in bubbles]
    OUT.mkdir(parents=True, exist_ok=True)
    INPUT.write_text(json.dumps(pages, ensure_ascii=False, indent=1), "utf-8")
    return pages


def chapter(rel: str) -> str:
    return rel.rsplit("/", 1)[0]


def run_config(name: str, pages: dict[str, list[str]]) -> dict:
    tier, parallel = CONFIGS[name]
    rss_before = psutil.Process().memory_info().rss
    t0 = time.perf_counter()
    tr = translators.make(tier, parallel=parallel)
    load_s = time.perf_counter() - t0
    try:
        first = next(iter(pages.values()))
        tr.translate(first)  # warm-up

        out, prev = {}, {}
        for rel, texts in pages.items():
            context = prev.get(chapter(rel))
            t = time.perf_counter()
            translated = tr.translate(texts, context)
            ms = (time.perf_counter() - t) * 1000
            usage = getattr(tr, "last_usage", [])
            out[rel] = {"ms": ms, "dst": translated,
                        "gen_tps": [u["gen_tps"] for u in usage if u.get("gen_tps")],
                        "requests": len(usage)}
            prev[chapter(rel)] = texts

        server = getattr(tr, "server", None)
        if server:  # memory of the llama-server process; weights on a GPU sit in VRAM
            mem = psutil.Process(server.pid).memory_info()
            ram_mb = getattr(mem, "peak_wset", mem.rss) / 2**20
        else:
            ram_mb = (psutil.Process().memory_info().rss - rss_before) / 2**20
        return {"name": name, "tier": tier, "device": tr.device,
                "where": server.offload if server else tr.device,
                "load_s": load_s, "ram_mb": ram_mb, "pages": out}
    finally:
        tr.close()


def summarize(r: dict, pages: dict[str, list[str]]) -> dict:
    ms = [p["ms"] for p in r["pages"].values()]
    n_bubbles = sum(len(t) for t in pages.values())
    tps = [x for p in r["pages"].values() for x in p["gen_tps"]]
    return {
        "name": r["name"], "where": r["where"],
        "ms": statistics.mean(ms), "p90": sorted(ms)[max(0, round(0.9 * len(ms)) - 1)],
        "ms_per_bubble": sum(ms) / n_bubbles,
        "gen_tps": statistics.mean(tps) if tps else None,
        "load_s": r["load_s"], "ram_mb": r["ram_mb"],
    }


def fmt_tps(tps: float | None) -> str:
    return "" if tps is None else f"{tps:.0f}"


def write_report(results: list[dict], rows: list[dict], pages: dict[str, list[str]]) -> Path:
    e = html.escape
    names = [r["name"] for r in results]
    summary = "".join(
        f"<tr><td>{e(s['name'])}</td><td>{e(s['where'])}</td><td>{s['ms']:.0f}</td><td>{s['p90']:.0f}</td>"
        f"<td>{s['ms_per_bubble']:.0f}</td><td>{fmt_tps(s['gen_tps'])}</td>"
        f"<td>{s['load_s']:.1f}</td><td>{s['ram_mb']:.0f}</td></tr>"
        for s in rows
    )
    body = []
    for rel, texts in pages.items():
        head = "".join(f"<th>{e(n)}<br><small>{r['pages'][rel]['ms']:.0f} ms</small></th>"
                       for n, r in zip(names, results))
        trs = "".join(
            f"<tr><td class=n>{i + 1}</td><td class=src>{e(src)}</td>"
            + "".join(f"<td>{e(r['pages'][rel]['dst'][i])}</td>" for r in results) + "</tr>"
            for i, src in enumerate(texts)
        )
        body.append(
            f"<section><h2>{e(rel)}</h2><div class=page><a href='../../samples/{e(rel)}'>"
            f"<img src='../../samples/{e(rel)}' loading=lazy></a>"
            f"<table><tr><th>#</th><th>source</th>{head}</tr>{trs}</table></div></section>"
        )
    doc = f"""<!doctype html><meta charset=utf-8><title>Translation tiers</title>
<style>
:root {{ --bg:#fff; --fg:#1d1d1f; --line:#d9d9de; --muted:#6e6e73; --head:#f4f4f6; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#161618; --fg:#ececf0; --line:#38383d; --muted:#a1a1a6; --head:#222226; }} }}
body {{ font: 14px/1.45 system-ui, sans-serif; background: var(--bg); color: var(--fg); margin: 24px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid var(--line); padding: 6px 8px; vertical-align: top; text-align: left; }}
th {{ background: var(--head); font-weight: 600; }}
small, .n {{ color: var(--muted); }}
.src {{ white-space: nowrap; }}
.page {{ display: grid; grid-template-columns: 220px 1fr; gap: 16px; align-items: start; }}
.page img {{ width: 220px; border: 1px solid var(--line); }}
h2 {{ font-size: 15px; margin: 32px 0 8px; }}
@media (max-width: 800px) {{ .page {{ grid-template-columns: 1fr; }} }}
</style>
<h1>Translation tiers</h1>
<p>{len(pages)} pages, {sum(len(t) for t in pages.values())} bubbles. Device summary: {e(devmod.detect().summary())}</p>
<table><tr><th>config</th><th>ran on</th><th>ms/page</th><th>p90</th><th>ms/bubble</th><th>gen tok/s</th><th>load s</th><th>RAM MB</th></tr>{summary}</table>
{''.join(body)}"""
    path = OUT / "mt_report.html"
    path.write_text(doc, "utf-8")
    return path


def table(rows: list[dict]) -> str:
    head = "| config | ran on | ms/page | p90 | ms/bubble | gen tok/s | load s | RAM MB |\n|---|---|---|---|---|---|---|---|\n"
    return head + "\n".join(
        f"| {r['name']} | {r['where']} | {r['ms']:.0f} | {r['p90']:.0f} | {r['ms_per_bubble']:.0f} | "
        f"{fmt_tps(r['gen_tps'])} | {r['load_s']:.1f} | {r['ram_mb']:.0f} |"
        for r in rows
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiers", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    pages = load_pages()
    print(f"{len(pages)} pages, {sum(len(t) for t in pages.values())} bubbles")
    results, rows = [], []
    for name in args.tiers:
        print(f"running {name} ...", flush=True)
        r = run_config(name, pages)
        results.append(r)
        rows.append(summarize(r, pages))
        s = rows[-1]
        print(f"  {s['where']}: {s['ms']:.0f} ms/page (p90 {s['p90']:.0f}), load {s['load_s']:.1f}s", flush=True)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    (OUT / f"mt_{stamp}.json").write_text(
        json.dumps({"summary": rows, "results": results}, ensure_ascii=False, indent=1), "utf-8")
    (OUT / "mt_latest.md").write_text(table(rows) + "\n", "utf-8")
    report = write_report(results, rows, pages)
    print("\n" + table(rows))
    print(f"\nreport: {report}")


if __name__ == "__main__":
    main()
