"""Compare quantizations of Hy-MT2-1.8B for a single-tier install.

    python -m atx.models hy-mt2-1.8b-q8_0 hy-mt2-1.8b-iq4_xs ...   # see atx.models.HY_MT_QUANTS
    python -m bench.bench_quant [--models ...] [--cpu]

Runs the accurate-hymt setup (Hy-MT2 with the preceding bubbles as context, 8
parallel slots) on the same OCR'd bubbles as bench_mt, with greedy decoding so
differences come from the quantization rather than sampling.

Quality, as chrF (character n-gram F-score, 0-100) on the 82 scored bubbles:
- vs ref: against samples/mt_truth.json, reference English written by Claude;
- vs Q8:  against the Q8_0 output, i.e. how far quantization moves the output
          away from the near-lossless model.
Plus failure counts: output mostly Chinese, empty, or far longer than the
reference (run-on). --cpu also times each model on the CPU build (-ngl 0).

Writes bench/out/quant_report.html (side by side), quant_<stamp>.json and quant_latest.md.
"""

from __future__ import annotations

import argparse
import html
import json
import statistics
import sys
import time
from collections import Counter

import psutil

from atx.llm_process import LlamaServer
from atx.models import HY_MT_QUANTS, gguf_path
from atx.translators.llm import HyMtTranslator, untranslated
from bench.bench_mt import OUT, SAMPLES, chapter, load_pages

GREEDY = {"temperature": 0.0, "top_k": 1, "repeat_penalty": 1.05}
DEFAULT_MODELS = ["hy-mt2-1.8b-q8_0", "hy-mt2-1.8b", *[m for m in HY_MT_QUANTS if m != "hy-mt2-1.8b-q8_0"]]
LABEL = {"hy-mt2-1.8b": "Q4_K_M (current)"} | {m: m.rsplit("-", 1)[1].upper() for m in HY_MT_QUANTS}
RUN_ON_RATIO = 2.5  # output this many times longer than the reference


# ---- chrF (Popović 2015), corpus level, like sacrebleu's default (n=6, beta=2, no word n-grams) ----

def _ngrams(text: str, n: int) -> Counter:
    s = "".join(text.lower().split())
    return Counter(s[i:i + n] for i in range(len(s) - n + 1))


def chrf(hyps: list[str], refs: list[str], order: int = 6, beta: float = 2.0) -> float:
    precisions, recalls = [], []
    for n in range(1, order + 1):
        match = hyp_total = ref_total = 0
        for h, r in zip(hyps, refs):
            hc, rc = _ngrams(h, n), _ngrams(r, n)
            match += sum((hc & rc).values())
            hyp_total += sum(hc.values())
            ref_total += sum(rc.values())
        if hyp_total and ref_total:
            precisions.append(match / hyp_total)
            recalls.append(match / ref_total)
    if not precisions:
        return 0.0
    p, r = statistics.mean(precisions), statistics.mean(recalls)
    return 0.0 if p + r == 0 else 100 * (1 + beta**2) * p * r / (beta**2 * p + r)


# ---- runs ----

def run(model: str, pages: dict[str, list[str]], cpu: bool) -> dict:
    server = LlamaServer(model, parallel=8, ctx_per_slot=1024)
    t0 = time.perf_counter()
    server.start(cpu=cpu)
    load_s = time.perf_counter() - t0
    tr = HyMtTranslator(server, use_context=True, sampling=GREEDY)
    try:
        tr.translate(next(iter(pages.values())))  # warm-up
        out, prev = {}, {}
        for rel, texts in pages.items():
            t = time.perf_counter()
            dst = tr.translate(texts, prev.get(chapter(rel)))
            out[rel] = {"ms": (time.perf_counter() - t) * 1000, "dst": dst,
                        "gen_tps": [u["gen_tps"] for u in tr.last_usage if u.get("gen_tps")]}
            prev[chapter(rel)] = texts
        mem = psutil.Process(server.pid).memory_info()
        return {"model": model, "where": server.offload, "device": server.device, "load_s": load_s,
                "ram_mb": getattr(mem, "peak_wset", mem.rss) / 2**20,
                "file_mb": gguf_path(model).stat().st_size / 2**20, "pages": out}
    finally:
        tr.close()
        server.stop()


def score(r: dict, truth: dict, q8: dict | None) -> dict:
    hyps, refs, vs_q8 = [], [], []
    fails = {"chinese": 0, "empty": 0, "run_on": 0}
    for rel, page in r["pages"].items():
        for i, (h, ref) in enumerate(zip(page["dst"], truth[rel])):
            if ref is None:
                continue
            hyps.append(h)
            refs.append(ref)
            if q8:
                vs_q8.append(q8["pages"][rel]["dst"][i])
            fails["chinese"] += untranslated(h)
            fails["empty"] += not h.strip()
            fails["run_on"] += len(h) > RUN_ON_RATIO * len(ref) + 20
    ms = [p["ms"] for p in r["pages"].values()]
    tps = [x for p in r["pages"].values() for x in p["gen_tps"]]
    return {
        "model": r["model"], "label": LABEL[r["model"]], "file_mb": r["file_mb"], "where": r["where"],
        "chrf_ref": chrf(hyps, refs), "chrf_q8": chrf(hyps, vs_q8) if q8 else None, **fails,
        "ms": statistics.mean(ms), "p90": sorted(ms)[max(0, round(0.9 * len(ms)) - 1)],
        "gen_tps": statistics.mean(tps) if tps else None, "load_s": r["load_s"], "ram_mb": r["ram_mb"],
    }


def _num(x, fmt=".1f") -> str:
    return "" if x is None else format(x, fmt)


def table(rows: list[dict], cpu_rows: dict[str, dict]) -> str:
    head = ("| quant | file MB | chrF vs ref | chrF vs Q8 | Chinese / empty / run-on | GPU ms/page (p90) "
            "| gen tok/s | llama-server peak RAM MB |" + (" CPU ms/page (p90) | CPU gen tok/s |" if cpu_rows else "")
            + "\n|" + "---|" * (8 + 2 * bool(cpu_rows)) + "\n")
    lines = []
    for r in rows:
        line = (f"| {r['label']} | {r['file_mb']:.0f} | {r['chrf_ref']:.1f} | {_num(r['chrf_q8'])} | "
                f"{r['chinese']} / {r['empty']} / {r['run_on']} | {r['ms']:.0f} ({r['p90']:.0f}) | "
                f"{_num(r['gen_tps'], '.0f')} | {r['ram_mb']:.0f} |")
        if cpu_rows:
            c = cpu_rows.get(r["model"])
            line += f" {c['ms']:.0f} ({c['p90']:.0f}) | {_num(c['gen_tps'], '.0f')} |" if c else " | |"
        lines.append(line)
    return head + "\n".join(lines)


def write_report(results: list[dict], rows: list[dict], cpu_rows: dict, pages: dict, truth: dict) -> None:
    e = html.escape
    body = []
    for rel, texts in pages.items():
        head = "".join(f"<th>{e(LABEL[r['model']])}</th>" for r in results)
        trs = []
        for i, src in enumerate(texts):
            ref = truth[rel][i]
            cells = []
            for r in results:
                h = r["pages"][rel]["dst"][i]
                s = chrf([h], [ref]) if ref else None
                cls = "" if s is None else ("bad" if s < 35 else "ok" if s < 60 else "good")
                cells.append(f"<td class={cls}>{e(h)}<small>{'' if s is None else f' {s:.0f}'}</small></td>")
            trs.append(f"<tr><td class=src>{e(src)}</td><td class=ref>{e(ref or '(not scored)')}</td>{''.join(cells)}</tr>")
        body.append(f"<section><h2>{e(rel)}</h2><div class=page><a href='../../samples/{e(rel)}'>"
                    f"<img src='../../samples/{e(rel)}' loading=lazy></a><table><tr><th>source</th>"
                    f"<th>reference</th>{head}</tr>{''.join(trs)}</table></div></section>")
    md = table(rows, cpu_rows)
    summary = "".join(f"<tr>{''.join(f'<td>{e(c.strip())}</td>' for c in line.strip('|').split('|'))}</tr>"
                      for line in md.splitlines()[2:])
    heads = "".join(f"<th>{e(c.strip())}</th>" for c in md.splitlines()[0].strip("|").split("|"))
    doc = f"""<!doctype html><meta charset=utf-8><title>Hy-MT2 quantizations</title>
<style>
:root {{ --bg:#fff; --fg:#1d1d1f; --line:#d9d9de; --muted:#6e6e73; --head:#f4f4f6; --bad:#fde2e1; --ok:#fff4d6; --good:#e3f5e6; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#161618; --fg:#ececf0; --line:#38383d; --muted:#a1a1a6; --head:#222226; --bad:#4a2323; --ok:#43391d; --good:#1f3a25; }} }}
body {{ font: 14px/1.45 system-ui, sans-serif; background: var(--bg); color: var(--fg); margin: 24px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid var(--line); padding: 6px 8px; vertical-align: top; text-align: left; }}
th {{ background: var(--head); font-weight: 600; }}
small {{ color: var(--muted); }}
.bad {{ background: var(--bad); }} .ok {{ background: var(--ok); }} .good {{ background: var(--good); }}
.src {{ white-space: nowrap; }} .ref {{ font-style: italic; }}
.page {{ display: grid; grid-template-columns: 200px 1fr; gap: 16px; align-items: start; }}
.page img {{ width: 200px; border: 1px solid var(--line); }}
h2 {{ font-size: 15px; margin: 32px 0 8px; }}
@media (max-width: 800px) {{ .page {{ grid-template-columns: 1fr; }} }}
</style>
<h1>Hy-MT2-1.8B quantizations</h1>
<p>{len(pages)} pages; Hy-MT2 with context, greedy decoding. Cell number = chrF of that bubble against the reference.</p>
<table><tr>{heads}</tr>{summary}</table>
{''.join(body)}"""
    (OUT / "quant_report.html").write_text(doc, "utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=list(LABEL))
    ap.add_argument("--cpu", action="store_true", help="also time each model on the CPU build")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    pages = load_pages()
    truth = json.loads((SAMPLES / "mt_truth.json").read_text("utf-8"))
    results, cpu_results = [], []
    for m in args.models:
        print(f"{LABEL[m]} on GPU ...", flush=True)
        results.append(run(m, pages, cpu=False))
        if args.cpu:
            print(f"{LABEL[m]} on CPU ...", flush=True)
            cpu_results.append(run(m, pages, cpu=True))

    q8 = next((r for r in results if r["model"] == "hy-mt2-1.8b-q8_0"), None)
    rows = [score(r, truth, q8) for r in results]
    cpu_rows = {r["model"]: score(r, truth, q8) for r in cpu_results}
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (OUT / f"quant_{stamp}.json").write_text(json.dumps(
        {"summary": rows, "cpu": cpu_rows, "results": results, "cpu_results": cpu_results},
        ensure_ascii=False, indent=1), "utf-8")
    md = table(rows, cpu_rows)
    (OUT / "quant_latest.md").write_text(md + "\n", "utf-8")
    write_report(results, rows, cpu_rows, pages, truth)
    print("\n" + md + f"\n\nreport: {OUT / 'quant_report.html'}")


if __name__ == "__main__":
    main()
