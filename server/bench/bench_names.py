"""Name bank benchmark: does the bank make names come out right and consistent?

    python -m bench.bench_names run --mode none   [--tier quick] [--limit N]
    python -m bench.bench_names run --mode auto
    python -m bench.bench_names run --mode oracle
    python -m bench.bench_names candidates        # names seen in the runs, for writing the reference
    python -m bench.bench_names report

Replays the corpus (samples/corpus/ocr.json) chapter by chapter in reading order,
as a reader would:
- none:   no name bank;
- auto:   the bank learns as it goes, exactly as the server does (jieba names on
          each page, plus names the translator spelled in pinyin), and the
          translator gets its active names;
- oracle: the reference bank (samples/corpus/names_truth.json) from the start.

Every run saves all translations with their confidence scores to
bench/out/names_<tier>_<mode>.json; the uncertainty benchmark reuses them.

report: for every bubble whose source contains a reference name, whether the
English contains that name's reference spelling (ignoring case, spaces and
hyphens). Also how well the auto bank's names match the reference.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict

from atx import translators
from atx.cache import Cache
from atx.names import NameBank, find_names, names_from_translation
from bench.bench_mt import OUT
from bench.fetch_corpus import CORPUS

TRUTH = CORPUS / "names_truth.json"


def corpus_pages() -> list[tuple[str, list[str]]]:
    """(page, bubble texts) in reading order: chapter ids are in story order."""
    ocr = json.loads((CORPUS / "ocr.json").read_text("utf-8"))
    key = lambda rel: (rel.split("/")[0], int(rel.split("/")[1]), rel.split("/")[2])
    return [(rel, [b["text"] for b in ocr[rel]]) for rel in sorted(ocr, key=key)]


def run_path(tier: str, mode: str):
    return OUT / f"names_{tier}_{mode}.json"


def run(tier: str, mode: str, limit: int | None) -> None:
    pages = corpus_pages()[:limit]
    truth = json.loads(TRUTH.read_text("utf-8"))["names"] if mode == "oracle" else {}
    db = OUT / f"names_{tier}_{mode}.db"
    db.unlink(missing_ok=True)
    bank = NameBank(Cache(db))
    impl = translators.resolve(tier)[0]
    tr = translators.make(impl)
    out, prev, t0 = {}, {}, time.perf_counter()
    try:
        for n, (rel, texts) in enumerate(pages, 1):
            series, chapter = rel.split("/")[:2]
            if mode == "auto":
                bank.observe(series, rel, texts)
            glossary = truth if mode == "oracle" else bank.active(series) if mode == "auto" else {}
            context = prev.get(chapter) if impl in ("accurate", "accurate-hymt") else None
            dst = tr.translate(texts, context, glossary) if texts else []
            if mode == "auto":
                bank.observe_names(series, rel, [x for s, d in zip(texts, dst) for x in names_from_translation(s, d)],
                                   texts)
            out[rel] = {"src": texts, "dst": dst, "confidence": list(getattr(tr, "last_scores", []) or [])}
            prev[chapter] = texts
            if n % 50 == 0:
                print(f"{n}/{len(pages)} pages, {(time.perf_counter() - t0) / n * 1000:.0f} ms/page", flush=True)
    finally:
        tr.close()
    result = {"tier": impl, "mode": mode, "pages": out}
    if mode == "auto":
        result["bank"] = bank.all("20001")
        result["active"] = bank.active("20001")
    run_path(tier, mode).write_text(json.dumps(result, ensure_ascii=False), "utf-8")
    print(f"saved {run_path(tier, mode)}")


def candidates() -> None:
    """Names jieba finds and names the translator spelled in pinyin, with page counts."""
    jieba_pages, mt_pages = defaultdict(set), defaultdict(set)
    for rel, texts in corpus_pages():
        for t in texts:
            for n in find_names(t):
                jieba_pages[n].add(rel)
    for path in OUT.glob("names_*_none.json"):
        for rel, page in json.loads(path.read_text("utf-8"))["pages"].items():
            for s, d in zip(page["src"], page["dst"]):
                for n in names_from_translation(s, d):
                    mt_pages[n].add(rel)
    names = set(jieba_pages) | set(mt_pages)
    rows = sorted(names, key=lambda n: -(len(jieba_pages[n]) + len(mt_pages[n])))
    for n in rows:
        j, m = len(jieba_pages[n]), len(mt_pages[n])
        if j + m >= 2:
            print(f"{n}\tjieba {j}\ttranslation {m}")


def norm(s: str) -> str:
    return re.sub(r"[\s\-'’]", "", s.lower())


def report() -> None:
    truth = json.loads(TRUTH.read_text("utf-8"))["names"]
    runs = sorted(OUT.glob("names_*_*.json"))
    print(f"reference: {len(truth)} names")
    for path in runs:
        data = json.loads(path.read_text("utf-8"))
        hits, total, per_name = 0, 0, defaultdict(lambda: [0, 0])
        for page in data["pages"].values():
            for s, d in zip(page["src"], page["dst"]):
                for zh, en in truth.items():
                    if zh in s:
                        ok = norm(en) in norm(d)
                        hits += ok
                        total += 1
                        per_name[zh][0] += ok
                        per_name[zh][1] += 1
        worst = sorted(per_name.items(), key=lambda kv: kv[1][0] / kv[1][1])[:5]
        line = f"{path.stem}: names right {hits}/{total} ({hits / max(total, 1):.0%})"
        if "bank" in data:
            active = data.get("active") or {r["zh"]: r["en"] for r in data["bank"] if r["pages"] >= 2}
            found = set(active) & set(truth)
            spelled = sum(norm(active[z]) == norm(truth[z]) for z in found)
            line += (f"; bank: {len(active)} active, {len(found)}/{len(truth)} reference names found "
                     f"({spelled} spelled the same), {len(set(active) - set(truth))} not in the reference")
        print(line)
        print("   weakest:", ", ".join(f"{zh}={truth[zh]} {a}/{b}" for zh, (a, b) in worst))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--mode", choices=["none", "auto", "oracle"], required=True)
    r.add_argument("--tier", default="quick")
    r.add_argument("--limit", type=int)
    sub.add_parser("candidates")
    sub.add_parser("report")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if args.cmd == "run":
        run(args.tier, args.mode, args.limit)
    elif args.cmd == "candidates":
        candidates()
    else:
        report()


if __name__ == "__main__":
    main()
