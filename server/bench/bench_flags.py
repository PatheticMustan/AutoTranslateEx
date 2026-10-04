"""Uncertainty flags: does the model's confidence pick out wrong translations?

    python -m bench.bench_flags sample [--run names_quick_none] [--n 300]   # bubbles to label
    python -m bench.bench_flags report [--run ...]

sample: picks a random set of bubbles from a bench_names run and writes
samples/corpus/flags_<run>.json with src, dst and an empty "label" for each.
Labels are written by Claude reading source and translation (the truth-file
rule): "wrong" = a reader would be misled or confused (mistranslation, nonsense,
unrecognizable name), "ok" otherwise (awkward but right counts as ok).

report: how well each signal separates wrong from ok (AUROC), and, per
threshold on the mean token log-probability, how many bubbles get flagged and
how many of the wrong ones that catches.
"""

from __future__ import annotations

import argparse
import json
import random
import sys

from atx.translators.llm import bad_output
from bench.bench_mt import OUT
from bench.fetch_corpus import CORPUS


def labels_path(run: str):
    return CORPUS / f"flags_{run}.json"


def sample(run: str, n: int) -> None:
    data = json.loads((OUT / f"{run}.json").read_text("utf-8"))
    items = []
    for rel, page in data["pages"].items():
        for i, (s, d) in enumerate(zip(page["src"], page["dst"])):
            conf = page["confidence"][i] if i < len(page["confidence"]) else None
            items.append({"id": f"{rel}#{i}", "src": s, "dst": d, "confidence": conf, "label": ""})
    random.Random(7).shuffle(items)
    path = labels_path(run)
    path.write_text(json.dumps(items[:n], ensure_ascii=False, indent=1), "utf-8")
    print(f"wrote {path} ({n} of {len(items)} bubbles)")


def auroc(scores: list[float], wrong: list[bool]) -> float:
    """Probability that a random wrong bubble scores lower than a random ok one."""
    neg = [s for s, w in zip(scores, wrong) if w]
    pos = [s for s, w in zip(scores, wrong) if not w]
    if not neg or not pos:
        return float("nan")
    wins = sum((a < b) + 0.5 * (a == b) for a in neg for b in pos)
    return wins / (len(neg) * len(pos))


def report(run: str) -> None:
    items = [x for x in json.loads(labels_path(run).read_text("utf-8")) if x["label"] in ("ok", "wrong")]
    wrong = [x["label"] == "wrong" for x in items]
    print(f"{len(items)} labelled bubbles, {sum(wrong)} wrong ({sum(wrong) / len(items):.0%})")
    signals = {
        "mean log-prob": [x["confidence"]["mean"] if x["confidence"] else 0.0 for x in items],
        "min log-prob": [x["confidence"]["min"] if x["confidence"] and x["confidence"]["min"] is not None else 0.0
                         for x in items],
        "source length (longer = riskier)": [-len(x["src"]) for x in items],
    }
    for name, s in signals.items():
        print(f"  AUROC {auroc(s, wrong):.2f}  {name}")
    print("\n  mean log-prob below | flagged | wrong caught | precision")
    means = signals["mean log-prob"]
    for t in (-0.2, -0.3, -0.4, -0.5, -0.6, -0.8, -1.0):
        flag = [m < t or bad_output(x["src"], x["dst"]) for m, x in zip(means, items)]
        caught = sum(f and w for f, w in zip(flag, wrong))
        print(f"  {t:>19} | {sum(flag) / len(items):6.0%} | {caught}/{sum(wrong)} ({caught / max(sum(wrong), 1):.0%}) "
              f"| {caught / max(sum(flag), 1):.0%}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["sample", "report"])
    ap.add_argument("--run", default="names_quick_none")
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    sample(args.run, args.n) if args.cmd == "sample" else report(args.run)


if __name__ == "__main__":
    main()
