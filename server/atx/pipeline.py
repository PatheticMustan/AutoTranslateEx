"""Page image -> translated speech bubbles.

`Pipeline` is what the server runs: it keeps the OCR model and one translation
tier warm, caches results, and gives the accurate tiers the previous page's
text as context. atx.app is a thin HTTP layer over it.

Debug CLI:
    python -m atx.pipeline samples/20001/229697/008.jpg --debug out.png [--models v6-tiny] [--device dml]
    python -m atx.pipeline samples/20001/229697/008.jpg --tier quick

Prints each bubble (index, box, direction, text, and with --tier the
translation) and, with --debug, saves a copy of the page with line boxes (thin)
and numbered bubble boxes (thick).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import sys
import threading
import time
from pathlib import Path

from atx import device as devmod
from atx import translators
from atx.cache import PIPELINE_VERSION, Cache
from atx.grouping import Bubble, attach_frames, group_lines, split_by_regions, uncovered
from atx.names import NameBank, names_from_translation, terms_for
from atx.ocr import DEFAULT_DET_SIDE, DEFAULT_MODEL_SET, MODEL_SETS, Line, Ocr, OcrResult
from atx.translators.base import Translator

log = logging.getLogger("atx")

CONTEXT_TIERS = {"accurate", "accurate-hymt", "accurate-2b"}


def flagged(impl: str, src: str, dst: str, conf: dict | None) -> bool:
    """Whether a translation is likely wrong enough to highlight: see FLAG_BELOW."""
    from atx.translators.llm import bad_output

    if bad_output(src, dst):
        return True
    threshold = FLAG_BELOW.get(impl)
    return bool(conf and threshold is not None and conf["mean"] < threshold)


# Flag a bubble when the model's mean token log-probability is below this.
# Set from the uncertainty benchmark (bench/bench_flags.py); None = never by score.
FLAG_BELOW: dict[str, float | None] = {}
CONTEXT_WAIT_S = 15  # how long a page waits for the previous page's OCR, if it's in flight
MAX_REMEMBERED_PAGES = 200


def ocr_page(ocr: Ocr, image, detector=None) -> tuple[list[Bubble], OcrResult]:
    """OCR a page and group its lines into bubbles.

    With a detector (atx.detector), text regions OCR missed are OCR'd again on
    their own (cropped, upscaled 2x), and each bubble gets the detected speech
    bubble around it as its frame.
    """
    result = ocr(image)
    lines = result.lines
    frames: list = []
    regions: list = []
    if detector is not None:
        page = _open(image)
        found = detector(page)
        frames = [d.box for d in found if d.label == "bubble"]
        regions = [d.box for d in found if d.label != "bubble"]
        lines = lines + recover_missed(ocr, page, uncovered(regions, lines))
        result.seconds["detect"] = detector.last_ms / 1000
    bubbles = split_by_regions(group_lines(lines), regions)
    attach_frames(bubbles, frames)
    return bubbles, result


RECOVER_MARGIN = 12


def recover_missed(ocr: Ocr, page, regions: list) -> list[Line]:
    """OCR each region on its own. The full-page pass misses some tilted or
    stylized text; a 2x crop reads it (e.g. 我媽想去血拼囉 over a sky panel)."""
    out = []
    w, h = page.size
    for r in regions:
        x0, y0 = max(0, r[0] - RECOVER_MARGIN), max(0, r[1] - RECOVER_MARGIN)
        x1, y1 = min(w, r[2] + RECOVER_MARGIN), min(h, r[3] + RECOVER_MARGIN)
        crop = page.crop((x0, y0, x1, y1))
        crop = crop.resize((crop.width * 2, crop.height * 2))
        buf = io.BytesIO()
        crop.save(buf, "PNG")
        for ln in ocr(buf.getvalue()).lines:
            bx0, by0, bx1, by1 = ln.box
            out.append(Line((x0 + bx0 // 2, y0 + by0 // 2, x0 + bx1 // 2, y0 + by1 // 2), ln.text, ln.score))
    return out


def _open(image):
    from PIL import Image

    if isinstance(image, (bytes, bytearray)):
        image = io.BytesIO(image)
    return Image.open(image).convert("RGB")


def load_detector():
    """The bubble detector, or None when its model isn't downloaded
    (python -m atx.models comic-detector) or ATX_DETECTOR=0."""
    import os

    if os.environ.get("ATX_DETECTOR", "1") == "0":
        return None
    from atx import detector

    if not detector.available():
        log.info("bubble detector not downloaded (python -m atx.models comic-detector); grouping without it")
        return None
    return detector.Detector()


def image_size(image: bytes) -> tuple[int, int]:
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(io.BytesIO(image)) as img:  # reads the header only
            return img.size
    except UnidentifiedImageError as e:
        raise ValueError("not an image") from e


def load_ocr(model_set: str = DEFAULT_MODEL_SET, device: str = "auto") -> Ocr:
    """device "auto": DirectML on a discrete GPU, else CPU.

    On the target's integrated GPU the CPU is the default (PLAN.md, Compute
    device): DirectML tends to lose on these small models there, and the CPU can
    OCR the next page while the iGPU runs the LLM. A DirectML failure falls back
    to CPU.
    """
    info = devmod.detect()
    if device == "auto":
        gpu = info.primary_gpu
        discrete = gpu is not None and gpu.vendor in ("nvidia", "amd")
        device = "dml" if discrete and "DmlExecutionProvider" in info.onnx_providers else "cpu"
    if device == "dml":
        try:
            ocr = Ocr(model_set, "dml")
            ocr(_blank_page())  # DirectML errors can surface on the first run, not at load
            return ocr
        except Exception as e:
            devmod.record_fallback("ocr", f"DirectML failed: {e}")
    return Ocr(model_set, "cpu")


def _blank_page():
    import numpy as np

    return np.full((256, 256, 3), 255, np.uint8)


class Pipeline:
    def __init__(self, ocr_models: str = DEFAULT_MODEL_SET, ocr_device: str = "auto",
                 cache: Cache | None = None):
        self.ocr = load_ocr(ocr_models, ocr_device)
        self.detector = load_detector()
        # Bubbles are cached after grouping, so the key carries the pipeline version too.
        self.ocr_key = (f"{self.ocr.model_set}@{self.ocr.det_side}"
                        f"{'+det' if self.detector else ''}|v{PIPELINE_VERSION}")
        self.cache = cache or Cache()
        self.names = NameBank(self.cache)
        self._ocr_lock = threading.Lock()
        # One page translates at a time: the LLM tiers already batch a page's
        # bubbles, and this also makes tier switches safe.
        self._mt_lock = threading.Lock()

        self.tier: str | None = None        # tier last asked for (very_quick / quick / accurate)
        self.resolved: str | None = None    # what it runs as on this machine
        self.tier_note: str | None = None   # why it isn't the default, if so
        self.loading: str | None = None     # tier being loaded right now
        self._opus: Translator | None = None
        self._llm = None                    # LlamaServer
        self._llm_spec: tuple | None = None
        self._on_llm: dict[str, Translator] = {}

        # page URL -> its bubble texts, for the next page's context
        self._texts: dict[str, list[str]] = {}
        self._pending: dict[str, threading.Event] = {}
        self._book_lock = threading.Lock()

    # ---- tiers ---------------------------------------------------------

    def resolve(self, tier: str) -> str:
        """The machine's implementation of a tier. Decided once per tier switch,
        so the free-memory check doesn't flip while the 4B model itself is loaded."""
        if tier not in translators.TIERS:
            raise ValueError(f"unknown tier {tier!r}; choose from {translators.PUBLIC_TIERS}")
        if tier != self.tier:
            spec = translators.server_spec("accurate")
            if tier == "accurate" and self._llm_spec == spec and self._llm.alive:
                # Already loaded: the free-RAM check would count the model against itself.
                self.resolved, self.tier_note = "accurate", None
            else:
                self.resolved, self.tier_note = translators.resolve(tier)
            self.tier = tier
            if self.tier_note:
                log.warning("tier %s runs as %s: %s", tier, self.resolved, self.tier_note)
        return self.resolved

    def warm(self, tier: str) -> float:
        """Load a tier's model if it isn't loaded. Returns the seconds spent loading."""
        with self._mt_lock:
            _, load_s = self._translator(self.resolve(tier))
            return load_s

    def _translator(self, impl: str) -> tuple[Translator, float]:
        """Call with _mt_lock held."""
        spec = translators.server_spec(impl)
        if spec is None:
            if self._opus is not None:
                return self._opus, 0.0
            self.loading = impl
            t0 = time.perf_counter()
            try:
                from atx.translators.opus import OpusTranslator
                self._opus = OpusTranslator()  # small, so it stays loaded next to the LLM
            finally:
                self.loading = None
            return self._opus, time.perf_counter() - t0

        if self._llm is not None and self._llm_spec == spec and self._llm.alive:
            if impl not in self._on_llm:
                self._on_llm[impl] = translators.on_server(impl, self._llm)
            return self._on_llm[impl], 0.0

        from atx.llm_process import LlamaServer

        if self._llm is not None and not self._llm.alive:
            log.warning("llama-server (%s) exited; restarting", self._llm_spec[0])
        self._stop_llm()  # only one LLM at a time
        model, parallel, ctx_per_slot = spec
        self.loading = impl
        t0 = time.perf_counter()
        try:
            server = LlamaServer(model, parallel=parallel, ctx_per_slot=ctx_per_slot)
            server.start()
        finally:
            self.loading = None
        self._llm, self._llm_spec = server, spec
        self._on_llm[impl] = translators.on_server(impl, server)
        load_s = time.perf_counter() - t0
        log.info("loaded %s on %s (%s) in %.1fs", model, server.device, server.offload, load_s)
        return self._on_llm[impl], load_s

    def _stop_llm(self) -> None:
        for t in self._on_llm.values():
            t.close()
        self._on_llm.clear()
        if self._llm is not None:
            self._llm.stop()
        self._llm = self._llm_spec = None

    def close(self) -> None:
        with self._mt_lock:
            self._stop_llm()
            if self._opus is not None:
                self._opus.close()
                self._opus = None
        self.cache.close()

    # ---- page context --------------------------------------------------

    def _expect(self, url: str) -> None:
        with self._book_lock:
            if url not in self._texts:
                self._pending.setdefault(url, threading.Event())

    def _remember(self, url: str, texts: list[str] | None) -> None:
        with self._book_lock:
            if texts is not None:
                self._texts[url] = texts
                while len(self._texts) > MAX_REMEMBERED_PAGES:
                    self._texts.pop(next(iter(self._texts)))
            event = self._pending.pop(url, None)
        if event:
            event.set()

    def _context_known(self, prev_url: str) -> bool:
        """Whether the previous page's text is available right now (no waiting)."""
        with self._book_lock:
            if self._texts.get(prev_url) is not None:
                return True
        sha1 = self.cache.page_sha1(prev_url)
        return sha1 is not None and self.cache.get_ocr(sha1, self.ocr_key) is not None

    def _stale_without_context(self, hit: dict, impl: str, prev_url: str | None) -> bool:
        return (impl in CONTEXT_TIERS and bool(prev_url) and not hit.get("with_context", True)
                and self._context_known(prev_url))

    def _context(self, prev_url: str | None) -> list[str] | None:
        """The previous page's bubble texts: from memory, by waiting for its OCR
        if that request is in flight, or from the cache (e.g. after a restart)."""
        if not prev_url:
            return None
        with self._book_lock:
            texts = self._texts.get(prev_url)
            event = self._pending.get(prev_url)
        if texts is None and event is not None:
            event.wait(CONTEXT_WAIT_S)
            with self._book_lock:
                texts = self._texts.get(prev_url)
        if texts is None and (sha1 := self.cache.page_sha1(prev_url)):
            bubbles = self.cache.get_ocr(sha1, self.ocr_key)
            texts = [b["text"] for b in bubbles] if bubbles is not None else None
        return texts

    # ---- the request ---------------------------------------------------

    def _bubbles(self, sha1: str, image: bytes) -> tuple[list[dict], float | None]:
        """OCR'd bubbles as dicts, and the OCR time (None when cached)."""
        bubbles = self.cache.get_ocr(sha1, self.ocr_key)
        if bubbles is not None:
            return bubbles, None
        with self._ocr_lock:
            t0 = time.perf_counter()
            found, _ = ocr_page(self.ocr, image, self.detector)
            ocr_s = time.perf_counter() - t0
        bubbles = [{"box": list(b.box), "text": b.text, "vertical": b.vertical,
                    "frame": list(b.frame) if b.frame else None} for b in found]
        self.cache.put_ocr(sha1, self.ocr_key, bubbles)
        return bubbles, ocr_s

    def translate(self, image: bytes, tier: str, page_url: str | None = None,
                  prev_url: str | None = None, series: str | None = None) -> dict:
        """-> {id, w, h, regions: [{box, src, dst, vertical, frame, confidence, flagged}],
               tier, cached, ms, note}

        `series` turns on the name bank: names found on the page are recorded, and
        the series' known names are given to the translator with fixed spellings.
        """
        t0 = time.perf_counter()
        sha1 = hashlib.sha1(image).hexdigest()
        impl = self.resolve(tier)
        model = translators.TIERS[impl][0] or "opus-mt-zh-en"
        if page_url:
            self._expect(page_url)
            self.cache.put_page(page_url, sha1)

        texts = None
        try:
            bubbles, ocr_s = self._bubbles(sha1, image)
            texts = [b["text"] for b in bubbles]
        finally:
            if page_url:
                self._remember(page_url, texts)  # also on failure, so the next page stops waiting

        glossary: dict[str, str] = {}
        if series:
            self.names.observe(series, sha1, texts)
            glossary = self.names.active(series)
        # The names this page uses are part of the key: when the bank learns a
        # name or a spelling is changed, pages with that name are redone.
        used = sorted({term for t in texts for term in terms_for(t, glossary)})
        names_key = hashlib.sha1(repr(used).encode()).hexdigest()[:8] if used else "-"
        key = f"{impl}|{model}|{self.ocr_key}|names:{names_key}|v{PIPELINE_VERSION}"

        hit = self.cache.get_result(sha1, key)
        if hit is not None and self._stale_without_context(hit, impl, prev_url):
            hit = None  # translated before the previous page was ready; redo it with context
        if hit is not None:
            ms = {"total": round((time.perf_counter() - t0) * 1000)}
            log.info("%s %s cached, %d bubbles, %d ms", sha1[:8], impl, len(texts), ms["total"])
            return {**hit, "id": sha1, "tier": impl, "cached": True, "ms": ms, "note": self.tier_note}

        w, h = image_size(image)
        context = self._context(prev_url) if impl in CONTEXT_TIERS else None
        load_s = mt_s = 0.0
        dst: list[str] = []
        scores: list = []
        if texts:
            with self._mt_lock:
                translator, load_s = self._translator(impl)
                t1 = time.perf_counter()
                dst = translator.translate(texts, context, glossary)
                scores = list(getattr(translator, "last_scores", []) or [])
                mt_s = time.perf_counter() - t1
        scores += [None] * (len(dst) - len(scores))
        if series:  # names the translator spelled out in pinyin, for later pages
            self.names.observe_names(series, sha1, [n for src, d in zip(texts, dst)
                                                    for n in names_from_translation(src, d)])

        result = {"w": w, "h": h, "with_context": bool(context), "regions": [
            {"box": b["box"], "src": b["text"], "dst": d, "vertical": b["vertical"], "frame": b.get("frame"),
             "confidence": c, "flagged": flagged(impl, b["text"], d, c)}
            for b, d, c in zip(bubbles, dst, scores)
        ]}
        self.cache.put_result(sha1, key, result)
        ms = {"ocr": None if ocr_s is None else round(ocr_s * 1000), "load": round(load_s * 1000),
              "translate": round(mt_s * 1000), "total": round((time.perf_counter() - t0) * 1000)}
        log.info("%s %s %d bubbles%s%s: ocr %s, translate %d ms, total %d ms%s",
                 sha1[:8], impl, len(texts), " (with context)" if context else "",
                 f" ({len(used)} names)" if used else "",
                 "cached" if ms["ocr"] is None else f"{ms['ocr']} ms", ms["translate"], ms["total"],
                 f" (cold: load {ms['load']} ms)" if load_s else "")
        return {**result, "id": sha1, "tier": impl, "cached": False, "ms": ms, "note": self.tier_note}

    # ---- status --------------------------------------------------------

    def health(self) -> dict:
        import psutil

        info = devmod.detect()
        llm = None
        if self._llm is not None:
            llm = {"model": self._llm_spec[0], "device": self._llm.device, "offload": self._llm.offload,
                   "alive": self._llm.alive}
        ocr_device = self.ocr.device
        if ocr_device == "dml" and info.primary_gpu:
            ocr_device = f"dml ({info.primary_gpu.name})"
        return {
            "ok": True,
            "version": PIPELINE_VERSION,
            "tiers": translators.PUBLIC_TIERS,
            "tier": self.tier, "resolved": self.resolved, "note": self.tier_note,
            "loading": self.loading,
            "device": info.summary(),
            "components": {
                "ocr": {"model": self.ocr_key, "device": ocr_device},
                "very_quick": None if self._opus is None else {"model": "opus-mt-zh-en", "device": self._opus.device},
                "llm": llm,
            },
            "fallbacks": info.fallbacks,
            "free_ram_mb": psutil.virtual_memory().available // 2**20,
        }


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
    ap.add_argument("--tier", choices=list(translators.TIERS), help="also translate with this tier")
    ap.add_argument("--debug", type=Path, help="save an annotated copy of the page here")
    ap.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252

    ocr = Ocr(args.models, args.device, det_side=args.det_side)
    bubbles, result = ocr_page(ocr, args.image)
    dst = [""] * len(bubbles)
    if args.tier and bubbles:
        tr = translators.make(translators.resolve(args.tier)[0])
        try:
            dst = tr.translate([b.text for b in bubbles])
        finally:
            tr.close()

    if args.json:
        print(json.dumps([{"box": b.box, "vertical": b.vertical, "text": b.text, "dst": d}
                          for b, d in zip(bubbles, dst)], ensure_ascii=False, indent=2))
    else:
        t = result.seconds
        print(f"{len(result.lines)} lines -> {len(bubbles)} bubbles  "
              f"(det {t['det'] * 1000:.0f} ms, rec {t['rec'] * 1000:.0f} ms, total {t['total'] * 1000:.0f} ms)")
        for i, (b, d) in enumerate(zip(bubbles, dst), 1):
            print(f"{i:2d} {'V' if b.vertical else 'H'} {b.box}  {b.text}" + (f"\n     -> {d}" if d else ""))
    if args.debug:
        draw_debug(args.image, bubbles, args.debug)
        print(f"saved {args.debug}")


if __name__ == "__main__":
    main()
