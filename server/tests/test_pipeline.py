"""Pipeline behaviour (cache, tier keys, previous-page context) with a fake OCR
and a fake translator, so no models are needed."""

import io
import threading
import time

import pytest
from PIL import Image

from atx import pipeline as pl
from atx.cache import Cache
from atx.ocr import Line, OcrResult


def png(color: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (60, 120), (color, color, color)).save(buf, "PNG")
    return buf.getvalue()


class FakeOcr:
    model_set, det_side, device = "fake", 0, "cpu"

    def __init__(self):
        self.calls = 0

    def __call__(self, image):
        self.calls += 1
        # two columns of one bubble, read right to left
        return OcrResult([Line((30, 10, 50, 100), f"右邊{self.calls}", 0.9),
                          Line((10, 10, 30, 100), "左邊", 0.9)])


class FakeTranslator:
    device = "cpu"

    def __init__(self):
        self.contexts = []

    def translate(self, texts, context=None, glossary=None):
        self.contexts.append(context)
        self.glossaries = getattr(self, "glossaries", []) + [glossary]
        self.last_scores = [{"mean": -0.1, "min": -0.5} for _ in texts]
        return [f"EN:{t}" for t in texts]

    def close(self):
        pass


@pytest.fixture
def p(tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "load_ocr", lambda *a, **k: FakeOcr())
    monkeypatch.setattr(pl, "load_detector", lambda: None)
    pipe = pl.Pipeline(cache=Cache(tmp_path / "cache.db"))
    pipe.fake = FakeTranslator()
    pipe._translator = lambda impl: (pipe.fake, 0.0)
    yield pipe
    pipe.cache.close()


def test_translate_and_cache(p):
    img = png(255)
    r = p.translate(img, "very_quick")
    assert (r["w"], r["h"]) == (60, 120)
    assert r["cached"] is False
    assert [x["src"] for x in r["regions"]] == ["右邊1左邊"]
    assert r["regions"][0]["dst"] == "EN:右邊1左邊"
    assert r["regions"][0]["box"] == [10, 10, 50, 100]
    assert r["regions"][0]["vertical"] is True

    again = p.translate(img, "very_quick")
    assert again["cached"] is True
    assert again["regions"] == r["regions"]
    assert p.ocr.calls == 1


def test_tier_switch_reuses_ocr(p):
    img = png(255)
    p.translate(img, "very_quick")
    r = p.translate(img, "quick")
    assert r["cached"] is False and r["ms"]["ocr"] is None  # translated again, OCR from cache
    assert p.ocr.calls == 1


def test_context_only_for_accurate(p):
    a, b = png(255), png(200)
    p.translate(a, "accurate", page_url="u1")
    p.translate(b, "accurate", page_url="u2", prev_url="u1")
    assert p.fake.contexts == [None, ["右邊1左邊"]]

    p.translate(png(100), "quick", page_url="u3", prev_url="u2")
    assert p.fake.contexts[-1] is None


def test_context_waits_for_in_flight_page(p):
    p._expect("u1")
    got = {}
    t = threading.Thread(target=lambda: got.setdefault("ctx", p._context("u1")))
    t.start()
    time.sleep(0.1)
    assert t.is_alive()  # waiting for u1's OCR
    p._remember("u1", ["一", "二"])
    t.join(2)
    assert got["ctx"] == ["一", "二"]


def test_context_after_restart_comes_from_cache(p, tmp_path, monkeypatch):
    p.translate(png(255), "accurate", page_url="u1")
    fresh = pl.Pipeline(cache=Cache(tmp_path / "cache.db"))
    assert fresh._context("u1") == ["右邊1左邊"]
    assert fresh._context("unknown") is None
    fresh.cache.close()


def test_failed_ocr_releases_waiting_page(p, monkeypatch):
    def boom(*a):
        raise RuntimeError("ocr broke")

    monkeypatch.setattr(pl, "ocr_page", boom)
    with pytest.raises(RuntimeError):
        p.translate(png(255), "accurate", page_url="u1")
    t0 = time.perf_counter()
    assert p._context("u1") is None
    assert time.perf_counter() - t0 < 1


def test_unknown_tier_and_bad_image(p):
    with pytest.raises(ValueError):
        p.translate(png(255), "nope")
    with pytest.raises(ValueError, match="not an image"):
        p.translate(b"not an image", "quick")


def test_loaded_4b_skips_free_ram_check(p, monkeypatch):
    from atx import translators

    class Alive:
        alive = True

    monkeypatch.setattr(translators, "resolve", lambda tier: ("accurate-hymt", "low RAM"))
    p._llm, p._llm_spec = Alive(), translators.server_spec("accurate")
    assert p.resolve("accurate") == "accurate" and p.tier_note is None
    p._llm = p._llm_spec = None
    p.resolve("quick")
    assert p.resolve("accurate") == "accurate-hymt" and p.tier_note == "low RAM"


def test_page_cached_without_context_is_redone_once_context_exists(p):
    a, b = png(255), png(200)
    # Page 2 arrives first: no context yet, so it's translated without it.
    r = p.translate(b, "accurate", page_url="u2", prev_url="u1")
    assert r["with_context"] is False and p.fake.contexts == [None]
    p.translate(a, "accurate", page_url="u1")
    again = p.translate(b, "accurate", page_url="u2", prev_url="u1")
    assert again["cached"] is False and again["with_context"] is True
    assert p.fake.contexts[-1] == ["右邊2左邊"]
    # Now it's cached with context and stays cached.
    assert p.translate(b, "accurate", page_url="u2", prev_url="u1")["cached"] is True
    # Tiers without context never redo.
    p.translate(b, "quick", page_url="u2", prev_url="u1")
    assert p.translate(b, "quick", page_url="u2", prev_url="u1")["cached"] is True


def test_recover_missed_maps_crop_lines_back_to_the_page():
    class CropOcr:
        def __call__(self, image):  # sees the 2x crop; reports a line in crop pixels
            return OcrResult([Line((24, 24, 64, 224), "我媽想去", 0.99)])

    page = Image.new("RGB", (400, 600), "white")
    lines = pl.recover_missed(CropOcr(), page, [(100, 200, 150, 320)])
    # crop origin = region - 12px margin = (88, 188); crop pixels / 2 + origin
    assert lines[0].box == (88 + 12, 188 + 12, 88 + 32, 188 + 112)
    assert lines[0].text == "我媽想去"


def test_frames_attach_to_the_bubble_containing_the_text():
    from atx.grouping import attach_frames, group_lines
    bubbles = group_lines([Line((30, 10, 50, 100), "右邊", 0.9), Line((300, 10, 320, 100), "左邊", 0.9)])
    attach_frames(bubbles, [(0, 0, 100, 150), (10, 5, 80, 120), (280, 0, 340, 140)])
    frames = {b.text: b.frame for b in bubbles}
    assert frames["右邊"] == (10, 5, 80, 120)  # the smaller of two nested frames
    assert frames["左邊"] == (280, 0, 340, 140)


def test_name_bank_feeds_the_translator_and_redoes_pages_when_it_learns(p, monkeypatch):
    from atx import names
    # Pretend jieba finds the name 右邊 on every page that has it.
    monkeypatch.setattr(names, "find_names", lambda text: ["右邊"] if "右邊" in text else [])
    a, b = png(255), png(200)
    p.translate(a, "quick", series="s1")
    assert p.fake.glossaries[-1] == {}  # seen on one page: not trusted yet
    p.translate(b, "quick", series="s1")
    assert p.fake.glossaries[-1] == {"右邊": "Youbian"}  # second page: in the bank
    r = p.translate(a, "quick", series="s1")  # page one now has a known name: redone
    assert r["cached"] is False and p.fake.glossaries[-1] == {"右邊": "Youbian"}
    assert p.translate(a, "quick", series="s1")["cached"] is True
    # A user spelling change redoes it again.
    p.names.set("s1", "右邊", "Right")
    assert p.translate(a, "quick", series="s1")["cached"] is False
    assert p.fake.glossaries[-1] == {"右邊": "Right"}
    # Without a series, no bank.
    p.translate(png(100), "quick")
    assert p.fake.glossaries[-1] == {}


def test_regions_carry_confidence_and_flags(p):
    r = p.translate(png(255), "quick")
    assert r["regions"][0]["confidence"] == {"mean": -0.1, "min": -0.5}
    # The fake translator echoes the Chinese back: a broken output is always flagged.
    assert r["regions"][0]["flagged"] is True


def test_retranslate_replaces_only_the_chosen_bubbles(p):
    class Better(FakeTranslator):
        def translate(self, texts, context=None, glossary=None):
            self.last_scores = [{"mean": -0.01, "min": -0.1} for _ in texts]
            return [f"BETTER:{i}" for i, _ in enumerate(texts)]

    r = p.translate(png(255), "quick")
    better = Better()
    p._translator = lambda impl: (better if impl != "quick" else p.fake, 0.0)
    out = p.retranslate(r["id"], "quick", [0])
    assert list(out["regions"]) == [0]
    assert out["regions"][0]["dst"] == "BETTER:0" and out["regions"][0]["by"] != "quick"
    # The page's cached result for its own tier now has the better text.
    assert p.translate(png(255), "quick")["regions"][0]["dst"] == "BETTER:0"
    with pytest.raises(ValueError):
        p.retranslate("0" * 40, "quick", [0])


def test_retranslate_works_after_the_name_bank_moved_the_cache_key(p, monkeypatch):
    from atx import names
    monkeypatch.setattr(names, "find_names", lambda text: ["右邊"] if "右邊" in text else [])
    r = p.translate(png(255), "quick", series="s1")
    p.translate(png(200), "quick", series="s1")  # now 右邊 is in the bank: page one's key changed
    out = p.retranslate(r["id"], "quick", [0], series="s1")
    assert out["regions"][0]["src"] == r["regions"][0]["src"]
