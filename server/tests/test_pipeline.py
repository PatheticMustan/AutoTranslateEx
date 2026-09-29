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

    def translate(self, texts, context=None):
        self.contexts.append(context)
        return [f"EN:{t}" for t in texts]

    def close(self):
        pass


@pytest.fixture
def p(tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "load_ocr", lambda *a, **k: FakeOcr())
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
