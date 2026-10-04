"""Local HTTP server for the extension (127.0.0.1 only).

    python -m atx.app [--tier quick] [--port 8765] [--ocr-device auto|cpu|dml]

POST /translate  multipart: image (file), tier, page_url?, prev_url?
                 -> {id, w, h, regions: [{box: [x0,y0,x1,y1], src, dst, vertical}],
                     tier, cached, ms: {ocr, load, translate, total}, note}
                 page_url/prev_url let the accurate tier use the previous page as context.
POST /warm       form: tier. Loads that tier's model now (the popup calls it on a tier change).
GET  /health     tiers, current tier, device per component, fallback reasons, free RAM.
DELETE /cache    forget all cached OCR and translations.

The OCR model and the startup tier load in the background, so /health answers
right away; a /translate that arrives first waits for them.
"""

from __future__ import annotations

import argparse
import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from atx.ocr import DEFAULT_MODEL_SET, MODEL_SETS
from atx.pipeline import Pipeline

log = logging.getLogger("atx")

MAX_IMAGE_BYTES = 25 * 2**20

config = {"tier": "quick", "ocr_models": DEFAULT_MODEL_SET, "ocr_device": "auto"}
_state: dict = {"pipeline": None, "error": None}
_ready = threading.Event()


def _startup() -> None:
    try:
        pipeline = Pipeline(config["ocr_models"], config["ocr_device"])
        _state["pipeline"] = pipeline
        _ready.set()  # OCR is up; the tier below can load while requests queue behind it
        pipeline.warm(config["tier"])
        log.info("ready: %s", pipeline.health()["device"])
    except Exception as e:
        log.exception("startup failed")
        _state["error"] = f"{type(e).__name__}: {e}"
        _ready.set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_startup, name="atx-startup", daemon=True).start()
    yield
    if _state["pipeline"] is not None:
        _state["pipeline"].close()


app = FastAPI(title="AutoTranslateEx", lifespan=lifespan)
# The extension's service worker doesn't need CORS (host permission), but
# extension pages such as the popup might fetch directly, and the dev harness
# (extension/dev/harness.html) is served from localhost.
app.add_middleware(CORSMiddleware,
                   allow_origin_regex=r"chrome-extension://.*|http://(localhost|127\.0\.0\.1)(:\d+)?",
                   allow_methods=["*"], allow_headers=["*"])


def _pipeline(wait: bool = True) -> Pipeline:
    if wait:
        _ready.wait(300)
    if _state["error"]:
        raise HTTPException(503, f"server failed to start: {_state['error']}")
    if _state["pipeline"] is None:
        raise HTTPException(503, "starting")
    return _state["pipeline"]


@app.get("/health")
def health() -> dict:
    if _state["pipeline"] is None:
        return {"ok": False, "error": _state["error"], "loading": None if _state["error"] else "ocr"}
    return _state["pipeline"].health()


# Sync handlers run in FastAPI's thread pool, so OCR of one page overlaps
# translation of another.
@app.post("/translate")
def translate(image: UploadFile = File(...), tier: str = Form("quick"),
              page_url: str | None = Form(None), prev_url: str | None = Form(None)) -> dict:
    data = image.file.read(MAX_IMAGE_BYTES + 1)
    if not data:
        raise HTTPException(400, "empty image")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "image too large")
    pipeline = _pipeline()
    try:
        return pipeline.translate(data, tier, page_url, prev_url)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.post("/warm")
def warm(tier: str = Form(...)) -> dict:
    pipeline = _pipeline()
    try:
        load_s = pipeline.warm(tier)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {**pipeline.health(), "load_ms": round(load_s * 1000)}


@app.delete("/cache")
def clear_cache() -> dict:
    _pipeline().cache.clear()
    return {"ok": True}


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--tier", default=config["tier"], help="tier to load at startup")
    ap.add_argument("--ocr-models", default=config["ocr_models"], choices=list(MODEL_SETS))
    ap.add_argument("--ocr-device", default=config["ocr_device"], choices=["auto", "cpu", "dml"],
                    help="auto: DirectML on a discrete GPU, CPU otherwise")
    args = ap.parse_args()
    config.update(tier=args.tier, ocr_models=args.ocr_models, ocr_device=args.ocr_device)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per llama-server request otherwise
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
