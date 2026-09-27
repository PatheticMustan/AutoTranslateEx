# AutoTranslateEx — Implementation Plan

Experimental, solo-maintained prototype. Anything here can be rewritten or thrown away; optimize for fast iteration, not stability.

## Goal & scope

A Chrome extension that replaces Chinese speech-bubble text in webcomic pages with English, drawn over the bubble on a canvas.

Deliberately narrow, so we can optimize hard:

| In scope | Out of scope (for now) |
|---|---|
| One site: `mycomic.com` chapter pages | Other sites, canvas-based readers |
| Traditional Chinese → English | Other language pairs |
| `<img>` pages ~620–690px wide, 1.3–2k px tall | Very tall webtoon strips (tiling) |
| Target laptop (integrated GPU), plus the dev machine's discrete GPU | Other hardware tuning |
| Clean white/flat-colored box over text | Inpainting / text removal |

**Target machine:** Lenovo Yoga 9 2-in-1 14IMH9 (Gen 9):
- Intel Core Ultra 7 155H: 6P + 8E + 2LPE cores, 22 threads.
- Intel Arc integrated GPU (8 Xe-LPG cores, **no XMX** matrix units). No NVIDIA GPU.
- 16 GB LPDDR5X-7467 (~120 GB/s theoretical), shared by CPU and GPU. **Only ~6.4 GB is typically free**; free RAM, not speed, is the tightest constraint.
- Windows 11 Home.

**Dev machine:** Dell G16 (i9-12900H, RTX 3070 Ti Laptop 8 GB, 32 GB). Benchmarks from it do **not** transfer to the target; Phase 2 must be re-run on the Yoga.

The GPU is used when it is available *and* faster; otherwise the CPU (see [Compute device](#compute-device)).

**Budgets:** time per page, assuming ~10 bubbles and ~250 English output tokens. These are estimates for now; Phase 2 replaces them with measurements.

| Tier | Target: Arc iGPU (155H) | Target: CPU only (155H) | Dev: discrete GPU (RTX 3070 Ti Laptop) |
|---|---|---|---|
| OCR | — (runs on CPU) | ≤ 0.35 s | ≤ 0.15 s |
| Very quick | — (runs on CPU) | ≤ 0.3 s (plus OCR) | ≤ 0.3 s |
| Quick (bubbles in parallel) | ≤ 3 s | ≤ 5 s | ≤ 2 s |
| Accurate, Qwen3.5-2B | ≤ 10 s | ≤ 15 s | — |
| Accurate, Qwen3.5-4B | 17–25 s (slower than reading) | 25–35 s | ≤ 4 s |

- **Token generation** is limited by memory bandwidth, so the Arc iGPU is only ~1.2× faster than the CPU there. **Prompt reading** is compute-bound; without XMX the iGPU is only ~2× faster. On the target, CPU and iGPU are close; the iGPU path mainly frees the CPU for OCR and the browser.
- A discrete GPU speeds up the LLM tiers ~8–12× because of its much higher VRAM bandwidth.
- Prefetching hides latency only when translation is faster than reading (~10–20 s per page). The accurate tier with the 4B model is not, so the target uses the 2B model for that tier (see [Model choices](#translation-three-tiers)).
- Estimates assume the laptop is plugged in on Best Performance. Expect ~15–25% slowdown under sustained load (28 W sustained power) and much more on battery in efficiency mode.

**Memory budget (target):** with an integrated GPU the LLM weights sit in *shared system RAM*, not VRAM.

| Tier | Total resident (server + llama-server) |
|---|---|
| Very quick | ~0.5 GB |
| Quick (Hy-MT2-1.8B) | ~1.8 GB |
| Accurate (Qwen3.5-2B) | ~2.3 GB |
| Accurate (Qwen3.5-4B) | ~3.5–4 GB (risky against ~6.4 GB free; swapping to the 5 GB page file would be catastrophic) |

### Site facts (verified 2026-09-26)
- Chapter URLs look like `https://mycomic.com/chapters/<id>`. Pages are `<img class="page ...">` elements, lazy-loaded by the `lozad` library, which moves `data-src` into `src`.
- Images come from `biccam.com`. The host **returns 403 without `Referer: https://mycomic.com/`** and sends **no CORS headers**. So the extension fetches images from its background worker, using a DNR rule to set the Referer.
- The page language is `zh-Hant`.

## Architecture

```
extension (MV3)                               local server (Python, 127.0.0.1:8765)
───────────────                               ─────────────────────────────────────
content.js   watches img.page, prefetches     POST /translate?tier=...
background.js fetch image (+Referer rule) ──► ├─ cache lookup (sha1 of image bytes, SQLite)
                                              ├─ OCR: RapidOCR (onnxruntime) det + rec
content.js   canvas render, swap img.src ◄──  ├─ group lines → bubbles
popup        on/off toggle, tier, status      └─ Translator[tier] → [{box, src, dst}]
                                                   └─ LLM tiers: llama-server subprocess
```

**Why one local Python server:**
- Two of the three translation tiers need llama.cpp anyway.
- Python is the fastest place to swap and benchmark models.

OCR could later move into the browser (ImageTrans shows that works) if the server turns out to be a burden.

### Compute device
The server detects the device once at startup and reports it in `/health`. Each runtime tries the GPU and drops to CPU if that fails. Falling back only on *failure* isn't enough on an integrated GPU, where the GPU can work but be slower, so the choice per component is also settled by benchmark (Phase 2) and saved as a per-machine default.

| Component | GPU path | Fallback | On the target (155H) |
|---|---|---|---|
| OCR (onnxruntime) | `onnxruntime-directml`: works on any DirectX 12 GPU (NVIDIA, AMD, Intel) with providers `["DmlExecutionProvider", "CPUExecutionProvider"]` | CPU provider, automatically | **CPU by default.** DirectML often loses on small models whose input width changes per line, and CPU OCR can run on the next page while the iGPU runs the LLM. |
| Very quick (CTranslate2) | `device="cuda"` when an NVIDIA GPU and CUDA libraries are present | `device="cpu"` if loading on CUDA throws | CPU (CTranslate2 has no Intel GPU support) |
| LLM tiers (llama.cpp) | NVIDIA found (`nvidia-smi`) → CUDA build; otherwise the Vulkan build (Intel/AMD GPUs). Start with `-ngl 99` (all layers on the GPU). | If the GPU start fails or runs out of memory, restart with `-ngl 0` on CPU | Arc iGPU via Vulkan; also benchmark the SYCL build |

- `atx/device.py` holds the detection. The rest of the code just asks it which device to use.
- The fallback is logged and shown in the popup's status line (e.g. "GPU: RTX 3070 Ti" or "CPU (GPU init failed)"), so a silent drop to CPU is visible.
- **CPU threads:** use the performance cores only (`-t 6` for llama.cpp on the 155H; 6 threads for CTranslate2 and onnxruntime). Efficiency and low-power cores tend to slow down work split evenly across all threads.
- **Free-memory check:** at startup and before each tier switch, the server reads free RAM. Below ~5 GB free, the accurate tier uses Qwen3.5-2B instead of 4B, and `/health` reports why.
- On an iGPU, check in Task Manager whether llama.cpp's Vulkan build keeps the weights twice (mapped file + GPU copy). If so, start it with `--no-mmap`.
- The Meteor Lake NPU (~11 TOPS) is not worth targeting; it would be slower than the iGPU for all of this.

**Why not use ImageTrans:** it's GPL-3.0 and a 6,000-line general-purpose tool. We only take ideas from it:
- sampling the fill color from a thin band just outside each box;
- two-pass line/block merging;
- caching results by image hash.

## Model choices

### Detection: not the bottleneck
PP-OCR's own text-line detector takes ~130 ms on CPU for a ~690×1500 page when its input is capped at 960px (measured in Phase 1). A separate bubble detector would **not** make things faster. It would only make them *better*:
- more correct grouping of lines into bubbles;
- skipping sound effects and signs;
- a fill shape that matches the bubble.

So v1 ships without one. The bubble detector is a Phase 5 experiment. Candidates, instead of the older YOLOv8 models:

| Candidate | Notes |
|---|---|
| **None** (PP-OCR detector + heuristic grouping) | v1 baseline |
| **YOLO26n**, fine-tuned on a bubble dataset | Jan 2026. Doesn't need the NMS post-processing step; claims up to 43% faster CPU ONNX inference than YOLO11n. AGPL-3.0, which is fine for a personal prototype. |
| **RF-DETR Nano**, fine-tuned | Apache-2.0. One report puts it at ~180 ms on CPU at 320px, probably slower than YOLO26n. |
| **ogkalu/comic-text-and-bubble-detector** | Ready to use: RT-DETR-v2 r50, Apache-2.0, 43M params. Trained on manga, manhua and webtoon pages; classes are bubble / text in bubble / text outside bubble. Accurate but heavy for CPU. Useful as a teacher for labelling training data for YOLO26n. |

### OCR
Via RapidOCR + onnxruntime: no PaddlePaddle or PyTorch install.
- PP-OCRv6 detector tiny/small + recognizer tiny/small (the models ImageTrans ships);
- PP-OCRv5 mobile as the comparison point.

Pick whichever has the lowest character error rate on our Traditional-Chinese samples within budget.

**Chosen (Phase 1):** PP-OCRv6 small detector + small recognizer, detection capped at 960px, on CPU: 3.7% error, 100% bubble recall, ~250 ms per page on the dev machine. See the Phase 1 results.

### Translation: three tiers

| Tier | Model | Runtime | Size | How it translates |
|---|---|---|---|---|
| **Very quick** | Helsinki-NLP `opus-mt-zh-en` + OpenCC Traditional→Simplified | CTranslate2 int8 | ~80 MB | Every bubble on the page in one batch |
| **Quick** | Tencent **Hy-MT2-1.8B** (translation-specialized, handles zh-Hant, Apache-2.0) | llama.cpp `llama-server` | ~1.1 GB (Q4_K_M) | One request per bubble with the official prompt template, **all bubbles sent concurrently** (`llama-server -np 8`, ~512 context tokens per slot) so decoding is batched |
| **Accurate** | **Qwen3.5-2B** on the target; **Qwen3.5-4B** on a discrete GPU. Thinking turned off (Apache-2.0) | llama.cpp `llama-server` | ~1.3 GB / ~2.5–3 GB (Q4) | The whole page in one prompt, plus the previous page's text for consistent names and tone |

- Accurate-tier prompt layout: fixed instructions first so llama-server's prompt cache reuses them, then the previous page, then the current page's bubbles as JSON. The output is **only a JSON array of translated strings** (no echoed source, no keys), which cuts output tokens ~20%.
- Accurate-tier model choice: the 4B model would take ~17–25 s per page on the target, slower than reading, and ~3.5–4 GB of the ~6.4 GB free RAM. It stays the default on a discrete GPU. Phase 2 checks whether the 2B model's quality is good enough; if not, try Hy-MT2 with page context.
- Qwen3.5's small models use a newer hybrid attention design; confirm the Vulkan build runs all of it on the GPU rather than quietly handing ops back to the CPU.
- Fallback for the quick tier: HY-MT1.5-1.8B GGUF, if Hy-MT2 misbehaves on stock llama.cpp. Its model card mentions a special kernel from a llama.cpp PR for some quantizations; check this in Phase 2.
- Only one LLM is loaded at a time. Switching tiers restarts the `llama-server` process with a different GGUF. The Python server manages that process.

## Repo layout

```
PLAN.md
server/
  requirements.txt
  atx/
    ocr.py           # RapidOCR wrapper → [Line{box, text, score}]
    grouping.py      # lines → bubbles (reading order, vertical detection)
    translators/
      base.py        # Translator protocol: translate(bubbles, context) -> list[str]
      opus.py        # very quick
      llm.py         # quick + accurate (OpenAI-compatible client → llama-server)
    device.py        # detect GPU once; choose providers/builds; CPU fallback
    llm_process.py   # start/stop llama-server with the chosen GGUF
    cache.py         # SQLite: image sha1 + tier → result JSON
    pipeline.py      # bytes → regions
    app.py           # FastAPI
  bench/
    fetch_samples.py # download sample pages into samples/ (with Referer)
    bench_ocr.py
    bench_mt.py      # side-by-side HTML report + latency/RAM table
  tests/
  models/  samples/  cache.db   # gitignored
extension/
  manifest.json  background.js  content.js  render.js  popup.html  popup.js  popup.css
```

## Phases

### Phase 0: Skeleton and samples
- Replace the leftover `server/requirements.txt` and `.venv` with the new, lighter stack: `rapidocr`, `onnxruntime-directml`, `ctranslate2`, `sentencepiece`, `opencc`, `fastapi`, `uvicorn`, `httpx`, `pillow`, `numpy`.
- Write `device.py` and a `python -m atx.device` check that prints what it detected and which paths it will use.
- Add a `.gitignore`.
- Write `fetch_samples.py`: about 20 pages from 3–4 chapters of the target series, for local testing only.
  - The chapter pages are behind Cloudflare bot protection, so the script doesn't scrape them. It downloads from `samples/urls.json`, a list of image URLs collected while viewing chapters in a browser (the script's docstring has the one-line console snippet). The image host itself only needs the Referer.
- Write the correct text for about 5 pages in `samples/truth.json`. Truth files are made with the highest-quality model available on the dev machine (Claude reading the page images), not by hand.

### Phase 1: OCR and grouping
- `ocr.py`, with a setting to pick the model set (v6-tiny, v6-small, v5-mobile).
- `grouping.py`:
  - grow each line box by about 0.6× the character size and merge boxes that touch (union-find);
  - reading order is top-to-bottom for horizontal text, right-to-left columns for vertical text;
  - drop results that are low-confidence or contain no Chinese characters.
- Debug CLI: `python -m atx.pipeline page.jpg --debug out.png` draws the boxes with their text.
- `bench_ocr.py` reports character error rate, bubbles found vs. true bubbles, time per page and peak RAM for each model set.
- Unit tests for grouping on synthetic boxes.

**Results (dev machine, 2026-09-27; 20 pages, truth = 5 pages / 30 text regions / 407 characters):**

| Config (detection size, device) | Error rate | Bubble recall | ms/page (mean / p90) | Peak MB |
|---|---|---|---|---|
| **v6-small @960 CPU** (chosen default) | **3.7%** | 100% | 253 / 432 | 353 |
| v6-small @960 DirectML (RTX 3070 Ti) | 3.7% | 100% | 110 / 209 | 599 |
| v6-small, full resolution, CPU | 3.2% | 100% | 734 / 942 | 591 |
| v6-small @720 CPU | 13.5% | 87% | 216 / 359 | 306 |
| v6-small-tiny @960 CPU | 11.8% | 100% | 176 / 224 | 248 |
| v6-tiny @960 CPU | 20.1% | 87% | 198 / 380 | 236 |
| v5-mobile @960 CPU | 9.6% | 93% | 627 / 1015 | 328 |
| cht-v3 (dedicated Traditional recognizer) @960 CPU | 28.5% | 70% | 278 / 551 | 251 |
| v6-medium, v5-server | not run: ~30–50 s per page on CPU | | | |

Findings:
- **The comic uses vertical text.** Almost all dialogue is in right-to-left columns; recap pages use horizontal narration boxes. PP-OCRv6 reads both.
- **Detection size is the main speed lever.** Capping the detection input at 960px (longest side) makes it ~4× faster than full resolution for +0.5 points of error; 720px is too small (error jumps to 13.5%). RapidOCR's own "max" mode ignores the size setting, so `ocr.py` overrides its hook.
- **The tiny recognizer is what costs accuracy**, not the tiny detector. The old dedicated Traditional-Chinese model is the worst option; PP-OCRv6's multilingual model reads Traditional better.
- **Remaining errors in v6-small @960:** 9 of the 15 wrong characters are one narration line under the site's watermark logo; the rest are tilted text and bubbles cut off at page edges. Excluding the watermarked line, the error rate is ~1.5%.
- **Grouping** was tuned on the truth pages: columns within a bubble touch, separate bubbles are ~0.8 glyphs apart, so lines merge when boxes grown by 0.45× glyph size touch. Column order uses clustering, not rounding.
- **Speed vs. budget:** CPU mean 253 ms is under the 0.35 s target, but p90 (432 ms) is over. The 155H's performance cores are roughly comparable to the 12900H's, so expect similar numbers on the target; confirm there in Phase 2. Timing on the dev laptop is noisy (hybrid-core scheduling).
- **Open for later:** filter the site watermark (a fixed logo; its regions don't match the glyph size of nearby text), and try a detection size between 720 and 960 if the target needs more speed.

### Phase 1b: Normalize text before scoring OCR
Pages can print variant glyphs (e.g. the Japanese-style 説 for 說, common in manga translations). OCR may return either form, and so may the truth files, so an exact-character comparison counts correct reads as errors.
- Add `normalize(text)` in `bench/` and apply it to both OCR output and truth before computing character error rate:
  - fold full-width ASCII letters, digits and punctuation to half-width (`，`→`,`, `Ｃ`→`C`) with NFKC, but keep CJK punctuation such as `。` and `、`;
  - map a small table of variant characters to one canonical form (説→說, and others as they turn up in the samples);
  - strip whitespace.
- Scoring only: the pipeline still sends the raw OCR text to translation.
- Unit tests for the fold and the variant table.

### Phase 2: Translation tiers and comparison
- `opus.py`: CTranslate2 conversion script, OpenCC Traditional→Simplified, batching.
- `llm_process.py` + `llm.py`: download the llama.cpp Windows builds (CUDA and Vulkan) and GGUFs into `models/`, start them with the GPU-first/CPU-fallback logic, and write the prompts for the quick and accurate tiers.
  - Parse the accurate tier's JSON output defensively; fall back to one bubble at a time if it fails.
- `bench_mt.py`: run all tiers on the same OCR'd bubbles and write `bench/report.html`, showing the source and all three translations side by side, with latency, RAM, and the device each tier actually ran on.
  - Include Qwen3.5-2B and 4B for the accurate tier, and serial vs. parallel (`-np 8`) for the quick tier.
  - Compare llama.cpp Vulkan vs. SYCL builds on the Arc iGPU, and CPU vs. DirectML for OCR.
- **Run the benchmarks on the target Yoga**, not the dev machine. Also run a ~5-minute sustained pass to measure thermal slowdown, and one on battery.
- Replace the estimated budget table with measured numbers, and save the fastest device per component as the target's default.
- **Checkpoint:** you read the report and decide which tiers to keep, swap or retune.

### Phase 3: Server
- `POST /translate`:
  - input: image bytes as multipart, plus `tier`;
  - output: `{w, h, regions:[{box:[x0,y0,x1,y1], src, dst, vertical}]}`.
- `GET /health`: returns the loaded models, the current tier, and the device per component (GPU name or CPU, plus the fallback reason if there was one).
- Load models once and keep them warm. OCR runs behind a lock. Results are cached by (image sha1, tier).
- Cold-start and warm latency are logged for each request.

### Phase 4: Extension
- **manifest:**
  - permissions: `storage`, `declarativeNetRequestWithHostAccess`;
  - host permissions: `*://mycomic.com/*`, `*://*.biccam.com/*`, `http://127.0.0.1/*`;
  - the content script runs on `mycomic.com/chapters/*`.
- **background.js:**
  - a fixed session DNR rule that sets `Referer: https://mycomic.com/` for requests to `biccam.com` coming from the extension itself;
  - fetch the image bytes, POST them to the server, and return the regions plus the image as base64.
- **content.js:**
  - on page load, read every `img.page`'s URL (`data-src || src`). All pages' URLs are in the HTML from the start (verified on 4 chapters with 23–28 pages each), so the whole chapter's queue is known upfront and prefetching doesn't wait for lazy loading or scrolling;
  - an IntersectionObserver tracks which page is on screen; the queue translates the next pages in order from there. A MutationObserver watches for `data-src` → `src` swaps so the translated image can be applied once lozad loads the real one;
  - process in page order, at most 2 at a time, and prefetch at most ~2 pages ahead (sustained load costs heat and battery on the target);
  - release translated page images with `URL.revokeObjectURL` once they're far off screen, to keep Chrome's memory down.
- **render.js:**
  - draw the original onto a canvas;
  - for each region, fill a rounded rect with the median color of the band just outside the box, and pick black or white text by brightness;
  - find the largest font that fits with word wrap, and widen narrow vertical boxes (staying inside the image) when needed;
  - `toBlob` → swap `img.src`, keeping the original so it can be restored.
- **popup (the extension's main page):**
  - a **toggle switch** (on/off, saved in `chrome.storage`);
  - a tier picker (very quick / quick / accurate);
  - a server status dot, plus the device in use (GPU name or CPU);
  - a "show original" button;
  - dev only: a "copy page URLs" button that copies the chapter's list in `samples/urls.json` format. It replaces the console snippet for collecting test pages.

### Phase 5: Experiments (pick based on results)
- **Bubble detector:** fine-tune YOLO26n, using the ogkalu model to auto-label pages, then compare grouping accuracy and fill quality against the heuristic.
- **Run OCR in the browser** (onnxruntime-web), so the server only translates.
- **Speed:** smaller quants (Hy-MT 2-bit / 1.25-bit), speculative prefetch of the next chapter, reusing the llama.cpp prompt cache for the accurate tier.
- **Better fill:** use the bubble mask from the detector instead of a rectangle.

## Risks
- Hy-MT2 GGUF may need a newer or patched llama.cpp for some quantizations. Use HY-MT1.5 as the fallback.
- PP-OCR on stylized comic fonts and vertical Traditional text needs measuring; the Phase 1 benchmark decides.
- **Free RAM on the target (~6.4 GB).** If Chrome plus the server exceed it, Windows swaps to the page file and a page can take minutes. Mitigations: the free-memory check that picks the 2B model, `--no-mmap` if weights are duplicated, and revoking off-screen page images.
- On the target, even with the iGPU, the accurate tier with Qwen3.5-4B (~17–25 s per page) is slower than reading. The target therefore defaults to Qwen3.5-2B for that tier; if its quality is too poor, try Hy-MT2 with page context. The 4B model stays discrete-GPU-only.
- Thermal throttling and battery: the 155H sustains ~28 W. Measured numbers may drop 15–25% after a few minutes of prefetching, and further on battery.
- GPU fallback edge cases: VRAM too small for the model, or CUDA DLLs missing for CTranslate2. Each one must fall back to CPU cleanly rather than crash; Phase 2 tests this by forcing a GPU failure.
- The site's markup or hotlink rules may change. Keep the site-specific selectors and Referer in one config object.
