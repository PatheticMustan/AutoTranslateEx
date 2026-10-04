"""Speech-bubble and text detector.

ogkalu/comic-text-and-bubble-detector, the small int8 variant (10.6 MB), on CPU.
RT-DETR exported to ONNX with post-processing built in. Input: RGB scaled to
0-1, resized (not cropped) to 640x640, plus the original size. Output: boxes in
page pixels, labels, scores. Classes: 0 bubble, 1 text inside a bubble,
2 text outside bubbles.

Phase 5 comparison (dev machine, 20 sample pages):

| variant | CPU ms/page | DirectML ms/page |
|---|---|---|
| v4-s int8 (10.6 MB), whole page | 131 | won't load |
| int8 (42 MB), whole page | 350 | 99 |
| fp32 (161 MB), whole page | 817 | 60 |

Cutting tall pages into square tiles (as the model's training did for webtoons)
was 3-4x slower and worse: it merged neighbouring narration paragraphs and
added a false bubble. The whole page squashed to 640x640 works well.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from atx import device as devmod
from atx.models import model_path

CLASSES = ("bubble", "text_bubble", "text_free")
SIDE = 640
MODEL_FILE = "detector-v4-s_int8.onnx"


@dataclass
class Detection:
    box: tuple[int, int, int, int]  # x0, y0, x1, y1
    label: str
    score: float


def available() -> bool:
    return (model_path("comic-detector") / MODEL_FILE).exists()


class Detector:
    def __init__(self, threads: int = devmod.CPU_THREADS, min_score: float = 0.4):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        path = model_path("comic-detector") / MODEL_FILE
        # The int8 model doesn't load on DirectML; on CPU it's ~130 ms per page.
        self._session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.min_score = min_score
        self.last_ms = 0.0

    def __call__(self, image: str | Path | bytes | Image.Image) -> list[Detection]:
        if not isinstance(image, Image.Image):
            image = Image.open(io.BytesIO(image) if isinstance(image, bytes) else image)
        page = image.convert("RGB")
        t0 = time.perf_counter()
        w, h = page.size
        x = np.asarray(page.resize((SIDE, SIDE), Image.BILINEAR), dtype=np.float32) / 255.0
        labels, boxes, scores = self._session.run(None, {
            "images": x.transpose(2, 0, 1)[None],
            "orig_target_sizes": np.array([[w, h]], dtype=np.int64),
        })
        found = []
        for lab, box, s in zip(labels[0], boxes[0], scores[0]):
            if s >= self.min_score:
                x0, y0, x1, y1 = (int(round(v)) for v in box)
                found.append(Detection((max(0, x0), max(0, y0), min(w, x1), min(h, y1)), CLASSES[int(lab)], float(s)))
        self.last_ms = (time.perf_counter() - t0) * 1000
        return found
