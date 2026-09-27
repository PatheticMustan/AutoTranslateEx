"""RapidOCR wrapper: page image -> text lines.

Model sets are named combinations of a detection and a recognition model, so the
benchmark can compare them by name. Runs on CPU by default (the target's default,
see PLAN.md "Compute device"); pass device="dml" to try DirectML.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rapidocr import LangDet, LangRec, ModelType, OCRVersion, RapidOCR

from atx import device as devmod

V4, V5, V6 = OCRVersion.PPOCRV4, OCRVersion.PPOCRV5, OCRVersion.PPOCRV6

# name -> (det version, det size, rec version, rec size, rec language)
MODEL_SETS: dict[str, tuple] = {
    "v6-tiny":       (V6, ModelType.TINY,   V6, ModelType.TINY,   LangRec.CH),
    "v6-small-tiny": (V6, ModelType.SMALL,  V6, ModelType.TINY,   LangRec.CH),  # ImageTrans' default
    "v6-small":      (V6, ModelType.SMALL,  V6, ModelType.SMALL,  LangRec.CH),
    "v6-medium":     (V6, ModelType.MEDIUM, V6, ModelType.MEDIUM, LangRec.CH),
    "v5-mobile":     (V5, ModelType.MOBILE, V5, ModelType.MOBILE, LangRec.CH),
    "v5-server":     (V5, ModelType.SERVER, V5, ModelType.SERVER, LangRec.CH),
    # Dedicated Traditional-Chinese recognizer (older PP-OCRv3 generation).
    "cht-v3":        (V5, ModelType.MOBILE, V4, ModelType.MOBILE, LangRec.CHINESE_CHT),
}
DEFAULT_MODEL_SET = "v6-small"
# Detection input size: the page is scaled so its longest side is this many px.
# Glyphs on the target site are ~50px wide, so detection doesn't need full size.
# 0 keeps RapidOCR's default (short side scaled up to 736px).
DEFAULT_DET_SIDE = 960


@dataclass
class Line:
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 (axis-aligned, image pixels)
    text: str
    score: float

    @property
    def w(self) -> int:
        return self.box[2] - self.box[0]

    @property
    def h(self) -> int:
        return self.box[3] - self.box[1]

    @property
    def vertical(self) -> bool:
        """A column of text. Single characters are square-ish, so they count as neither."""
        return len(self.text) > 1 and self.h > 1.5 * self.w

    @property
    def horizontal(self) -> bool:
        return len(self.text) > 1 and self.w > 1.5 * self.h

    @property
    def char_size(self) -> float:
        """Approximate glyph size: the column width, or the line height."""
        if self.vertical:
            return self.w
        if self.horizontal:
            return self.h
        return min(self.w, self.h)


@dataclass
class OcrResult:
    lines: list[Line]
    seconds: dict[str, float] = field(default_factory=dict)  # det / rec / total


class Ocr:
    def __init__(self, model_set: str = DEFAULT_MODEL_SET, device: str = "cpu",
                 threads: int = devmod.CPU_THREADS, det_side: int = DEFAULT_DET_SIDE):
        if model_set not in MODEL_SETS:
            raise ValueError(f"unknown model set {model_set!r}; choose from {list(MODEL_SETS)}")
        det_ver, det_size, rec_ver, rec_size, rec_lang = MODEL_SETS[model_set]
        params = {
            "Global.log_level": "warning",
            "Global.use_cls": False,  # comic pages are upright
            "EngineConfig.onnxruntime.intra_op_num_threads": threads,
            "EngineConfig.onnxruntime.use_dml": device == "dml",
            "Det.ocr_version": det_ver,
            "Det.model_type": det_size,
            "Det.lang_type": LangDet.CH,
            "Rec.ocr_version": rec_ver,
            "Rec.model_type": rec_size,
            "Rec.lang_type": rec_lang,
        }
        self.model_set = model_set
        self.device = device
        self.det_side = det_side
        self._engine = RapidOCR(params=params)
        if det_side:
            self._cap_detection_size(det_side)

    def _cap_detection_size(self, side: int) -> None:
        """Run detection on the page scaled down to `side` px on its longest edge.

        RapidOCR's "max" mode ignores limit_side_len (it picks 960/1500/2000 from
        the image size, so these pages are never shrunk), so replace its hook.
        Recognition still crops from the full-resolution page.
        """
        from rapidocr.ch_ppocr_det.utils import DetPreProcess

        det = self._engine.text_det
        det.get_preprocess = lambda max_wh: DetPreProcess(side, "max", det.mean, det.std)

    def __call__(self, image: str | Path | bytes | np.ndarray) -> OcrResult:
        t0 = time.perf_counter()
        r = self._engine(image)
        total = time.perf_counter() - t0
        lines = []
        if r.boxes is not None:
            for poly, text, score in zip(r.boxes, r.txts, r.scores):
                x0, y0 = poly.min(0)
                x1, y1 = poly.max(0)
                lines.append(Line((int(x0), int(y0), int(x1), int(y1)), text, float(score)))
        # elapse_list is [det, cls, rec]; cls is disabled.
        det, _, rec = (list(r.elapse_list or []) + [0, 0, 0])[:3]
        return OcrResult(lines, {"det": det or 0.0, "rec": rec or 0.0, "total": total})
