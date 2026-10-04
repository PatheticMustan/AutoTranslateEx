"""Very quick tier: Helsinki-NLP opus-mt-zh-en on CTranslate2 (int8).

The model was trained mostly on Simplified Chinese, so the Traditional source is
converted with OpenCC first. All bubbles on a page go through in one batch.
"""

from __future__ import annotations

import ctranslate2
import opencc
import sentencepiece as spm

from atx import device as devmod
from atx.models import model_path
from atx.names import terms_for

MODEL = "opus-mt-zh-en"


class OpusTranslator:
    name = "very_quick"

    def __init__(self, threads: int = devmod.CPU_THREADS, beam_size: int = 4):
        path = model_path(MODEL)
        if not (path / "model.bin").exists():
            raise FileNotFoundError(f"{path} missing; run `python -m atx.models {MODEL}`")
        self._t2s = opencc.OpenCC("t2s")
        self._src = spm.SentencePieceProcessor(model_file=str(path / "source.spm"))
        self._tgt = spm.SentencePieceProcessor(model_file=str(path / "target.spm"))
        self._beam = beam_size
        self._model, self.device = self._load(str(path), threads)

    def _load(self, path: str, threads: int) -> tuple[ctranslate2.Translator, str]:
        if devmod.detect().ct2_device == "cuda":
            try:
                model = ctranslate2.Translator(path, device="cuda", compute_type="int8_float16")
                # Loading succeeds even without cuBLAS; the DLL is only needed once
                # real tokens are decoded, so probe with a short phrase first.
                probe = self._src.encode("你好", out_type=str) + ["</s>"]
                model.translate_batch([probe], beam_size=self._beam, max_decoding_length=8)
                return model, "cuda"
            except Exception as e:  # e.g. "Library cublas64_12.dll is not found"
                devmod.record_fallback("very_quick", f"CUDA unusable: {e}")
        return ctranslate2.Translator(path, device="cpu", compute_type="int8", intra_threads=threads), "cpu"

    def translate(self, texts: list[str], context: list[str] | None = None,
                  glossary: dict[str, str] | None = None) -> list[str]:
        self.last_scores = []
        if not texts:
            return []
        # No prompt to put names in, so put the English name straight into the
        # source; the model copies Latin words through.
        sources = []
        for t in texts:
            for zh, en in terms_for(t, glossary or {}):
                t = t.replace(zh, f" {en} ")
            sources.append(t)
        batch = [self._src.encode(self._t2s.convert(t), out_type=str) + ["</s>"] for t in sources]
        results = self._model.translate_batch(batch, beam_size=self._beam, max_decoding_length=200,
                                              return_scores=True)  # length-normalized (length_penalty=1)
        # The score is the mean token log-probability, as for the LLM tiers (no min here).
        self.last_scores = [{"mean": r.scores[0], "min": None} for r in results]
        return [self._tgt.decode([tok for tok in r.hypotheses[0] if tok != "</s>"]) for r in results]

    def close(self) -> None:
        del self._model
