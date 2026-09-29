"""Translation tiers. `make()` builds one by name; LLM tiers start their llama-server."""

from __future__ import annotations

from atx import device as devmod
from atx.translators.base import Translator

# tier -> (llama model, parallel slots); None model = not an LLM tier
TIERS = {
    "very_quick": (None, 0),
    "quick": ("hy-mt2-1.8b", 8),
    "accurate": ("qwen3.5-4b", 1),       # discrete GPU default (see PLAN.md)
    "accurate-hymt": ("hy-mt2-1.8b", 8),  # target candidate: Hy-MT2 with preceding text as context
    "accurate-2b": ("qwen3.5-2b", 1),    # unreliable (PLAN.md, Phase 2 results)
}


MIN_VRAM_FOR_4B_MB = 6000


def resolve(tier: str) -> str:
    """Pick the machine's implementation of a tier.

    "accurate" means Qwen3.5-4B only on a discrete GPU with enough VRAM; elsewhere
    (the target's integrated GPU, or CPU) it's Hy-MT2 with context, which is
    reliable and ~2x faster (PLAN.md, Phase 2 results).
    """
    if tier != "accurate":
        return tier
    gpu = devmod.detect().primary_gpu
    discrete = gpu is not None and gpu.vendor in ("nvidia", "amd") and (gpu.vram_mb or 0) >= MIN_VRAM_FOR_4B_MB
    return "accurate" if discrete else "accurate-hymt"


def make(tier: str, *, model: str | None = None, parallel: int | None = None) -> Translator:
    """Build a tier by name. Use resolve() first to get the machine's default for "accurate"."""
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; choose from {list(TIERS)}")
    default_model, default_parallel = TIERS[tier]
    if default_model is None:
        from atx.translators.opus import OpusTranslator
        return OpusTranslator()

    from atx.llm_process import LlamaServer
    from atx.translators.llm import HyMtTranslator, QwenPageTranslator

    ctx_per_slot = {"quick": 512, "accurate-hymt": 1024}.get(tier, 4096)
    server = LlamaServer(model or default_model, parallel=parallel or default_parallel,
                         ctx_per_slot=ctx_per_slot)
    server.start()
    if tier in ("quick", "accurate-hymt"):
        translator = HyMtTranslator(server, use_context=tier == "accurate-hymt")
    else:
        translator = QwenPageTranslator(server)
    translator.name = tier
    close = translator.close

    def close_all() -> None:
        close()
        server.stop()

    translator.close = close_all
    return translator
