"""Translation tiers. `make()` builds one by name; LLM tiers start their llama-server.

The server (atx.app) keeps one llama-server at a time and reuses it when the
next tier needs the same one (`server_spec`), so switching between quick and
accurate on Hy-MT2 doesn't restart anything.
"""

from __future__ import annotations

import psutil

from atx import device as devmod
from atx.translators.base import Translator

# tier -> (llama model, parallel slots, context tokens per slot); None model = not an LLM tier.
# Both Hy-MT2 tiers use the same slots, so they share one llama-server.
TIERS = {
    "very_quick": (None, 0, 0),
    "quick": ("hy-mt2-1.8b", 8, 1024),
    "accurate": ("qwen3.5-4b", 1, 4096),       # discrete GPU default (see PLAN.md)
    "accurate-hymt": ("hy-mt2-1.8b", 8, 1024),  # everywhere else: Hy-MT2 with preceding text as context
    "accurate-2b": ("qwen3.5-2b", 1, 4096),    # unreliable (PLAN.md, Phase 2 results)
}
PUBLIC_TIERS = ["very_quick", "quick", "accurate"]  # what the extension offers

MIN_VRAM_FOR_4B_MB = 6000
MIN_FREE_RAM_FOR_4B_MB = 5000


def resolve(tier: str) -> tuple[str, str | None]:
    """Pick the machine's implementation of a tier, and why it isn't the default one.

    "accurate" means Qwen3.5-4B only on a discrete GPU with enough VRAM and enough
    free RAM (llama-server reads the whole file through memory while loading);
    elsewhere (the target's integrated GPU, or CPU) it's Hy-MT2 with context,
    which is reliable and ~2x faster (PLAN.md, Phase 2 results).
    """
    if tier != "accurate":
        return tier, None
    gpu = devmod.detect().primary_gpu
    if gpu is None or gpu.vendor not in ("nvidia", "amd") or (gpu.vram_mb or 0) < MIN_VRAM_FOR_4B_MB:
        return "accurate-hymt", None  # the normal choice without a discrete GPU, not a fallback
    free_mb = psutil.virtual_memory().available // 2**20
    if free_mb < MIN_FREE_RAM_FOR_4B_MB:
        return "accurate-hymt", f"only {free_mb} MB RAM free (Qwen3.5-4B needs {MIN_FREE_RAM_FOR_4B_MB})"
    return "accurate", None


def server_spec(tier: str, parallel: int | None = None) -> tuple[str, int, int] | None:
    """(model, parallel, ctx_per_slot) of the llama-server a tier needs, or None."""
    model, default_parallel, ctx_per_slot = TIERS[tier]
    return None if model is None else (model, parallel or default_parallel, ctx_per_slot)


def on_server(tier: str, server) -> Translator:
    """Build an LLM tier's translator on an already started llama-server."""
    from atx.translators.llm import HyMtTranslator, QwenPageTranslator

    if TIERS[tier][0] == "hy-mt2-1.8b":
        translator = HyMtTranslator(server, use_context=tier == "accurate-hymt")
    else:
        translator = QwenPageTranslator(server)
    translator.name = tier
    return translator


def make(tier: str, *, parallel: int | None = None) -> Translator:
    """Build a tier by name, with its own llama-server that close() stops.
    Use resolve() first to get the machine's choice for "accurate"."""
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; choose from {list(TIERS)}")
    spec = server_spec(tier, parallel)
    if spec is None:
        from atx.translators.opus import OpusTranslator
        return OpusTranslator()

    from atx.llm_process import LlamaServer

    model, n_parallel, ctx_per_slot = spec
    server = LlamaServer(model, parallel=n_parallel, ctx_per_slot=ctx_per_slot)
    server.start()
    translator = on_server(tier, server)
    close = translator.close

    def close_all() -> None:
        close()
        server.stop()

    translator.close = close_all
    return translator
