"""Translation tiers. `make()` builds one by name; LLM tiers start their llama-server."""

from __future__ import annotations

from atx.translators.base import Translator

# tier -> (llama model, parallel slots); None model = not an LLM tier
TIERS = {
    "very_quick": (None, 0),
    "quick": ("hy-mt2-1.8b", 8),
    "accurate": ("qwen3.5-4b", 1),     # discrete GPU default (see PLAN.md)
    "accurate-2b": ("qwen3.5-2b", 1),  # target default
}


def make(tier: str, *, model: str | None = None, parallel: int | None = None) -> Translator:
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; choose from {list(TIERS)}")
    default_model, default_parallel = TIERS[tier]
    if default_model is None:
        from atx.translators.opus import OpusTranslator
        return OpusTranslator()

    from atx.llm_process import LlamaServer
    from atx.translators.llm import HyMtTranslator, QwenPageTranslator

    server = LlamaServer(model or default_model, parallel=parallel or default_parallel,
                         ctx_per_slot=512 if tier == "quick" else 4096)
    server.start()
    cls = HyMtTranslator if tier == "quick" else QwenPageTranslator
    translator = cls(server)
    translator.name = tier
    close = translator.close

    def close_all() -> None:
        close()
        server.stop()

    translator.close = close_all
    return translator
