"""LLM tiers, talking to llama-server's OpenAI-compatible API.

Quick:    Hy-MT2-1.8B, one request per bubble with Tencent's prompt template; a
          page's bubbles are sent concurrently so llama-server batches decoding
          across its parallel slots (-np).
Accurate: Qwen3.5 (thinking off), the whole page in one request with the
          previous page as context. Output is constrained to a JSON array with
          exactly one string per bubble; if that still fails, it falls back to
          one bubble per request.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor

import httpx

from atx.llm_process import LlamaServer

# Tencent's Chinese-language template (used when Chinese is the source or target).
HY_MT_PROMPT = "将以下文本翻译为英语，注意只需要输出翻译后的结果，不要额外解释：\n\n{text}"
# Tencent's recommended sampling for the 1.8B model.
HY_MT_SAMPLING = {"temperature": 0.7, "top_p": 0.6, "top_k": 20, "repeat_penalty": 1.05}

# Fixed instructions first, so llama-server's prompt cache reuses them across pages.
QWEN_SYSTEM = """You translate Traditional Chinese comic speech bubbles into natural, casual English.
Rules:
- Translate each bubble on its own line of the array, in the same order. Never merge or split bubbles.
- Keep character names and forms of address consistent with the context.
- Keep it short enough to fit in a speech bubble; keep the tone (jokes, anger, sarcasm).
- Transliterate names with pinyin unless they have an obvious English form.
- Output only a JSON array of English strings."""
QWEN_SAMPLING = {"temperature": 0.3, "top_p": 0.8, "top_k": 20}

CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")


def untranslated(text: str) -> bool:
    """Mostly Chinese characters: the model echoed the source instead of translating."""
    return len(CJK.findall(text)) > 0.3 * max(len(text), 1)


class _LlmTranslator:
    name = "llm"

    def __init__(self, server: LlamaServer):
        self.server = server
        self._http = httpx.Client(base_url=server.url, timeout=httpx.Timeout(10, read=300))
        self.last_usage: list[dict] = []  # per request: prompt/completion tokens and speeds

    @property
    def device(self) -> str:
        return self.server.device

    def _chat(self, messages: list[dict], max_tokens: int, **extra) -> str:
        resp = self._http.post("/v1/chat/completions", json={
            "messages": messages, "max_tokens": max_tokens, "cache_prompt": True, **extra,
        })
        resp.raise_for_status()
        data = resp.json()
        timings = data.get("timings", {})
        self.last_usage.append({
            "prompt_tokens": data["usage"]["prompt_tokens"],
            "completion_tokens": data["usage"]["completion_tokens"],
            "prompt_tps": timings.get("prompt_per_second"),
            "gen_tps": timings.get("predicted_per_second"),
        })
        return data["choices"][0]["message"]["content"].strip()

    def close(self) -> None:
        self._http.close()


class HyMtTranslator(_LlmTranslator):
    name = "quick"

    def translate(self, texts: list[str], context: list[str] | None = None) -> list[str]:
        self.last_usage = []

        def one(text: str) -> str:
            return self._chat([{"role": "user", "content": HY_MT_PROMPT.format(text=text)}],
                              max_tokens=160, **HY_MT_SAMPLING)

        with ThreadPoolExecutor(max_workers=self.server.parallel) as pool:
            return list(pool.map(one, texts))


class QwenPageTranslator(_LlmTranslator):
    name = "accurate"

    def translate(self, texts: list[str], context: list[str] | None = None) -> list[str]:
        self.last_usage = []
        if not texts:
            return []
        try:
            out = self._page(texts, context)
            if len(out) != len(texts):
                raise ValueError("wrong number of translations")
        except (json.JSONDecodeError, TypeError, ValueError):
            # Fallback: one bubble per request, still with the page as context.
            return [self._page([t], context)[0] for t in texts]
        # The schema fixes the count, not the language: small models sometimes
        # copy a whole page's source back. Retry those bubbles one at a time.
        for i, s in enumerate(out):
            if untranslated(s):
                out[i] = self._page([texts[i]], context)[0]
        return out

    def _page(self, texts: list[str], context: list[str] | None) -> list[str]:
        parts = []
        if context:
            parts.append("Previous page, for context only (do not translate):\n"
                         + json.dumps(context, ensure_ascii=False))
        parts.append(f"Translate these {len(texts)} bubbles, in order:\n" + json.dumps(texts, ensure_ascii=False))
        schema = {"type": "array", "items": {"type": "string"}, "minItems": len(texts), "maxItems": len(texts)}
        content = self._chat(
            [{"role": "system", "content": QWEN_SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}],
            max_tokens=80 + 60 * len(texts),
            response_format={"type": "json_schema", "json_schema": {"name": "translations", "schema": schema}},
            chat_template_kwargs={"enable_thinking": False},
            **QWEN_SAMPLING,
        )
        out = json.loads(content)
        if not isinstance(out, list) or not all(isinstance(s, str) for s in out):
            raise ValueError("expected a JSON array of strings")
        return [s.strip() for s in out]
