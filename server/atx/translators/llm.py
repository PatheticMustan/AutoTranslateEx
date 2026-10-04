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
# Tencent's contextual template: preceding text first, then "translate only the
# text below, not the text above, without explanation".
HY_MT_CONTEXT_PROMPT = "{context}\n参考上面的信息，把下面的文本翻译成英语，注意不需要翻译上文，也不要额外解释：\n{text}"
HY_MT_CONTEXT_BUBBLES = 8  # how many preceding bubbles to show as context
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


# English runs ~2.5-4 characters per Chinese character; leaked or run-on output is
# far longer (a lone 嗶 "beep" answered with the previous four bubbles, ~180 chars).
MAX_CHARS_PER_SOURCE_CHAR = 6
LENGTH_SLACK = 30


def run_on(src: str, dst: str) -> bool:
    """Output that can't be just this bubble's translation: far longer than the
    source, or several lines from a one-line source (merged or leaked context)."""
    if len(dst) > MAX_CHARS_PER_SOURCE_CHAR * len(src) + LENGTH_SLACK:
        return True
    return "\n" in dst.strip() and "\n" not in src


def bad_output(src: str, dst: str) -> bool:
    return not dst.strip() or untranslated(dst) or run_on(src, dst)


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
    """One request per bubble. With use_context, each bubble also gets the text
    before it (previous page, then earlier bubbles on this page) as context."""

    name = "quick"

    def __init__(self, server: LlamaServer, use_context: bool = False, sampling: dict | None = None):
        super().__init__(server)
        self.use_context = use_context
        self.sampling = HY_MT_SAMPLING if sampling is None else sampling

    def translate(self, texts: list[str], context: list[str] | None = None) -> list[str]:
        self.last_usage = []
        preceding = list(context or [])

        def prompt(i: int) -> str:
            before = (preceding + texts[:i])[-HY_MT_CONTEXT_BUBBLES:]
            if self.use_context and before:
                return HY_MT_CONTEXT_PROMPT.format(context="\n".join(before), text=texts[i])
            return HY_MT_PROMPT.format(text=texts[i])

        def ask(content: str) -> str:
            return self._chat([{"role": "user", "content": content}], max_tokens=160, **self.sampling)

        def one(i: int) -> str:
            out = ask(prompt(i))
            if self.use_context and bad_output(texts[i], out):
                # Short bubbles (sound effects, one-word replies) sometimes get the
                # preceding context translated instead; retry without it.
                retry = ask(HY_MT_PROMPT.format(text=texts[i]))
                if not bad_output(texts[i], retry):
                    return retry
            return out

        with ThreadPoolExecutor(max_workers=self.server.parallel) as pool:
            return list(pool.map(one, range(len(texts))))


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
        # The schema fixes the count, not the content: small models sometimes copy
        # the source back or put several bubbles into one. Retry those one at a time.
        for i, s in enumerate(out):
            if bad_output(texts[i], s):
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
