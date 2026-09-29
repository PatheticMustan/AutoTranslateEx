"""The interface every translation tier implements."""

from __future__ import annotations

from typing import Protocol


class Translator(Protocol):
    name: str      # tier name, e.g. "very_quick"
    device: str    # what it actually runs on, e.g. "cuda", "cpu", "gpu (cuda build)"

    def translate(self, texts: list[str], context: list[str] | None = None) -> list[str]:
        """Translate one page's bubbles (in reading order) to English.

        `context` is the previous page's source bubbles, for tiers that use it.
        Returns one translation per input, in the same order.
        """
        ...

    def close(self) -> None:
        ...
