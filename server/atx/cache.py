"""SQLite cache in server/cache.db.

- ocr:     (image sha1, OCR config) -> bubbles, so switching tiers skips OCR;
- results: (image sha1, result key) -> the /translate response;
- pages:   page URL -> image sha1, so a request can name its previous page by URL
           and get that page's text as translation context.

Keys include the model names, so changing a model misses the cache instead of
returning stale results. Bump PIPELINE_VERSION when grouping or prompts change.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

PIPELINE_VERSION = 4  # 2: watermark filter, with_context flag; 3: detector frames + recovered text;
#                       4: glyph size and OCR score per bubble
DB = Path(__file__).resolve().parent.parent / "cache.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ocr (sha1 TEXT, key TEXT, bubbles TEXT, PRIMARY KEY (sha1, key));
CREATE TABLE IF NOT EXISTS results (sha1 TEXT, key TEXT, result TEXT, created REAL, PRIMARY KEY (sha1, key));
CREATE TABLE IF NOT EXISTS pages (url TEXT PRIMARY KEY, sha1 TEXT);
"""


class Cache:
    def __init__(self, path: Path | str = DB):
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_SCHEMA)
        self._lock = threading.Lock()

    def _one(self, sql: str, args: tuple):
        with self._lock:
            row = self._db.execute(sql, args).fetchone()
        return row[0] if row else None

    def _put(self, sql: str, args: tuple) -> None:
        with self._lock:
            self._db.execute(sql, args)

    def get_ocr(self, sha1: str, key: str) -> list[dict] | None:
        raw = self._one("SELECT bubbles FROM ocr WHERE sha1=? AND key=?", (sha1, key))
        return json.loads(raw) if raw else None

    def put_ocr(self, sha1: str, key: str, bubbles: list[dict]) -> None:
        self._put("INSERT OR REPLACE INTO ocr VALUES (?,?,?)", (sha1, key, json.dumps(bubbles, ensure_ascii=False)))

    def get_result(self, sha1: str, key: str) -> dict | None:
        raw = self._one("SELECT result FROM results WHERE sha1=? AND key=?", (sha1, key))
        return json.loads(raw) if raw else None

    def put_result(self, sha1: str, key: str, result: dict) -> None:
        self._put("INSERT OR REPLACE INTO results VALUES (?,?,?,?)",
                  (sha1, key, json.dumps(result, ensure_ascii=False), time.time()))

    def page_sha1(self, url: str) -> str | None:
        return self._one("SELECT sha1 FROM pages WHERE url=?", (url,))

    def put_page(self, url: str, sha1: str) -> None:
        self._put("INSERT OR REPLACE INTO pages VALUES (?,?)", (url, sha1))

    def clear(self) -> None:
        with self._lock:
            self._db.executescript("DELETE FROM ocr; DELETE FROM results; DELETE FROM pages;")

    def close(self) -> None:
        self._db.close()
