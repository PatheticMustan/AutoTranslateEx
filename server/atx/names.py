"""Per-series name bank: character names found in the text, each with one fixed
English spelling, so a name reads the same on every page.

Names are found with jieba's person-name tag (nr, nrfg, nrt) on a Simplified
copy of the text, and mapped back to the Traditional original by position.
A name joins the bank once it shows up on MIN_PAGES different pages, which
filters out most one-off false hits. Its English is pinyin: "Surname Given"
when the first character is a common surname (黎玥 -> Li Yue), "Xiao Hong" for
小/阿/老 nicknames, one word otherwise (妮卡 -> Nika). Spellings marked as
set by the user are never changed.
"""

from __future__ import annotations

import logging
import re
import threading
from functools import cache

log = logging.getLogger("atx")

MIN_PAGES = 2
NAME_TAGS = {"nr", "nrfg", "nrt"}
CJK_ONLY = re.compile(r"^[㐀-䶿一-鿿豈-﫿]{2,4}$")
# Common Chinese surnames (Traditional forms), for "Surname Given" spelling.
SURNAMES = set(
    "王李張劉陳楊黃趙吳周徐孫馬朱胡郭何高林羅鄭梁謝宋唐許韓馮鄧曹彭曾蕭田董袁潘于蔣蔡余杜葉程蘇魏呂丁任沈"
    "姚盧姜崔鍾譚陸汪范金石廖賈夏韋付方白鄒孟熊秦邱江尹薛閻段雷侯龍史陶黎賀顧毛郝龔邵萬錢嚴覃武戴莫孔向湯"
    "柯池宮花游簡溫藍連施洪溫柳殷莊利里靳")
NICKNAME_PREFIXES = {"小": "Xiao", "阿": "A", "老": "Lao"}
TITLE_SUFFIXES = {"叔": "Uncle", "伯": "Uncle", "姨": "Auntie"}  # 森叔 -> Uncle Sen
# Characters that start a phrase, not a name: segmentation leftovers like 了黎.
PARTICLES = set("了的是在和跟把被給對讓叫說找向與也都就還又這那你我他她")
# Words jieba often tags as names that aren't (honorifics, common words).
NOT_NAMES = {"學姐", "學長", "學妹", "學弟", "老師", "同學", "社長", "會長", "姐姐", "哥哥", "妹妹", "弟弟",
             "媽媽", "爸爸", "大家", "小姐", "先生", "老大", "小鬼", "阿姨"}


@cache
def _tools():
    import jieba
    import jieba.posseg
    import opencc

    jieba.setLogLevel(logging.WARNING)
    return jieba.posseg, opencc.OpenCC("t2s")


def find_names(text: str) -> list[str]:
    """Person names in a Traditional-Chinese text, in their original form."""
    posseg, t2s = _tools()
    simple = t2s.convert(text)
    if len(simple) != len(text):  # can't map positions back
        return []
    out, pos = [], 0
    for word, flag in posseg.cut(simple):
        original = text[pos:pos + len(word)]
        pos += len(word)
        # jieba's name tag also lands on ordinary words (谢谢 "thanks", 明白
        # "understand"); those are in its dictionary, while real character names
        # aren't, so only out-of-vocabulary "names" count.
        if flag in NAME_TAGS and CJK_ONLY.match(original) and original not in NOT_NAMES                 and original[0] not in PARTICLES and not _dictionary_word(word):
            out.append(original)
    return out


def _common_word(text: str) -> bool:
    """A word the model would also capitalize without it being a name: an
    interjection (哈哈 "Haha", 叮咚 "Ding dong") or a frequent word (利亞, which
    the model writes "Liya" for 利亞繪). Rare dictionary nouns stay: nicknames
    like 花花 are in the dictionary too."""
    import jieba

    _, t2s = _tools()
    simplified = t2s.convert(text)
    tag = jieba.posseg.dt.word_tag_tab.get(simplified)
    return (jieba.dt.FREQ.get(simplified) or 0) >= 50 or (tag is not None and not tag.startswith("n"))


def _dictionary_word(simplified: str) -> bool:
    import jieba

    return bool(jieba.dt.FREQ.get(simplified)) or simplified in jieba.posseg.dt.word_tag_tab


CAPITALIZED_RUN = re.compile(r"\b[A-Z][a-z]*(?:[ -][A-Z][a-z]*)*\b")


def names_from_translation(src: str, dst: str) -> list[str]:
    """Names the translator spelled out in pinyin: a capitalized word or phrase in
    the English whose letters equal the pinyin of 2-4 source characters
    ("Li Yue" <-> 黎玥). jieba misses rare-character names; this catches the ones
    the model recognized, while garbled spellings simply don't match."""
    from pypinyin import Style, lazy_pinyin

    want = set()
    for m in CAPITALIZED_RUN.finditer(dst):  # "Xiao Hong", "Li Yue", "Nika"; every sub-run of up to 3 words
        words = re.split(r"[ -]", m.group())
        for i in range(len(words)):
            for j in range(i + 1, min(len(words), i + 3) + 1):
                want.add("".join(words[i:j]).lower())
    if not want:
        return []
    found = []
    for n in (2, 3, 4):
        for i in range(len(src) - n + 1):
            chunk = src[i:i + n]
            if CJK_ONLY.match(chunk) and chunk not in NOT_NAMES \
                    and "".join(lazy_pinyin(chunk, style=Style.NORMAL)) in want and not _common_word(chunk):
                found.append(chunk)
    # A shorter match inside a longer one (玥 in 黎玥) is the same name.
    return [f for f in found if not any(f != g and f in g for g in found)]


def romanize(name: str) -> str:
    from pypinyin import Style, lazy_pinyin

    syl = lazy_pinyin(name, style=Style.NORMAL)
    if name[-1] in TITLE_SUFFIXES and len(name) >= 2:
        return f"{TITLE_SUFFIXES[name[-1]]} {romanize(name[:-1]) if len(name) > 2 else syl[0].capitalize()}"
    if name[0] in NICKNAME_PREFIXES and len(name) >= 2:
        return f"{NICKNAME_PREFIXES[name[0]]} {''.join(syl[1:]).capitalize()}"
    if name[0] in SURNAMES and len(name) >= 2:
        return f"{syl[0].capitalize()} {''.join(syl[1:]).capitalize()}"
    return "".join(syl).capitalize()


class NameBank:
    """Backed by the server's SQLite cache (atx.cache.Cache)."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS names (series TEXT, zh TEXT, en TEXT, user_set INTEGER DEFAULT 0,
                                      PRIMARY KEY (series, zh));
    CREATE TABLE IF NOT EXISTS name_pages (series TEXT, zh TEXT, page TEXT, PRIMARY KEY (series, zh, page));
    """

    def __init__(self, cache):
        self._cache = cache
        self._lock = threading.Lock()
        with cache._lock:
            cache._db.executescript(self.SCHEMA)

    def _sql(self, sql: str, args: tuple = ()) -> list[tuple]:
        with self._cache._lock:
            return self._cache._db.execute(sql, args).fetchall()

    def observe(self, series: str, page: str, texts: list[str]) -> None:
        """Record the names jieba finds on a page (idempotent per page)."""
        self.observe_names(series, page, [n for t in texts for n in find_names(t)])

    def observe_names(self, series: str, page: str, names: list[str]) -> None:
        for zh in set(names):
            self._sql("INSERT OR IGNORE INTO name_pages VALUES (?,?,?)", (series, zh, page))
            self._sql("INSERT OR IGNORE INTO names (series, zh, en) VALUES (?,?,?)", (series, zh, romanize(zh)))

    def active(self, series: str) -> dict[str, str]:
        """zh -> en for names seen on enough pages, or set by the user."""
        rows = self._sql("""
            SELECT n.zh, n.en FROM names n
            WHERE n.series = ? AND (n.user_set = 1 OR
                  (SELECT COUNT(*) FROM name_pages p WHERE p.series = n.series AND p.zh = n.zh) >= ?)""",
                         (series, MIN_PAGES))
        return dict(rows)

    def all(self, series: str) -> list[dict]:
        rows = self._sql("""
            SELECT n.zh, n.en, n.user_set,
                   (SELECT COUNT(*) FROM name_pages p WHERE p.series = n.series AND p.zh = n.zh)
            FROM names n WHERE n.series = ? ORDER BY 4 DESC""", (series,))
        return [{"zh": zh, "en": en, "user_set": bool(u), "pages": c} for zh, en, u, c in rows]

    def set(self, series: str, zh: str, en: str | None) -> None:
        """Fix a spelling (en), or with en=None remove the name and stop using it."""
        if en is None:
            self._sql("DELETE FROM names WHERE series=? AND zh=?", (series, zh))
            self._sql("INSERT OR REPLACE INTO names (series, zh, en, user_set) VALUES (?,?,?,1)", (series, zh, ""))
        else:
            self._sql("INSERT OR REPLACE INTO names (series, zh, en, user_set) VALUES (?,?,?,1)", (series, zh, en))


def terms_for(text: str, bank: dict[str, str]) -> list[tuple[str, str]]:
    """The bank entries appearing in a text, longest first, without overlaps."""
    out, taken = [], []
    for zh in sorted(bank, key=len, reverse=True):
        if not bank[zh]:  # removed by the user
            continue
        for m in re.finditer(re.escape(zh), text):
            span = (m.start(), m.end())
            if not any(a < span[1] and span[0] < b for a, b in taken):
                taken.append(span)
                if (zh, bank[zh]) not in out:
                    out.append((zh, bank[zh]))
    return out
