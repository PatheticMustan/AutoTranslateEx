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
# A name that's followed by the same character on this share of its pages is a
# cut-off longer name: jieba finds 白智 in 白智晷, the model writes "Liya" for 利亞繪.
EXTEND_SHARE = 0.8
NAME_TAGS = {"nr", "nrfg", "nrt"}
CJK_ONLY = re.compile(r"^[㐀-䶿一-鿿豈-﫿]{2,4}$")
# Common Chinese surnames (Traditional forms), for "Surname Given" spelling.
SURNAMES = set(
    "王李張劉陳楊黃趙吳周徐孫馬朱胡郭何高林羅鄭梁謝宋唐許韓馮鄧曹彭曾蕭田董袁潘于蔣蔡余杜葉程蘇魏呂丁任沈"
    "姚盧姜崔鍾譚陸汪范金石廖賈夏韋付方白鄒孟熊秦邱江尹薛閻段雷侯龍史陶黎賀顧毛郝龔邵萬錢嚴覃武戴莫孔向湯"
    "柯池宮花游簡溫藍連施洪溫柳殷莊利里靳")
NICKNAME_PREFIXES = {"小": "Xiao", "阿": "A", "老": "Lao"}
# 森叔 -> Uncle Sen, 神哥 -> Brother Shen, 天海會 -> Tianhai Society, 愛班 -> Class Ai
TITLE_SUFFIXES = {"叔": "Uncle {}", "伯": "Uncle {}", "姨": "Auntie {}", "哥": "Brother {}", "姐": "Sister {}",
                  "媽": "Mrs. {}", "會": "{} Society", "班": "Class {}"}
# Characters that start a phrase, not a name: segmentation leftovers like 了黎.
PARTICLES = set("了的是在和跟把被給對讓叫說找向與也都就還又這那你我他她")
# Name-shaped strings (below) never contain these: pronouns, sentence-final
# particles and interjections (謝妳 "thank you", 錢嗎, 嗯嗯).
NOT_IN_NAMES = PARTICLES | set("妳們嗎呢吧啦啊呀哦喔欸嗯哈嘿呵嘻喀咩唉哎耶噢家大好嘛副")
NGRAM_MIN_PAGES = 3  # the weakest signal, so it needs more pages
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


@cache
def _s2t():
    import opencc

    return opencc.OpenCC("s2t")


# Valid in Traditional text too, though OpenCC's s2t maps them (里 -> 裡, 后 -> 後).
BOTH_SCRIPTS = set("里后台面松干云余谷系卷才只冲表")


def _simplified_only(text: str) -> bool:
    """Contains Simplified-only characters: in a Traditional comic that's the
    site's watermark (集云数据...), not a name."""
    return any(_s2t().convert(c) != c for c in text if c not in BOTH_SCRIPTS)


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
    return (jieba.dt.FREQ.get(simplified) or 0) >= 1000 or (tag is not None and not tag.startswith("n"))


def name_shaped(text: str) -> list[str]:
    """2-3 character strings shaped like names, for series where the translator
    never spells names out (opus-mt garbles them): starting with a common
    surname or a 小/阿 prefix, or a doubled character (妮妮), not a dictionary
    word, and not a prefix plus a common word (小心點, 池老師). Recurring ones
    become names; most noise doesn't recur."""
    _, t2s = _tools()
    out = set()
    for n in (2, 3):
        for i in range(len(text) - n + 1):
            g = text[i:i + n]
            if not CJK_ONLY.match(g) or g in NOT_NAMES or any(c in NOT_IN_NAMES for c in g):
                continue
            if not (g[0] in SURNAMES or g[0] in NICKNAME_PREFIXES or (n == 2 and g[0] == g[1])):
                continue
            simple = t2s.convert(g)
            if _dictionary_word(simple) or _simplified_only(g):
                continue
            if n == 3 and (_common(simple[:2]) or _common(simple[1:])):
                continue
            # The edge character belongs to a common word with its neighbour: 嚴同 in
            # 嚴同學, 池老 in 池老師, 熊副社 in 熊副社長.
            after, before = text[i + n:i + n + 1], text[i - 1:i] if i else ""
            if (after and _common(t2s.convert(g[-1] + after), 200))                     or (before and _common(t2s.convert(before + g[0]), 200)):
                continue
            out.add(g)
    return sorted(out)


def _rare_char(char: str) -> bool:
    """Rare on its own, and not a sound word or interjection (喵 "meow")."""
    import jieba

    _, t2s = _tools()
    simplified = t2s.convert(char)
    tag = jieba.posseg.dt.word_tag_tab.get(simplified) or ""
    return (jieba.dt.FREQ.get(simplified) or 0) < 1000 and tag not in ("o", "e", "y")


def _common(simplified: str, at_least: int = 1000) -> bool:
    import jieba

    return (jieba.dt.FREQ.get(simplified) or 0) >= at_least


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
        base = romanize(name[:-1]) if len(name) > 2 else syl[0].capitalize()
        return TITLE_SUFFIXES[name[-1]].format(base)
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
    -- the character after each occurrence ('' when none, or not part of a name)
    CREATE TABLE IF NOT EXISTS name_next (series TEXT, zh TEXT, page TEXT, next TEXT,
                                          PRIMARY KEY (series, zh, page, next));
    -- name-shaped strings (name_shaped), counted separately: a weaker signal
    CREATE TABLE IF NOT EXISTS name_shapes (series TEXT, zh TEXT, page TEXT, PRIMARY KEY (series, zh, page));
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
        """Record the names jieba finds on a page, and name-shaped strings
        (idempotent per page)."""
        self.observe_names(series, page, [n for t in texts for n in find_names(t)], texts)
        for g in {g for t in texts for g in name_shaped(t)}:
            self._sql("INSERT OR IGNORE INTO name_shapes VALUES (?,?,?)", (series, g, page))

    def observe_names(self, series: str, page: str, names: list[str], texts: list[str] = ()) -> None:
        for zh in {n for n in names if not _simplified_only(n)}:
            self._sql("INSERT OR IGNORE INTO name_pages VALUES (?,?,?)", (series, zh, page))
            self._sql("INSERT OR IGNORE INTO names (series, zh, en) VALUES (?,?,?)", (series, zh, romanize(zh)))
            for t in texts:
                for m in re.finditer(re.escape(zh), t):
                    nxt = t[m.end():m.end() + 1]
                    if not CJK_ONLY.match(zh + nxt) or nxt in NOT_IN_NAMES:
                        nxt = ""
                    self._sql("INSERT OR IGNORE INTO name_next VALUES (?,?,?,?)", (series, zh, page, nxt))

    def active(self, series: str) -> dict[str, str]:
        """zh -> en for names seen on enough pages, or set by the user. A name
        nearly always followed by the same character is replaced by the longer one."""
        rows = self._sql("""
            SELECT n.zh, n.en, n.user_set FROM names n
            WHERE n.series = ? AND (n.user_set = 1 OR
                  (SELECT COUNT(*) FROM name_pages p WHERE p.series = n.series AND p.zh = n.zh) >= ?)""",
                         (series, MIN_PAGES))
        user = dict(self._sql("SELECT zh, en FROM names WHERE series=? AND user_set=1", (series,)))
        out: dict[str, str] = {zh: romanize(zh) for zh in self._recurring_shapes(series)}
        for zh, en, user_set in rows:
            longer = None if user_set else self._extension(series, zh)
            if longer and longer not in user:
                out.setdefault(longer, romanize(longer))
            else:
                out[zh] = en
        out.update(user)
        # A nickname 小X / 阿X also teaches its bare X (宵 for 小宵), when X is a
        # rare character on its own; term_spans checks each use.
        for zh, en in list(out.items()):
            bare = zh[1:]
            if len(zh) == 2 and zh[0] in NICKNAME_PREFIXES and en and bare not in out and _rare_char(bare):
                out[bare] = romanize(bare)
        return out

    def _recurring_shapes(self, series: str) -> list[str]:
        """Name-shaped strings on enough pages. Of two nested ones, the longer
        wins only if it carries nearly all of the shorter one's pages (白智晷
        over 白智, but 黎玥 over 黎玥同)."""
        counts = dict(self._sql("""SELECT zh, COUNT(*) FROM name_shapes WHERE series=?
                                   GROUP BY zh HAVING COUNT(*) >= ?""", (series, NGRAM_MIN_PAGES)))
        keep = []
        for g, c in counts.items():
            if any(s != g and s in g and c < EXTEND_SHARE * counts[s] for s in counts):
                continue  # a longer string that's usually just the shorter name plus something
            if any(l != g and g in l and counts[l] >= EXTEND_SHARE * c for l in counts):
                continue  # cut off: the longer form is the name
            keep.append(g)
        return keep

    def _extension(self, series: str, zh: str) -> str | None:
        """zh + c when c follows zh on at least EXTEND_SHARE of its pages."""
        rows = self._sql("SELECT next, COUNT(DISTINCT page) FROM name_next WHERE series=? AND zh=? GROUP BY next",
                         (series, zh))
        pages = self._sql("SELECT COUNT(*) FROM name_pages WHERE series=? AND zh=?", (series, zh))[0][0]
        best = max(((n, c) for n, c in rows if n), key=lambda r: r[1], default=None)
        if best and len(zh) < 4 and best[1] >= MIN_PAGES and best[1] >= EXTEND_SHARE * pages:
            return zh + best[0]
        return None

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


def term_spans(text: str, bank: dict[str, str]) -> list[tuple[int, int, str, str]]:
    """(start, end, zh, en) for each bank entry in a text, longest entries
    first, without overlaps. A one-character entry (宵) only counts where it
    doesn't form a dictionary word with a neighbour (通宵 "all night")."""
    spans: list[tuple[int, int, str, str]] = []
    for zh in sorted(bank, key=len, reverse=True):
        if not bank[zh]:  # removed by the user
            continue
        for m in re.finditer(re.escape(zh), text):
            a, b = m.start(), m.end()
            if any(x < b and a < y for x, y, _, _ in spans):
                continue
            if len(zh) == 1 and not _standalone(text, a):
                continue
            spans.append((a, b, zh, bank[zh]))
    return sorted(spans)


def terms_for(text: str, bank: dict[str, str]) -> list[tuple[str, str]]:
    """The bank entries appearing in a text (see term_spans), each once."""
    out = []
    for _, _, zh, en in term_spans(text, bank):
        if (zh, en) not in out:
            out.append((zh, en))
    return out


def substitute(text: str, bank: dict[str, str]) -> str:
    """The text with each bank entry replaced by its English (for opus-mt)."""
    for a, b, _, en in reversed(term_spans(text, bank)):
        text = f"{text[:a]} {en} {text[b:]}"
    return text


def _standalone(text: str, i: int) -> bool:
    _, t2s = _tools()
    before, after = text[i - 1:i + 1] if i else "", text[i:i + 2]
    return not any(len(w) == 2 and CJK_ONLY.match(w) and _dictionary_word(t2s.convert(w)) for w in (before, after))
