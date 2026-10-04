"""Group OCR text lines into speech bubbles.

Each line's box is grown by a fraction of its glyph size and lines whose grown
boxes touch are merged (union-find). Growth is larger *across* the text
direction (the gap between neighbouring columns or rows of one bubble) than
*along* it (a bubble's lines don't continue past each other's ends).

Lines only merge when their orientation is compatible and their glyph sizes are
similar, which keeps big sound-effect text from swallowing nearby dialogue.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from atx.ocr import Line

CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")

# Columns of one bubble touch or overlap (gap ~0), while neighbouring bubbles are
# ~0.8 glyphs apart on the target site; narration rows have gaps up to ~0.35 glyphs.
GROW_ACROSS = 0.45  # x glyph size, between columns (vertical) / rows (horizontal)
GROW_ALONG = 0.3    # x glyph size, past the ends of a line
MAX_SIZE_RATIO = 1.6
MIN_SCORE = 0.5


@dataclass
class Bubble:
    lines: list[Line]  # in reading order
    vertical: bool

    @property
    def text(self) -> str:
        return "".join(line.text for line in self.lines)

    @property
    def box(self) -> tuple[int, int, int, int]:
        xs0, ys0, xs1, ys1 = zip(*(line.box for line in self.lines))
        return min(xs0), min(ys0), max(xs1), max(ys1)


def _grown(line: Line, vertical: bool | None) -> tuple[float, float, float, float]:
    s = line.char_size
    if vertical is None:  # single character: unknown direction, grow evenly
        gx = gy = GROW_ACROSS * s
    elif vertical:
        gx, gy = GROW_ACROSS * s, GROW_ALONG * s
    else:
        gx, gy = GROW_ALONG * s, GROW_ACROSS * s
    x0, y0, x1, y1 = line.box
    return x0 - gx, y0 - gy, x1 + gx, y1 + gy


def _direction(line: Line) -> bool | None:
    if line.vertical:
        return True
    if line.horizontal:
        return False
    return None


def _touch(a, b) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def _compatible(a: Line, b: Line) -> bool:
    da, db = _direction(a), _direction(b)
    if da is not None and db is not None and da != db:
        return False
    big, small = max(a.char_size, b.char_size), min(a.char_size, b.char_size)
    return small > 0 and big / small <= MAX_SIZE_RATIO


# mycomic.com's watermark: 集云数据 over "ACloudMerge.com". It's in Simplified
# Chinese, which never appears in the Traditional text it sits on, and OCR
# sometimes reads it as part of a real line ("ACloudMe信件內只有n張紙條").
WATERMARK = re.compile(r"集云数据|集云|云数据|数据|[A-Za-z]*Cloud[A-Za-z]*(?:\.com)?|[A-Za-z]*erge\.com")


def strip_watermark(line: Line) -> Line:
    text = WATERMARK.sub("", line.text).strip()
    return line if text == line.text else Line(line.box, text, line.score)


def keep(line: Line) -> bool:
    """Drop low-confidence lines and ones with no Chinese (watermarks, page numbers)."""
    return line.score >= MIN_SCORE and bool(CJK.search(line.text))


def group_lines(lines: list[Line]) -> list[Bubble]:
    lines = [ln for ln in map(strip_watermark, lines) if keep(ln)]
    n = len(lines)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    grown = [_grown(ln, _direction(ln)) for ln in lines]
    for i in range(n):
        for j in range(i + 1, n):
            # Either line's grown box reaching the other's real box is enough.
            if (_touch(grown[i], lines[j].box) or _touch(grown[j], lines[i].box)) \
                    and _compatible(lines[i], lines[j]):
                parent[find(i)] = find(j)

    groups: dict[int, list[Line]] = {}
    for i, ln in enumerate(lines):
        groups.setdefault(find(i), []).append(ln)

    bubbles = [_make_bubble(g) for g in groups.values()]
    # Page reading order: top to bottom, then right to left for bubbles side by side.
    bubbles.sort(key=lambda b: (_row_key(b), -b.box[2]))
    return bubbles


def _make_bubble(group: list[Line]) -> Bubble:
    # Direction by majority of characters; single characters don't vote.
    v = sum(len(ln.text) for ln in group if ln.vertical)
    h = sum(len(ln.text) for ln in group if ln.horizontal)
    vertical = v >= h if (v or h) else group[0].h >= group[0].w
    if vertical:
        # Columns right to left, then top to bottom within a column.
        ordered = _in_tracks(group, center=lambda ln: -(ln.box[0] + ln.box[2]) / 2, along=lambda ln: ln.box[1])
    else:
        # Rows top to bottom, then left to right within a row.
        ordered = _in_tracks(group, center=lambda ln: (ln.box[1] + ln.box[3]) / 2, along=lambda ln: ln.box[0])
    return Bubble(ordered, vertical)


def _in_tracks(group: list[Line], center, along) -> list[Line]:
    """Cluster lines into tracks (columns or rows) by center, then read track by track.

    A line starts a new track when its center is more than half a glyph past the
    previous track's; `center` is signed so tracks come out in reading order.
    """
    size = max(1.0, sorted(ln.char_size for ln in group)[len(group) // 2])
    tracks: list[list[Line]] = []
    for ln in sorted(group, key=center):
        if tracks and center(ln) - center(tracks[-1][0]) <= size / 2:
            tracks[-1].append(ln)
        else:
            tracks.append([ln])
    return [ln for track in tracks for ln in sorted(track, key=along)]


def _row_key(b: Bubble) -> int:
    # Bubbles whose tops are within ~100px count as the same row.
    return b.box[1] // 100
