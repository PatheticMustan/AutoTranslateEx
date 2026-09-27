"""Grouping on synthetic boxes. Sizes mirror the target site: ~50px glyphs in
dialogue columns, ~32px in horizontal narration boxes."""

from atx.grouping import group_lines, keep
from atx.ocr import Line


def col(x0, y0, text, w=50, score=1.0):
    """A vertical column: one glyph wide, one glyph tall per character."""
    return Line((x0, y0, x0 + w, y0 + w * len(text)), text, score)


def row(x0, y0, text, h=32, score=1.0):
    return Line((x0, y0, x0 + h * len(text), y0 + h), text, score)


def texts(bubbles):
    return [b.text for b in bubbles]


def test_touching_columns_form_one_bubble_read_right_to_left():
    lines = [col(100, 0, "三三"), col(150, 0, "二二二"), col(200, 0, "一一一一")]
    bubbles = group_lines(lines)
    assert texts(bubbles) == ["一一一一二二二三三"]
    assert bubbles[0].vertical


def test_bubbles_a_glyph_apart_stay_separate():
    # Each bubble's columns touch; the bubbles are 40px apart (220 -> 260), the
    # case that used to over-merge.
    right = [col(260, 640, "四四四"), col(310, 640, "三三三")]
    left = [col(120, 690, "二二二二"), col(170, 690, "一一一一")]
    bubbles = group_lines(right + left)
    assert len(bubbles) == 2


def test_columns_with_close_centers_keep_right_to_left_order():
    # Overlapping columns whose centers differ by less than a glyph used to tie.
    lines = [Line((427, 1430, 465, 1492), "二。", 1.0), Line((457, 1431, 492, 1547), "一一一一", 1.0)]
    assert texts(group_lines(lines)) == ["一一一一二。"]


def test_narration_rows_merge_top_to_bottom():
    # 11px between rows of 31px glyphs; the next narration box is 51px below.
    rows = [row(40, 1132, "一一一一一", h=31), row(40, 1174, "二二二", h=31), row(40, 1214, "三三三三", h=31)]
    next_box = [row(40, 1296, "四四四四", h=31)]
    bubbles = group_lines(rows + next_box)
    assert texts(bubbles) == ["一一一一一二二二三三三三", "四四四四"]
    assert not bubbles[0].vertical


def test_big_sound_effect_does_not_swallow_dialogue():
    dialogue = col(100, 100, "一一一", w=40)
    sfx = col(145, 60, "二二", w=90)
    assert len(group_lines([dialogue, sfx])) == 2


def test_vertical_and_horizontal_lines_do_not_merge():
    assert len(group_lines([col(100, 100, "一一一"), row(155, 100, "二二二")])) == 2


def test_keep_drops_low_confidence_and_non_chinese():
    assert keep(col(0, 0, "一一"))
    assert not keep(col(0, 0, "一一", score=0.3))
    assert not keep(row(0, 0, "AcloudMerge"))
    assert not keep(row(0, 0, "12"))


def test_page_order_is_top_to_bottom_then_right_to_left():
    top_left = col(50, 0, "一一")
    top_right = col(500, 20, "二二")
    bottom = col(300, 800, "三三")
    assert texts(group_lines([bottom, top_left, top_right])) == ["二二", "一一", "三三"]
