"""Name bank: finding names, spelling them, and applying them."""

from atx.cache import Cache
from atx.names import NameBank, find_names, names_from_translation, romanize, terms_for


def test_find_names_skips_dictionary_words():
    found = find_names("黃可馨告訴小紅，放出王宥卉潑咖啡影片的人，其實就是妮卡。")
    assert {"黃可馨", "小紅", "妮卡"} <= set(found)
    # Ordinary words that jieba also tags as names.
    assert find_names("謝謝！拜託你原諒我，我明白了。") == []


def test_names_from_translation_matches_pinyin():
    src = "妮卡認為黎玥跟自己一樣都是不被愛的孩子，黎玥答應要幫妮卡扳倒利亞繪。"
    dst = "Nika believed that Li Yue, like her, was unloved. Li Yue agreed to help Nika bring down Lia Yan."
    assert set(names_from_translation(src, dst)) == {"妮卡", "黎玥"}  # "Lia Yan" doesn't spell 利亞繪
    assert names_from_translation("你好嗎", "How are you") == []


def test_romanize():
    assert romanize("黎玥") == "Li Yue"
    assert romanize("王宥卉") == "Wang Youhui"
    assert romanize("小紅") == "Xiao Hong"
    assert romanize("妮卡") == "Nika"
    assert romanize("森叔") == "Uncle Sen"


def test_terms_for_prefers_longest_and_skips_removed():
    bank = {"小奈": "Xiao Nai", "里小奈": "Li Xiaonai", "妮卡": ""}
    assert terms_for("第二個里小奈！小奈呢？", bank) == [("里小奈", "Li Xiaonai"), ("小奈", "Xiao Nai")]
    assert terms_for("里小奈", bank) == [("里小奈", "Li Xiaonai")]
    assert terms_for("妮卡", bank) == []  # removed by the user


def test_bank_needs_two_pages_and_keeps_user_spellings(tmp_path):
    bank = NameBank(Cache(tmp_path / "c.db"))
    bank.observe_names("s", "p1", ["黎玥"])
    bank.observe_names("s", "p1", ["黎玥"])  # same page again: still one page
    assert bank.active("s") == {}
    bank.observe_names("s", "p2", ["黎玥"])
    assert bank.active("s") == {"黎玥": "Li Yue"}
    bank.set("s", "黎玥", "Lee Yue")
    bank.observe_names("s", "p3", ["黎玥"])
    assert bank.active("s") == {"黎玥": "Lee Yue"}
    bank.set("s", "黎玥", None)  # removed: stays known, but has no spelling
    assert terms_for("黎玥", bank.active("s")) == []
    assert bank.active("other") == {}


def test_translation_names_skip_interjections_but_keep_nicknames():
    assert names_from_translation("哈哈哈！", "Hahaha!") == []
    assert names_from_translation("叮咚", "Ding Dong") == []
    assert names_from_translation("花花！", "Huahua!") == ["花花"]
