"""Output checks and retries in the LLM translators, with a scripted fake model."""

import json

from atx.translators import llm
from atx.translators.llm import HyMtTranslator, QwenPageTranslator, bad_output, badness, run_on


class FakeServer:
    url = "http://127.0.0.1:1"
    parallel = 2
    device = "cpu"


def scripted(cls, reply, **kw):
    """A translator whose model answers each prompt with reply(prompt_text)."""
    tr = cls(FakeServer(), **kw)
    tr.prompts = []

    def chat(messages, max_tokens, **extra):
        text = messages[-1]["content"]
        tr.prompts.append(text)
        return reply(text), [("x", -0.1)]

    tr._chat_scored = chat
    return tr


def test_run_on():
    leaked = "I am!\nGuess where I am?\nYou definitely can't guess that I've already arrived in Taiwan!"
    assert run_on("嗶", leaked)
    assert run_on("哪個？", "Which one? I know! That girl who wants to become popular right now!")
    assert not run_on("嗶", "Beep")
    narration = "妮卡認為黎玥跟自己一樣都是不被愛的孩子，黎玥順水推舟答應要幫妮卡扳倒利亞繪。"
    assert not run_on(narration, "Nika believes that Li Yue, like her, is a child who isn't loved. "
                                 "Li Yue readily agrees to help Nika bring down Li Yahui.")


def test_bad_output():
    assert bad_output("學姐就很木訥啊！", "学姐好木讷啊！")  # echoed in Chinese
    assert bad_output("好", "  ")
    assert not bad_output("好", "Okay")


def test_hymt_retries_leaked_context_without_it():
    def reply(prompt):
        if prompt.startswith(llm.HY_MT_PROMPT.split("{")[0]):  # plain template: behaves
            return "Beep"
        return "It's me!\nGuess where I am?"  # contextual template: leaks the context

    tr = scripted(HyMtTranslator, reply, use_context=True)
    assert tr.translate(["嗶"], context=["是我！", "猜猜我在哪？"]) == ["Beep"]
    assert len(tr.prompts) == 2


def test_hymt_keeps_good_contextual_output():
    tr = scripted(HyMtTranslator, lambda p: "Which one?", use_context=True)
    assert tr.translate(["哪個？"], context=["小奈？"]) == ["Which one?"]
    assert len(tr.prompts) == 1


def test_qwen_retries_merged_bubble_alone():
    page = ["是我！", "猜猜我在哪？"]

    def reply(prompt):
        texts = json.loads(prompt.rsplit("\n", 1)[1])
        if len(texts) == 2:  # whole page: first bubble swallowed the second
            return json.dumps(["It's me! Guess where I am? I bet you'll never guess, I'm already in Taiwan!",
                               "Guess where I am?"])
        return json.dumps(["It's me!"])

    tr = scripted(QwenPageTranslator, reply)
    assert tr.translate(page) == ["It's me!", "Guess where I am?"]


def test_badness_ranks_attempts():
    assert badness("盲腸炎", "Appendicitis?") == 0
    assert badness("盲腸炎", "Blind腸itis?") == 1
    assert badness("學姐就很木訥啊！", "学姐好木讷啊！") >= 100


def test_hymt_retries_stray_chinese_and_keeps_the_better_one():
    replies = iter(["Blind腸itis?", "Appendicitis?"])
    tr = scripted(HyMtTranslator, lambda p: next(replies))
    assert tr.translate(["盲腸炎？"]) == ["Appendicitis?"]

    replies = iter(["Blind腸itis?", "盲腸炎？"])  # the retry is worse: keep the first
    tr = scripted(HyMtTranslator, lambda p: next(replies))
    assert tr.translate(["盲腸炎？"]) == ["Blind腸itis?"]
