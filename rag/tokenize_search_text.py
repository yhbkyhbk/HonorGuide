"""攻略检索的中文分词与文本清理。"""

from collections.abc import Iterable
import logging
import re

import jieba


STOP_WORDS = {"的", "了", "是", "和", "与", "在", "有", "能", "吗", "什么", "哪些", "怎么", "如何", "可以", "请问"}
jieba.setLogLevel(logging.WARNING)


def strip_reference_lines(text: str) -> str:
    return re.sub(r"(?m)^(?:参考|机制参考|铭文参考)[：:].*$", "", text)


def tokenize(text: str) -> list[str]:
    """文档与问题使用同样的规则：去来源行和 URL，分词后过滤标点与停用词。"""
    text = re.sub(r"https?://[^\s)]+", "", strip_reference_lines(text))
    return [
        word for word in jieba.lcut(text.lower())
        if re.fullmatch(r"[\u4e00-\u9fffA-Za-z0-9]+", word) and word not in STOP_WORDS
    ]


def register_terms(terms: Iterable[str]) -> None:
    for term in terms:
        jieba.add_word(term)
