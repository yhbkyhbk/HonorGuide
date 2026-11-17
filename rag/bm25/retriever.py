"""BM25 索引：分词、持久化统计与倒排检索。"""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache

import jieba
from rank_bm25 import BM25Okapi

from config import STORAGE

CHUNKS_FILE = STORAGE / "chunks.json"
INDEX_FILE = STORAGE / "bm25.json"
STOP_WORDS = {"的", "了", "是", "和", "与", "在", "有", "能", "吗", "什么", "哪些", "怎么", "如何", "可以", "请问"}
jieba.setLogLevel(logging.WARNING)


def tokenize(text: str) -> list[str]:
    """文档与问题使用同样的规则：去来源行、URL、标点和常见停用词，再中文分词。"""
    text = re.sub(r"(?m)^(?:参考|机制参考|铭文参考)[：:].*$", "", text)
    text = re.sub(r"https?://[^\s)]+", "", text)
    return [
        word for word in jieba.lcut(text.lower())
        if re.fullmatch(r"[\u4e00-\u9fffA-Za-z0-9]+", word) and word not in STOP_WORDS
    ]


def build_index() -> BM25Okapi:
    """保存词频、IDF、长度，以及“词 → 块编号列表”的倒排索引。"""
    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("chunks.json 必须是非空分块列表，请先在知识库管理页面处理文档")
    # 从现有元数据加入英雄名，避免“艾琳”等名称被拆开，不需要单独的词典文件。
    for hero in {chunk["metadata"]["hero"] for chunk in chunks}:
        jieba.add_word(hero)

    corpus = [tokenize(chunk["page_content"]) for chunk in chunks]
    if not any(corpus):
        raise ValueError("文档中没有可用于 BM25 检索的词")
    index = BM25Okapi(corpus)
    index.postings = {}
    for position, frequencies in enumerate(index.doc_freqs):
        for word in frequencies:
            index.postings.setdefault(word, []).append(position)
    INDEX_FILE.write_text(
        json.dumps(index.__dict__, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    load_index.cache_clear()
    print(f"已建立 BM25 索引：{index.corpus_size} 个块，{len(index.idf)} 个词", flush=True)
    return index


@lru_cache(maxsize=1)
def load_index() -> tuple[BM25Okapi, list[dict]]:
    """从 JSON 恢复统计对象；同一进程反复查询时复用内存缓存。"""
    if not INDEX_FILE.is_file():
        build_index()
    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    state = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    if "postings" not in state:
        raise ValueError("旧 BM25 索引不含倒排表，请在知识库管理页面重新构建索引")
    # 绕过 __init__ 恢复已有统计，避免再次分词和计算 IDF。
    index = BM25Okapi.__new__(BM25Okapi)
    index.__dict__.update(state)
    if not chunks or index.corpus_size != len(chunks):
        raise ValueError("BM25 索引与分块数量不匹配，请在知识库管理页面重新构建索引")
    for hero in {chunk["metadata"]["hero"] for chunk in chunks}:
        jieba.add_word(hero)
    return index, chunks


def search(question: str, top_k: int = 3, *, trace=None) -> list[dict]:
    """问题分词后，通过倒排表找到候选，仅计算这些块的 BM25 得分。"""
    if not question.strip() or top_k <= 0:
        raise ValueError("问题不能为空，top_k 必须大于 0")
    if trace:
        trace.step("BM25 统计加载（可命中缓存）")
    index, chunks = load_index()
    if trace:
        trace.step("问题分词")
    query = tokenize(question)
    if not query:
        return []
    if trace:
        trace.step("BM25 倒排召回、评分与排序")
    candidates = sorted({i for word in query for i in index.postings.get(word, [])})
    if not candidates:
        return []
    scores = index.get_batch_scores(query, candidates)
    ranked = sorted(zip(candidates, scores), key=lambda item: (-item[1], item[0]))[:top_k]
    return [{**chunks[i], "score": float(score)} for i, score in ranked]
