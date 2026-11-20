"""BM25 索引：分词、持久化统计与倒排检索。"""

from __future__ import annotations

import json
from functools import lru_cache

from rank_bm25 import BM25Okapi

from config import STORAGE
from .knowledge.prepare_documents import load_chunks
from .tokenize_search_text import register_terms, tokenize

INDEX_FILE = STORAGE / "bm25.json"


def _register_heroes(chunks: list[dict]) -> None:
    # 从现有元数据加入英雄名，避免“艾琳”等名称被拆开，不需要单独的词典文件。
    register_terms({chunk["metadata"]["hero"] for chunk in chunks})


def build_index(chunks: list[dict] | None = None) -> BM25Okapi:
    """保存词频、IDF、长度，以及“词 → 块编号列表”的倒排索引。"""
    chunks = load_chunks() if chunks is None else chunks
    _register_heroes(chunks)

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
    return index


@lru_cache(maxsize=1)
def load_index() -> tuple[BM25Okapi, list[dict]]:
    """恢复 BM25 统计和分块，初始化结巴词典并注册英雄名。"""
    chunks = load_chunks()
    state = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    if "postings" not in state:
        raise ValueError("旧 BM25 索引不含倒排表，请在侧栏更新知识库")
    # 绕过 __init__ 恢复已有统计，避免再次分词和计算 IDF。
    index = BM25Okapi.__new__(BM25Okapi)
    index.__dict__.update(state)
    if index.corpus_size != len(chunks):
        raise ValueError("BM25 索引与分块数量不匹配，请在侧栏更新知识库")
    _register_heroes(chunks)
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
