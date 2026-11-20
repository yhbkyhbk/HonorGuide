"""持久化本地问答并维护 SQLite 全文索引。"""

from contextlib import closing, contextmanager
from itertools import combinations
import json
import logging
import re
import sqlite3

import jieba

from config import ROOT


MEMORY_DIR = ROOT / "data" / "memory"
CONVERSATION_FILE = MEMORY_DIR / "conversations.db"
SEARCH_INDEX_VERSION = 1
_tokenizer = jieba.Tokenizer()
jieba.setLogLevel(logging.WARNING)

TIME_PHRASES = re.compile(r"回到|上次|之前|先前|聊过|提过|说过|前面|刚才|刚刚|前几轮")
GENERAL_WORDS = {
    "为什么", "其他", "关于", "问题", "这个", "那个", "那次", "这次", "一题",
    "告诉", "继续", "讨论", "一下", "怎么回事", "记得",
    "什么", "哪些", "怎么", "如何", "可以", "请问",
}
COMMON_GAME_TERMS = {
    "出装", "防装", "装备", "铭文", "技能", "连招", "打法", "对线", "团战",
    "走位", "操作", "思路", "搭配", "推荐", "建议", "英雄",
}


@contextmanager
def open_memory_database():
    CONVERSATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(CONVERSATION_FILE)) as db, db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY,
                request_id TEXT NOT NULL UNIQUE,
                payload TEXT NOT NULL
            )
        """)
        yield db


def load_history() -> list[dict]:
    with open_memory_database() as db:
        return [json.loads(payload) for (payload,) in db.execute("SELECT payload FROM turns ORDER BY id")]


def _topic_words(text: str) -> list[str]:
    cleaned = re.sub(r"https?://[^\s)]+", " ", TIME_PHRASES.sub(" ", text[:1000]).lower())
    words = _tokenizer.cut(cleaned)
    return list(dict.fromkeys(
        word for word in words if re.fullmatch(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", word)
        and word not in GENERAL_WORDS
    ))


def _index_question(db, turn_id: int, question: str) -> None:
    db.execute(
        "INSERT INTO turn_search (rowid, terms) VALUES (?, ?)",
        (turn_id, " ".join(_topic_words(question))),
    )


def _search_index_ready(db) -> bool:
    return db.execute("""
        SELECT 1 FROM sqlite_master, pragma_user_version
        WHERE type = 'table' AND name = 'turn_search' AND user_version = ?
    """, (SEARCH_INDEX_VERSION,)).fetchone() is not None


def _ensure_search_index(db) -> bool:
    if _search_index_ready(db):
        return True
    if not db.in_transaction:
        db.execute("BEGIN IMMEDIATE")
        if _search_index_ready(db):
            return True
    try:
        db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS turn_search USING fts5(terms)")
        # 首次使用独立的记忆分词器时，从旧对话原文重建关键词索引。
        db.execute("DELETE FROM turn_search")
    except sqlite3.OperationalError as exc:
        if "no such module: fts5" in str(exc).lower():
            return False
        raise
    for turn_id, payload in db.execute("SELECT id, payload FROM turns"):
        entry = json.loads(payload)
        if "result" in entry:
            _index_question(db, turn_id, entry["question"])
    db.execute(f"PRAGMA user_version = {SEARCH_INDEX_VERSION}")
    return True


def save_turn(request_id: str, entry: dict) -> int:
    """按请求编号写入，返回用于关联向量的 SQLite 记录 ID。"""
    payload = json.dumps(entry, ensure_ascii=False)
    with open_memory_database() as db:
        (turn_id,) = db.execute("""
            INSERT INTO turns (request_id, payload) VALUES (?, ?)
            ON CONFLICT(request_id) DO UPDATE SET payload = excluded.payload
            RETURNING id
        """, (request_id, payload)).fetchone()
        if "result" in entry and _ensure_search_index(db) and not db.execute(
            "SELECT 1 FROM turn_search WHERE rowid = ?", (turn_id,),
        ).fetchone():
            _index_question(db, turn_id, entry["question"])
        return turn_id


def make_memory_query(question: str) -> str:
    """过滤短指代和只有通用游戏词的提问，构建 FTS 查询。"""
    return " OR ".join(
        f'("{left}" AND "{right}")'
        for left, right in combinations(_topic_words(question)[:8], 2)
        if left not in COMMON_GAME_TERMS or right not in COMMON_GAME_TERMS
    )


def find_keyword_turn_ids(match: str, before_id: int, limit: int) -> list[int]:
    with open_memory_database() as db:
        if not _ensure_search_index(db):
            return []
        return [turn_id for (turn_id,) in db.execute("""
            SELECT rowid FROM turn_search
            WHERE turn_search MATCH ? AND rowid < ?
            ORDER BY rank, rowid DESC
            LIMIT ?
        """, (match, before_id, limit))]
