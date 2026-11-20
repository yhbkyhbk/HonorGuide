"""用关键词和向量检索旧对话，为问答 Agent 选择历史上下文。"""

import atexit
from functools import lru_cache
import hashlib
from itertools import islice
import json
import re
import sqlite3
from threading import Lock
from uuid import uuid4

from openai import APIError
from pymilvus import DataType, MilvusClient
from pymilvus.exceptions import MilvusException

from config import BASE_URL, MEMORY_EMBEDDING_MODEL
from infrastructure.call_model_api import embed_texts
from infrastructure.open_milvus_client import open_client

from .store_conversation_history import (
    MEMORY_DIR,
    find_keyword_turn_ids,
    make_memory_query,
    open_memory_database,
)


RECENT_TURNS = 3
RECALL_LIMIT = 8
OLD_MEMORY_CHAR_BUDGET = 2400
VECTOR_MIN_SCORE = 0.55
EMBEDDING_TIMEOUT = 5.0
BACKFILL_BATCH = 19  # 给当前查询或最新问答留一个嵌入位置。
MEMORY_MILVUS_PATH = MEMORY_DIR / "milvus.db"
EMBEDDING_KEY = hashlib.sha256(f"{BASE_URL}\n{MEMORY_EMBEDDING_MODEL}".encode()).hexdigest()
HISTORICAL_CITATION = re.compile(r"\[[TG]\d+\]")
MEMORY_VECTOR_ERRORS = (APIError, ImportError, MilvusException, OSError, sqlite3.Error, ValueError)
_vector_lock = Lock()


@lru_cache(maxsize=1)
def _get_client() -> MilvusClient:
    return open_client(str(MEMORY_MILVUS_PATH))


@lru_cache(maxsize=4)
def _load_collection(name: str) -> None:
    _get_client().load_collection(collection_name=name)


@atexit.register
def _close_client() -> None:
    if _get_client.cache_info().currsize:
        _get_client().close()
    _get_client.cache_clear()
    _load_collection.cache_clear()


def _context_turn(entry: dict, question_chars: int = 1000, answer_chars: int = 1600) -> dict[str, str]:
    return {
        "question": entry["question"][:question_chars].strip(),
        "answer": HISTORICAL_CITATION.sub("", entry["result"]["answer"])[:answer_chars].strip(),
    }


def _memory_vector_state() -> tuple[str, int]:
    with open_memory_database() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS memory_vector_state (
                embedding_key TEXT PRIMARY KEY,
                collection_name TEXT NOT NULL,
                last_turn_id INTEGER NOT NULL
            )
        """)
        row = db.execute(
            "SELECT collection_name, last_turn_id FROM memory_vector_state WHERE embedding_key = ?",
            (EMBEDDING_KEY,),
        ).fetchone()
        if row is None:
            row = (f"honorguide_memory_{uuid4().hex}", 0)
            db.execute("INSERT INTO memory_vector_state VALUES (?, ?, ?)", (EMBEDDING_KEY, *row))
        return row


def _set_vector_cursor(turn_id: int) -> None:
    with open_memory_database() as db:
        db.execute(
            "UPDATE memory_vector_state SET last_turn_id = ? WHERE embedding_key = ?",
            (turn_id, EMBEDDING_KEY),
        )


def _create_collection(client, name: str, dimension: int) -> None:
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dimension)
    indexes = client.prepare_index_params()
    indexes.add_index(field_name="vector", index_type="FLAT", metric_type="COSINE")
    client.create_collection(collection_name=name, schema=schema, index_params=indexes)
    _load_collection.cache_clear()


def _sync_vector_index(
    before_id: int, *, question: str | None = None, newest: tuple[int, dict] | None = None,
):
    """追补一批漏存记录，优先写入新一轮，查询时顺便编码当前问题。"""
    client = _get_client()
    name, last_turn_id = _memory_vector_state()
    exists = client.has_collection(collection_name=name)
    if not exists and last_turn_id:
        last_turn_id = 0
        _set_vector_cursor(last_turn_id)
    with open_memory_database() as db:
        batch = db.execute("""
            SELECT id, payload FROM turns
            WHERE id > ? AND id < ? AND json_type(payload, '$.result') = 'object'
            ORDER BY id
            LIMIT ?
        """, (last_turn_id, before_id, BACKFILL_BATCH)).fetchall()

    checked_through = batch[-1][0] if batch else last_turn_id
    candidates = [(turn_id, json.loads(payload)) for turn_id, payload in batch]
    if newest and newest[0] > checked_through:
        candidates.append(newest)
    if not exists and not candidates:
        return client, name, None

    if exists and candidates:
        _load_collection(name)
        indexed = client.get(
            collection_name=name, ids=[turn_id for turn_id, _ in candidates],
            output_fields=["id"], consistency_level="Strong",
        )
        indexed_ids = {row["id"] for row in indexed}
        candidates = [(turn_id, entry) for turn_id, entry in candidates if turn_id not in indexed_ids]

    contexts = [_context_turn(entry, 400, 1000) for _, entry in candidates]
    texts = [f"{item['question']}\n{item['answer']}" for item in contexts]
    offset = int(question is not None)
    if question is not None:
        texts.insert(0, question[:1000])
    query_vector = None
    if texts:
        vectors = embed_texts(texts, model=MEMORY_EMBEDDING_MODEL, timeout=EMBEDDING_TIMEOUT)
        if not exists:
            _create_collection(client, name, vectors.shape[1])
        if candidates:
            rows = [{"id": turn_id, "vector": vector.tolist()}
                    for (turn_id, _), vector in zip(candidates, vectors[offset:], strict=True)]
            saved = client.upsert(collection_name=name, data=rows)
            if saved["upsert_count"] != len(rows):
                raise ValueError("记忆向量写入数量不完整")
            client.flush(collection_name=name)
        if question is not None:
            query_vector = vectors[0].tolist()
    if checked_through > last_turn_id:
        _set_vector_cursor(checked_through)
    return client, name, query_vector


def index_saved_turn(turn_id: int, entry: dict) -> None:
    """SQLite 提交成功后立刻将这轮完整问答写入 Milvus。"""
    with _vector_lock:
        _sync_vector_index(turn_id + 1, newest=(turn_id, entry))


def _find_vector_turn_ids(question: str, before_id: int) -> list[int]:
    with _vector_lock:
        client, name, query_vector = _sync_vector_index(before_id, question=question)
        if query_vector is None:
            return []
        _load_collection(name)
        hits = client.search(
            collection_name=name, data=[query_vector], anns_field="vector",
            filter=f"id < {before_id}", limit=RECALL_LIMIT,
            search_params={"metric_type": "COSINE"}, consistency_level="Strong",
        )[0]
    return [hit["id"] for hit in hits if hit["distance"] >= VECTOR_MIN_SCORE]


def _rank_memory_turns(keyword_ids: list[int], vector_ids: list[int]) -> list[int]:
    """按两路命中的名次做 RRF，返回旧对话的 SQLite ID。"""
    scores: dict[int, float] = {}
    for ids in (keyword_ids, vector_ids):
        for rank, turn_id in enumerate(ids, 1):
            scores[turn_id] = scores.get(turn_id, 0.0) + 1 / (60 + rank)
    return sorted(scores, key=lambda turn_id: (-scores[turn_id], turn_id))


def _read_related_turns(ids: list[int], recent: list[dict]) -> list[dict[str, str]]:
    if not ids:
        return []
    placeholders = ", ".join("?" for _ in ids)
    with open_memory_database() as db:
        payloads = dict(db.execute(f"SELECT id, payload FROM turns WHERE id IN ({placeholders})", ids))

    excerpts = (_context_turn(entry, 400, 650) for entry in recent)
    seen = {(turn["question"], turn["answer"]) for turn in excerpts}
    related = []
    remaining = OLD_MEMORY_CHAR_BUDGET
    for turn_id in ids:
        if turn_id not in payloads:
            continue
        turn = _context_turn(json.loads(payloads[turn_id]), 400, 650)
        signature = turn["question"], turn["answer"]
        size = sum(map(len, signature))
        if signature in seen or size > remaining:
            continue
        seen.add(signature)
        remaining -= size
        related.append({**turn, "source": "related"})
    return related


def select_context(question: str, history: list[dict]) -> list[dict[str, str]]:
    """保留最近几轮，融合两路旧记忆，并在长度预算内选取片段。"""
    completed = (entry for entry in reversed(history) if "result" in entry)
    recent = list(islice(completed, RECENT_TURNS))
    context = [_context_turn(entry) for entry in reversed(recent)]
    if next(completed, None) is None:
        return context
    match = make_memory_query(question)
    if not match:
        return context

    cutoff_request = recent[-1]["result"].get("timings", {}).get("request_id")
    if not cutoff_request:
        return context
    try:
        with open_memory_database() as db:
            cutoff = db.execute("SELECT id FROM turns WHERE request_id = ?", (cutoff_request,)).fetchone()
        if not cutoff:
            return context
        before_id = cutoff[0]
        try:
            keyword_ids = find_keyword_turn_ids(match, before_id, RECALL_LIMIT)
        except sqlite3.Error:
            keyword_ids = []
        try:
            vector_ids = _find_vector_turn_ids(question, before_id)
        except MEMORY_VECTOR_ERRORS:
            vector_ids = []
        ranked_ids = _rank_memory_turns(keyword_ids, vector_ids)
        return [*_read_related_turns(ranked_ids, recent), *context]
    except (OSError, ValueError, sqlite3.Error):
        return context
