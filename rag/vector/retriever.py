"""Milvus 向量模块：嵌入、持久化、HNSW 索引、集合切换与检索。"""

from __future__ import annotations

import atexit
from functools import lru_cache
import json
from pathlib import Path
import re
from uuid import uuid4

import numpy as np
from pymilvus import DataType, MilvusClient
from pymilvus.exceptions import MilvusException

from config import (
    EMBEDDING_MODEL as MODEL, STORAGE, ROOT,
    MILVUS_URI, MILVUS_TOKEN, MILVUS_COLLECTION,
)
from ..model_client import get_model_client

HNSW_M = 16
EF_CONSTRUCTION = 100
EF_SEARCH = 32


def embed_texts(texts: list[str]) -> np.ndarray:
    """每批最多 20 段文本；返回归一化后的 float32 向量。"""
    if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("待编码的文本不能为空")
    vectors = []
    client = get_model_client()
    for start in range(0, len(texts), 20):
        batch = texts[start:start + 20]
        response = client.embeddings.create(
            model=MODEL, input=batch, encoding_format="float"
        )
        items = sorted(response.data, key=lambda item: item.index)
        if [item.index for item in items] != list(range(len(batch))):
            raise ValueError("接口返回的向量数量或顺序编号异常")
        vectors.extend(item.embedding for item in items)
    array = np.asarray(vectors, dtype="float32")
    if array.ndim != 2 or not np.isfinite(array).all() or np.any(np.linalg.norm(array, axis=1) == 0):
        raise ValueError("接口返回的向量包含无效数值或零向量")
    array /= np.linalg.norm(array, axis=1, keepdims=True)
    return array


@lru_cache(maxsize=1)
def get_client() -> MilvusClient:
    """本地 .db 路径使用 Milvus Lite，http(s) 地址使用 Milvus 服务。"""
    uri = MILVUS_URI
    if not uri.startswith(("http://", "https://")):
        path = Path(uri)
        path = path if path.is_absolute() else ROOT / path
        if path.suffix != ".db":
            raise ValueError("本地 Milvus 路径应以 .db 结尾，服务端连接应使用 http(s) 地址")
        path.parent.mkdir(parents=True, exist_ok=True)
        # Lite 与应用处于同一进程，避免空闲时的 gRPC 心跳触发 too_many_pings。
        return MilvusClient(
            uri=str(path.resolve()), token=MILVUS_TOKEN, timeout=10,
            grpc_options={"grpc.keepalive_permit_without_calls": False},
        )
    return MilvusClient(uri=uri, token=MILVUS_TOKEN, timeout=10)


@lru_cache(maxsize=1)
def load_index() -> None:
    """载入一次现有集合，由页面预加载和向量检索共享。"""
    get_client().load_collection(collection_name=MILVUS_COLLECTION)


def close_client() -> None:
    if get_client.cache_info().currsize:
        get_client().close()
    get_client.cache_clear()
    load_index.cache_clear()


atexit.register(close_client)


def load_chunks() -> list[dict]:
    chunks = json.loads((STORAGE / "chunks.json").read_text(encoding="utf-8"))
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("文本块为空，请检查攻略后更新知识库")
    ids = [row["metadata"]["chunk_id"] for row in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("文本块编号重复，请重新处理文档")
    return chunks


def build_index() -> dict:
    """先建新集合，完整写入、索引并加载成功后切换固定别名。"""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,100}", MILVUS_COLLECTION):
        raise ValueError("MILVUS_COLLECTION 应使用不超过 101 位的英文字母、数字和下划线")
    chunks = load_chunks()
    client = get_client()
    previous = None
    if client.has_collection(collection_name=MILVUS_COLLECTION):
        previous = client.describe_collection(collection_name=MILVUS_COLLECTION)["collection_name"]
        if previous == MILVUS_COLLECTION:
            raise ValueError("Milvus 逻辑名称被实体集合占用，请修改 MILVUS_COLLECTION 后重建")
    vectors = embed_texts([row["page_content"] for row in chunks])
    name = MILVUS_COLLECTION + "__" + uuid4().hex[:12]
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=512)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=vectors.shape[1])
    schema.add_field("page_content", DataType.VARCHAR, max_length=65535)
    schema.add_field("metadata", DataType.JSON)
    indexes = client.prepare_index_params()
    indexes.add_index(
        field_name="vector", index_name="vector_hnsw", index_type="HNSW", metric_type="COSINE",
        params={"M": HNSW_M, "efConstruction": EF_CONSTRUCTION},
    )
    try:
        client.create_collection(
            collection_name=name, schema=schema, index_params=indexes,
            consistency_level="Strong",
        )
        for start in range(0, len(chunks), 100):
            rows = [{
                "id": row["metadata"]["chunk_id"], "vector": vectors[i].tolist(),
                "page_content": row["page_content"], "metadata": row["metadata"],
            } for i, row in enumerate(chunks[start:start + 100], start)]
            client.insert(collection_name=name, data=rows)
        client.flush(collection_name=name)
        client.load_collection(collection_name=name)
        count = int(client.get_collection_stats(collection_name=name)["row_count"])
        if count != len(chunks):
            raise ValueError("Milvus 写入数量不完整，新集合未启用，请重新构建")
        if previous:
            client.alter_alias(collection_name=name, alias=MILVUS_COLLECTION)
        else:
            client.create_alias(collection_name=name, alias=MILVUS_COLLECTION)
    except Exception:
        try:
            # 切换别名时即使响应丢失，也保留可能已经生效的集合。
            if client.has_collection(collection_name=name) and not client.list_aliases(
                collection_name=name
            ).get("aliases"):
                client.drop_collection(collection_name=name)
        except MilvusException:
            pass
        raise
    load_index.cache_clear()
    result = {"vectors": count, "dimensions": vectors.shape[1], "physical_collection": name,
              "logical_collection": MILVUS_COLLECTION, "status": "就绪"}
    if previous and previous.startswith(MILVUS_COLLECTION + "__"):
        try:
            aliases = client.list_aliases(collection_name=previous).get("aliases", [])
            if not aliases:
                client.drop_collection(collection_name=previous)
        except MilvusException:
            result["note"] = "新集合已就绪，旧集合清理未完成"
    return result


def search(question: str, top_k: int = 3, ef_search: int = EF_SEARCH, *, trace=None) -> list[dict]:
    if not question.strip() or min(top_k, ef_search) <= 0:
        raise ValueError("问题不能为空，top_k 和 ef_search 必须大于 0")
    if trace:
        trace.step("Milvus 集合准备（复用启动预加载）")
    load_index()
    if trace:
        trace.step("问题向量化（嵌入 API、校验与归一化）")
    vectors = embed_texts([question])
    if trace:
        trace.step("Milvus 向量检索与结果整理")
    hits = get_client().search(
        collection_name=MILVUS_COLLECTION, anns_field="vector", data=vectors.tolist(),
        limit=top_k,
        search_params={"metric_type": "COSINE", "params": {"ef": max(ef_search, top_k)}},
        output_fields=["page_content", "metadata"], consistency_level="Strong",
    )
    return [{
        "id": hit["id"], "page_content": hit["entity"]["page_content"],
        "metadata": hit["entity"]["metadata"], "score": float(hit["distance"]),
    } for hit in hits[0]]
