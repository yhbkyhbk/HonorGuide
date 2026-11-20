"""构建攻略的 Milvus HNSW 索引并检索。"""

from __future__ import annotations

import re
from uuid import uuid4

from pymilvus import DataType
from pymilvus.exceptions import MilvusException

from config import MILVUS_COLLECTION
from infrastructure.call_model_api import embed_texts

from .connect_milvus import get_client, load_collection
from .knowledge.prepare_documents import load_chunks

HNSW_M = 16
EF_CONSTRUCTION = 100
EF_SEARCH = 32


def load_index() -> None:
    load_collection(MILVUS_COLLECTION)


def build_index(chunks: list[dict] | None = None) -> int:
    """先建新集合，完整写入、索引并加载成功后切换固定别名。"""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,100}", MILVUS_COLLECTION):
        raise ValueError("MILVUS_COLLECTION 应使用不超过 101 位的英文字母、数字和下划线")
    chunks = load_chunks() if chunks is None else chunks
    ids = {row["metadata"]["chunk_id"] for row in chunks}
    if not ids or len(ids) != len(chunks):
        raise ValueError("文本块为空或编号重复，请重新处理文档")
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
    load_collection.cache_clear()
    if previous and previous.startswith(MILVUS_COLLECTION + "__"):
        try:
            aliases = client.list_aliases(collection_name=previous).get("aliases", [])
            if not aliases:
                client.drop_collection(collection_name=previous)
        except MilvusException:
            # 新别名已经生效，旧集合清理失败不影响查询。
            pass
    return count


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
