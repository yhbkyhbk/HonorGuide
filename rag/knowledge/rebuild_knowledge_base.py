"""从同一批攻略重建知识库，协调更新与正在读索引的请求。"""

from contextlib import contextmanager
from threading import Lock
from uuid import uuid4

from config import STORAGE
from .. import search_bm25_keywords as bm25
from .. import search_milvus_vectors as vector
from .. import search_neo4j_graph as graph
from .prepare_documents import prepare_documents


UPDATE_PENDING = STORAGE / ".knowledge_update_pending"
REVISION_FILE = STORAGE / "knowledge_revision.txt"
_state_lock = Lock()
_readers = 0
_updating = False


class KnowledgeUnavailableError(RuntimeError):
    """更新期间或更新失败后，知识库暂不可读。"""


def current_revision() -> str | None:
    """返回最近一次完整更新的标识。旧版知识库可能没有标识。"""
    try:
        return REVISION_FILE.read_text(encoding="ascii").strip()
    except FileNotFoundError:
        return None


def update_state() -> str:
    with _state_lock:
        if _updating:
            return "updating"
        return "pending" if UPDATE_PENDING.is_file() else "ready"


@contextmanager
def read_access():
    """允许并行查询，发布过程中阻止提交新请求。"""
    global _readers
    with _state_lock:
        if _updating:
            raise KnowledgeUnavailableError("知识库正在更新，请稍后再试")
        if UPDATE_PENDING.is_file():
            raise KnowledgeUnavailableError("知识库更新尚未完成，请在侧栏重新更新")
        _readers += 1
    try:
        yield
    finally:
        with _state_lock:
            _readers -= 1


def rebuild(on_step=None) -> dict:
    """依次处理攻略和三套索引。全部成功后恢复问答与测评。"""
    global _updating
    with _state_lock:
        if _updating:
            raise KnowledgeUnavailableError("知识库正在更新，请等待当前更新完成")
        if _readers:
            raise KnowledgeUnavailableError("还有问答或测评正在运行，请等它完成后再更新")
        _updating = True
    try:
        if on_step:
            on_step(1, "读取并切分攻略")
        UPDATE_PENDING.parent.mkdir(parents=True, exist_ok=True)
        UPDATE_PENDING.touch()
        parents, chunks = prepare_documents()
        if on_step:
            on_step(2, "重建 BM25 索引")
        keywords = bm25.build_index(chunks)
        if on_step:
            on_step(3, "重建 Milvus 向量索引")
        vectors = vector.build_index(chunks)
        if on_step:
            on_step(4, "更新 Neo4j 英雄关系")
        graph.start_database()
        relationships = graph.build_graph(parents)
        REVISION_FILE.write_text(uuid4().hex + "\n", encoding="ascii")
        UPDATE_PENDING.unlink()
        return {"documents": len(parents), "chunks": len(chunks), "terms": len(keywords.idf),
                "vectors": vectors, "relationships": relationships}
    finally:
        with _state_lock:
            _updating = False
