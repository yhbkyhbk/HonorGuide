"""启动 HonorGuide 页面并提供知识库更新入口。"""

from pathlib import Path
import sys

# 支持从项目根目录或 frontend 目录启动 Streamlit。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neo4j.exceptions import Neo4jError, ServiceUnavailable
from pymilvus.exceptions import MilvusException
import streamlit as st

from rag import search_bm25_keywords as bm25
from rag import search_milvus_vectors as vector
from rag import search_neo4j_graph as graph
from rag.knowledge import rebuild_knowledge_base as knowledge
from frontend.chat_page import initialize_chat, show_chat
from frontend.evaluation_page import show_evaluation
from frontend.render_answer import error_message


@st.cache_resource(show_spinner="正在检查图数据库……")
def ensure_graph_database() -> None:
    """页面首次加载时检查并启动图数据库，页面重跑时复用结果。"""
    graph.start_database()


def prepare_search_indexes() -> tuple[str, ...]:
    """使用检索模块的缓存预热并报告索引状态。"""
    warnings = []
    try:
        bm25.load_index()
    except (OSError, ValueError, TypeError, KeyError):
        warnings.append("BM25 索引未就绪，请在侧栏更新知识库。")
    try:
        vector.load_index()
    except (MilvusException, OSError, ValueError, ImportError):
        warnings.append("Milvus 向量索引未就绪。首次使用请更新知识库，已有索引时请检查连接配置。")
    return tuple(warnings)


def show_knowledge_update() -> None:
    state = knowledge.update_state()
    if not st.button(
        "更新知识库", key="update_knowledge", disabled=state == "updating", width="stretch",
        help="重新读取攻略并构建 BM25、Milvus 和 Neo4j 数据。部分步骤会调用模型接口。",
    ):
        if state == "pending":
            st.warning("上次更新未完成，请重新更新后再使用问答或测评。")
        return

    progress = st.progress(0, text="准备更新知识库……")
    try:
        result = knowledge.rebuild(
            lambda step, label: progress.progress((step - 1) / 4, text=f"{step}/4 · {label}"),
        )
    except Exception as exc:
        st.error("知识库更新未完成。" + error_message(exc))
    else:
        progress.empty()
        st.success(
            f"知识库已更新。{result['documents']} 篇攻略 · {result['chunks']} 个文本块 · "
            f"{result['terms']} 个 BM25 词项 · {result['vectors']} 条向量 · "
            f"{result['relationships']} 条图谱关系"
        )


def main() -> None:
    st.set_page_config(page_title="HonorGuide · 王者荣耀攻略助手", page_icon="🎮", layout="wide")
    st.title("🎮 HonorGuide")
    st.caption("王者荣耀攻略问答 · 艾琳 / 虞姬")
    with st.sidebar:
        page = st.radio("功能", ("攻略问答", "效果测评"), key="page")
        show_knowledge_update()
    ready = knowledge.update_state() == "ready"
    if ready:
        try:
            with knowledge.read_access():
                try:
                    ensure_graph_database()
                except (Neo4jError, ServiceUnavailable, ValueError, OSError) as exc:
                    st.warning(error_message(exc) + " Agent 可继续查攻略文本。")
                index_warnings = prepare_search_indexes()
        except knowledge.KnowledgeUnavailableError:
            ready = False
        else:
            for warning in index_warnings:
                st.warning(warning)
    if not ready:
        st.info("知识库更新完成后可继续问答和测评。")
    if page == "攻略问答":
        initialize_chat()
        show_chat(ready)
    else:
        show_evaluation(ready)


if __name__ == "__main__":
    main()
