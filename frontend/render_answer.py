"""为问答和测评统一显示回答、引用、耗时与错误。"""

import json

from neo4j.exceptions import AuthError, Neo4jError, ServiceUnavailable
from openai import APIError, APIStatusError
from pymilvus.exceptions import MilvusException
import streamlit as st

from config import ROOT, TIMINGS_FILE
from rag.knowledge.rebuild_knowledge_base import KnowledgeUnavailableError
from agent.run_qa_agent import GRAPH_TOOL, GUIDE_TOOL


REQUEST_ERRORS = (
    APIError, MilvusException, Neo4jError, ServiceUnavailable,
    OSError, ValueError, RuntimeError, ImportError, KeyError,
)


def show_timings(timings):
    if "request_id" not in timings:
        return  # 兼容尚未记录首 token 耗时的历史评测。
    with st.expander("首 token 耗时与阶段明细"):
        browser, server, model = st.columns(3)
        def seconds(value):
            return "尚未采集" if value is None else f"{value / 1000:.3f} 秒"
        browser.metric("提交 → 浏览器首段渲染", seconds(timings.get("browser_first_render_ms")))
        server.metric("服务端收到 → 首段下发", seconds(timings.get("server_first_dispatch_ms")))
        model.metric("模型请求 → 首段文本", seconds(timings.get("model_ttft_ms")))
        st.caption("浏览器以双 requestAnimationFrame 近似首次绘制；服务端从提交回调开始计时。"
                   "首 token 指首个非空回答文本增量，一次可能包含多个 token。无证据时的本地提示不计模型耗时。")
        rows = timings.get("first_token_stages", [])
        if rows:
            st.dataframe([{"阶段": row["stage"], "耗时（毫秒）": round(row["duration_ms"], 3)}
                          for row in rows], hide_index=True, width="stretch")
            st.caption(f"首段前各阶段合计：{sum(row['duration_ms'] for row in rows):.3f} 毫秒。"
                       "主表各阶段互不重叠。")
        branch_rows = [
            {"分支": name, "阶段": row["stage"], "耗时（毫秒）": round(row["duration_ms"], 3)}
            for name, stages in timings.get("parallel_branches", {}).items()
            for row in stages
        ]
        if branch_rows:
            st.caption("并行召回明细。两路耗时可能重叠，实际等待时间已计入主表。")
            st.dataframe(branch_rows, hide_index=True, width="stretch")
        gap = timings.get("transport_scheduling_render_ms")
        if gap is not None:
            st.caption(f"浏览器总耗时与服务端首段下发耗时之差：{gap:.3f} 毫秒，"
                       "包含提交传输、调度、返回传输和渲染等开销，无法据此单独拆分网络耗时。")
        if timings.get("complete_ms") is not None:
            st.caption(f"服务端完整回答耗时：{timings['complete_ms'] / 1000:.3f} 秒（另计）。")
        log_path = TIMINGS_FILE.relative_to(ROOT).as_posix()
        st.caption(f"请求编号：{timings['request_id']} · 日志：{log_path}")

MODE_NAMES = {
    "hybrid": "文本混合检索",
    "graph": "图检索", "combined": "文本 + 图联合检索",
}
TOOL_NAMES = {GUIDE_TOOL: "查攻略文本", GRAPH_TOOL: "查英雄克制"}
TOOL_STATUS = {"ok": "已找到", "empty": "无匹配", "error": "失败"}


def error_message(exc: Exception) -> str:
    """展示可操作的提示，避免把接口原始响应和连接凭据显示到页面。"""
    if isinstance(exc, KnowledgeUnavailableError):
        return str(exc)
    if isinstance(exc, MilvusException):
        return "Milvus 操作失败，请检查连接配置。若修改了攻略或嵌入模型，请在侧栏更新知识库。"
    if isinstance(exc, AuthError):
        return "图数据库认证失败，请检查 .env 中的 Neo4j 用户名和密码。"
    if isinstance(exc, ServiceUnavailable):
        return "无法连接图数据库，请检查 Neo4j 服务及 .env 中的连接配置。"
    if isinstance(exc, Neo4jError):
        return "图数据库查询失败，请检查数据库状态和已导入的英雄关系。"
    if isinstance(exc, APIStatusError):
        return f"模型接口请求失败（HTTP {exc.status_code}），请检查模型配置、权限和额度。"
    if isinstance(exc, APIError):
        return "模型接口连接失败，请检查网络与 .env 中的接口配置后重试。"
    if isinstance(exc, FileNotFoundError):
        return "缺少本地数据或索引，请检查攻略文件，并在侧栏更新知识库。"
    if isinstance(exc, ValueError):
        return f"无法完成请求：{exc}"
    return f"请求未完成（{type(exc).__name__}），请检查当前环境和本地索引。"


def show_result(result: dict, render_answer: bool = True) -> None:
    if render_answer:
        request_id = result["timings"].get("request_id")
        with st.container(key=f"hg_answer_{request_id}" if request_id else None):
            st.markdown(result["answer"])
    text_count = sum(item["type"] == "text" for item in result["evidence"])
    complete_ms = result["timings"].get("complete_ms")
    full_seconds = complete_ms / 1000 if complete_ms is not None else result["timings"]["total_seconds"]
    st.caption(
        f"本次使用：{MODE_NAMES[result['used_mode']]} · "
        f"{text_count} 个文本块 / {result['graph_relations']} 条图关系 · "
        f"完整回答 {full_seconds:.2f} 秒"
    )
    if result.get("tool_trace"):
        with st.expander("Agent 工具调用"):
            st.dataframe([
                {"步骤": "首查" if item["round"] == 1 else "补查",
                 "工具": TOOL_NAMES[item["tool"]],
                 "查询参数": json.dumps(item["arguments"], ensure_ascii=False),
                 "结果": TOOL_STATUS[item["status"]], "命中数": item["matches"],
                 "耗时（毫秒）": item["duration_ms"]}
                for item in result["tool_trace"]
            ], hide_index=True, width="stretch")
    for note in result["notes"]:
        st.warning(note)
    show_timings(result["timings"])
    if result["graph_status"] == "no_match":
        st.info("图谱未收录匹配的关系，这不代表实际对局中不存在该关系。")

    with st.expander(f"引用与原文（{len(result['evidence'])} 份证据）"):
        if not result["evidence"]:
            st.write("本次没有检索到可用证据。")
        for item in result["evidence"]:
            st.markdown(f"**[{item['id']}] {'文本资料' if item['type'] == 'text' else '图谱资料'}**")
            st.caption(f"来源：{item['source_file']} · 文档：{item['source_doc_id']}")
            if item["type"] == "text":
                st.markdown(item["text"])
                st.caption(f"分块：{item['chunk_id']}")
            else:
                st.markdown(f"**{item['head']} → 克制 → {item['tail']}**")
                st.write("原因：", item["reason"])
                st.write("适用条件：", item["condition"])
                st.write("原文：", item["evidence_quote"])
            for index, url in enumerate(item["source_urls"], 1):
                st.markdown(f"[参考来源 {index}]({url})")
            st.divider()
