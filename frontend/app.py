"""HonorGuide 页面：攻略问答与 RAGAS 效果测评。"""

import math
from pathlib import Path
import sys
from uuid import uuid4

# 支持从项目根目录或 frontend 目录启动 Streamlit。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from streamlit.components.v2 import component
from neo4j.exceptions import AuthError, Neo4jError, ServiceUnavailable
from openai import APIError, APIStatusError
from pymilvus.exceptions import MilvusException

from config import TIMINGS_FILE
from rag import evaluate as evaluation, knowledge
from rag.bm25 import retriever as bm25
from rag.graph import retriever as graph
from rag.pipeline import RequestTrace, ask, save_timing
from rag.vector import retriever as vector


@st.cache_resource
def browser_clock_component():
    """采集浏览器首段渲染时间，并在生成时仅锁定提交，保留输入和草稿。"""
    return component("honorguide_browser_clock", js=r'''
export default function(component) {
    const state = window.__honorguideClock ||= {};
    state.send = value => component.setTriggerValue("measurement", value);
    const data = component.data || {};
    if (state.runId !== data.run_id) {
        state.runId = data.run_id;
        // 新一轮脚本没有接收问题时，解除因切页、停止等遗留的提交锁。
        if (!data.request_id) state.busy = false;
    }
    state.syncSubmit?.();
    if (state.installed) return;
    state.installed = true;
    const answers = () => [...document.querySelectorAll('[class*="st-key-hg_answer_"]')];
    const answerId = node => [...node.classList]
        .find(name => /^st-key-hg_answer_[a-f0-9]{32}$/.test(name))?.slice(17);
    state.syncSubmit = () => {
        const input = document.querySelector('[data-testid="stChatInputTextArea"]');
        const button = document.querySelector('[data-testid="stChatInputSubmitButton"]');
        if (!input || !button) return;
        const disabled = Boolean(state.busy || input.disabled || !input.value.trim());
        if (button.disabled !== disabled) button.disabled = disabled;
    };
    const submit = event => {
        const input = document.querySelector('[data-testid="stChatInputTextArea"]');
        if (!input || input.disabled) return;
        const keyboard = event.type === "keydown" && event.target === input &&
            event.key === "Enter" && !event.shiftKey && !event.isComposing;
        const button = event.type === "click" &&
            event.target.closest('[data-testid="stChatInputSubmitButton"]');
        if (!keyboard && !button) return;
        if (state.busy || (keyboard && event.repeat)) {
            event.preventDefault();
            event.stopImmediatePropagation();
            state.syncSubmit();
            return;
        }
        if (!input.value.trim()) return;
        state.busy = true;
        state.pending = {start: performance.now(), existing: new Set(answers().map(answerId))};
        // 先让本次原生提交完成，再锁定按钮，避免阻止第一次发送。
        queueMicrotask(state.syncSubmit);
    };
    const check = () => {
        state.syncSubmit();
        const p = state.pending;
        if (!p || p.sent) return;
        if (!p.id) {
            const node = answers().find(node => !p.existing.has(answerId(node)));
            if (node) p.id = answerId(node);
        }
        if (!p.id) return;
        const done = document.querySelector('.st-key-hg_done_' + p.id);
        if (done) {
            state.busy = false;
            state.syncSubmit();
        }
        const node = document.querySelector('.st-key-hg_answer_' + p.id);
        const text = node?.querySelector('[data-testid="stMarkdownContainer"]')?.textContent.trim();
        if (text && p.first === undefined && !p.scheduled && document.visibilityState === "visible") {
            p.scheduled = true;
            p.detected = performance.now() - p.start;
            // 双 rAF 近似首次文本绘制；不声称获得物理屏幕的精确刷新时刻。
            requestAnimationFrame(() => requestAnimationFrame(() => {
                if (state.pending !== p) return;
                p.scheduled = false;
                if (!node.isConnected || document.visibilityState !== "visible") {
                    check();
                    return;
                }
                p.first = performance.now() - p.start;
                check();
            }));
        }
        if (p.first !== undefined && done) {
            p.sent = true;
            state.send({request_id: p.id, browser_first_render_ms: p.first,
                browser_text_detected_ms: p.detected, browser_render_wait_ms: p.first - p.detected});
        }
    };
    document.addEventListener("keydown", submit, true);
    document.addEventListener("click", submit, true);
    document.addEventListener("input", state.syncSubmit, true);
    document.addEventListener("visibilitychange", check);
    state.observer = new MutationObserver(check);
    state.observer.observe(document.body, {subtree: true, childList: true, characterData: true,
        attributes: true, attributeFilter: ["disabled"]});
    check();
}
''')


def record_request_start():
    st.session_state["request_trace"] = RequestTrace()


def receive_browser_timing():
    trace = st.session_state.get("request_trace")
    clock = browser_clock_component()(key="browser_clock", height=0,
                                      data={"run_id": uuid4().hex,
                                            "request_id": trace.request_id if trace else None},
                                      on_measurement_change=lambda: None)
    measurement = clock.measurement
    if not isinstance(measurement, dict):
        return
    value = measurement.get("browser_first_render_ms")
    if type(value) not in (int, float) or not 0 <= value < math.inf:
        return
    for entry in st.session_state.history:
        timings = entry.get("result", {}).get("timings", entry.get("timings", {}))
        if timings.get("request_id") == measurement.get("request_id"):
            if "browser_first_render_ms" not in timings:
                timings["browser_first_render_ms"] = value
                for key in ("browser_text_detected_ms", "browser_render_wait_ms"):
                    duration = measurement.get(key)
                    if type(duration) in (int, float) and 0 <= duration <= value:
                        timings[key] = duration
                server_ms = timings.get("server_first_dispatch_ms")
                if server_ms is not None:
                    timings["transport_scheduling_render_ms"] = value - server_ms
                persist_timing(timings, "browser_first_render", "error" if "error" in entry else "ok")
            break


def persist_timing(timings, event="server_complete", status="ok"):
    try:
        save_timing(timings, event, status)
    except OSError:
        st.warning("耗时日志写入失败，当前会话仍可查看计时结果。")


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
        log_path = TIMINGS_FILE.relative_to(PROJECT_ROOT).as_posix()
        st.caption(f"请求编号：{timings['request_id']} · 日志：{log_path}")

MODE_NAMES = {
    "hybrid": "文本混合检索",
    "graph": "图检索", "combined": "文本 + 图联合检索",
}


def error_message(exc: Exception) -> str:
    """展示可操作的提示，避免把接口原始响应和连接凭据显示到页面。"""
    if isinstance(exc, knowledge.KnowledgeUnavailableError):
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


@st.cache_resource
def ensure_graph_database() -> None:
    """页面首次加载时检查并启动图数据库，页面重跑时复用结果。"""
    graph.start_database()


@st.cache_resource(show_spinner="正在加载检索索引……")
def prepare_search_indexes() -> tuple[str, ...]:
    """页面首次打开时预热 BM25 和 Milvus。"""
    warnings = []
    with knowledge.read_access():
        try:
            bm25.load_index()
        except (OSError, ValueError, TypeError, KeyError):
            warnings.append("BM25 索引未就绪，请在侧栏更新知识库。")
        try:
            if vector.get_client().has_collection(collection_name=vector.MILVUS_COLLECTION):
                vector.load_index()
                return tuple(warnings)
        except (MilvusException, OSError, ValueError, ImportError):
            pass
    warnings.append("Milvus 向量索引未就绪。首次使用请在侧栏更新知识库，已有索引时请检查 Milvus 连接配置。")
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
        prepare_search_indexes.clear()
        progress.empty()
        st.success(
            f"知识库已更新。{result['documents']} 篇攻略 · {result['chunks']} 个文本块 · "
            f"{result['terms']} 个 BM25 词项 · {result['vectors']} 条向量 · "
            f"{result['relationships']} 条图谱关系"
        )


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
    if result.get("routing", {}).get("reason"):
        st.caption("检索选择依据：" + result["routing"]["reason"])
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


def show_chat(ready: bool) -> None:
    st.caption("LLM 根据问题语义自动选择检索方式；每次提问独立检索，历史记录仅用于展示。")
    for entry in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(entry["question"])
        with st.chat_message("assistant"):
            if "result" in entry:
                show_result(entry["result"])
            else:
                if entry.get("partial_answer"):
                    with st.container(key=f"hg_answer_{entry['timings']['request_id']}"):
                        st.markdown(entry["partial_answer"])
                    st.warning("以上内容生成中断或未通过引用校验，请勿视为完整回答。")
                st.error(entry["error"])
                show_timings(entry.get("timings", {}))
    question = st.chat_input("例如：为什么艾琳克制虞姬，虞姬应该如何调整装备？", disabled=not ready,
                             key="chat_question", on_submit=record_request_start, submit_mode="submit")
    if not question:
        return
    question = question.strip()
    if not question:
        st.warning("请输入问题内容。")
        return
    entry = {"question": question}
    trace = st.session_state.pop("request_trace", None) or RequestTrace()
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.container(key=f"hg_answer_{trace.request_id}"):
            output = st.empty()
        pieces = []
        def display_token(text):
            pieces.append(text)
            output.markdown("".join(pieces))
        try:
            with st.spinner("正在检索资料并生成回答……"):
                entry["result"] = ask(question, on_token=display_token, trace=trace)
        except (APIError, MilvusException, Neo4jError, ServiceUnavailable, OSError, ValueError, RuntimeError, ImportError, KeyError) as exc:
            entry["error"] = error_message(exc)
            entry["partial_answer"] = "".join(pieces)
            trace.mark("failed")
            trace.step(None)
            entry["timings"] = trace.snapshot()
            if pieces:
                st.warning("以上内容生成中断或未通过引用校验，请勿视为完整回答。")
            st.error(entry["error"])
            show_timings(entry["timings"])
        else:
            show_result(entry["result"], render_answer=False)
    st.session_state.history.append(entry)
    timings = entry.get("result", {}).get("timings", entry.get("timings", {}))
    persist_timing(timings, status="error" if "error" in entry else "ok")
    # 完成标记在历史保存之后下发；浏览器此时回传不会打断正在生成的回答。
    with st.container(key=f"hg_done_{trace.request_id}"):
        st.empty()


def percent(value) -> str:
    return "—" if value is None else f"{value:.1%}"


def show_evaluation(ready: bool) -> None:
    st.subheader("效果测评")
    try:
        data = evaluation.load_dataset()
    except (OSError, ValueError, KeyError) as exc:
        st.error(error_message(exc))
        return
    settings = data.get("run_settings", data["protocol"]["run_settings"])
    with st.expander("测评检索参数"):
        top_k = st.slider("最终文本块数", 1, 10, settings["top_k"], key="evaluation_top_k")
        recall_k = st.slider("每路文本召回数", 10, 30, settings["recall_k"], key="evaluation_recall_k")
        st.caption("仅用于本次测评，问答使用默认 3 / 10。检索方式始终由 LLM 选择；"
                   "参数变化后需点击“重新评测”。下方已有指标仍对应保存结果时的参数。")
    st.dataframe(
        [{"编号": row["id"], "问题": row["question"], "资料中有答案": row["answerable"]} for row in data["cases"]],
        hide_index=True,
    )
    st.caption("使用 RAGAS 0.4.4。继续评测复用已有回答和有效的逐指标断点；旧自定义分数会单独归档，"
               "并重新计算 RAGAS 指标。重新评测会重新检索、生成回答并覆盖结果。")
    left, right = st.columns(2)
    resume = left.button("运行 / 继续评测", key="run_evaluation", disabled=not ready)
    restart = right.button("重新评测（覆盖结果）", key="restart_evaluation", disabled=not ready)
    if resume or restart:
        progress = st.progress(0.0, text="准备评测")
        def update(done, total, case_id, status):
            progress.progress(done / total if total else 1.0, text=f"{case_id} · {status}（{done}/{total}）")
        try:
            summary = evaluation.run(restart=restart, progress=update, top_k=top_k, recall_k=recall_k)
            data = evaluation.load_dataset()
            if summary["completed"] == summary["total"]:
                st.success(f"已完成 {summary['completed']} 道题。")
            else:
                st.warning("部分题目未完成，断点已保存；可点击“运行 / 继续评测”重试。")
        except (APIError, MilvusException, Neo4jError, ServiceUnavailable, OSError, ValueError, RuntimeError, ImportError, KeyError) as exc:
            st.error(error_message(exc))
    summary = data.get("summary")
    if summary and data.get("engine", {}).get("name") == "ragas":
        columns = st.columns(4)
        for column, (name, label) in zip(columns, (
            ("faithfulness", "Faithfulness · 忠实度"),
            ("answer_relevancy", "Answer Relevancy · 相关性"),
            ("context_precision", "Context Precision · 上下文精度"),
            ("context_recall", "Context Recall · 上下文召回"),
        )):
            column.metric(label, percent(summary.get(name)))
            count = summary.get("metric_counts", {}).get(name, {})
            column.caption(f"有效 {count.get('scored', 0)} 题 / 不适用 {count.get('not_applicable', 0)} 题")
        st.caption("以上为 RAGAS 内置指标，按有效题取平均。无答案题或无证据时，忠实度和上下文指标不适用。"
                   "相关性分可能惩罚恰当拒答，需结合拒答通过率阅读。")
        left, middle, right = st.columns(3)
        retrieval_k = summary.get("retrieval_k", 3)
        left.metric(f"检索检查 · 文本 Hit@{retrieval_k}",
                    percent(summary.get("text_hit_at_k", summary.get("text_hit_at_3"))))
        middle.metric("引用审计 · 支持率", percent(summary.get("citation_precision")))
        right.metric("引用审计 · 覆盖率", percent(summary.get("citation_recall")))
        st.caption("检索检查是项目规则；引用审计通过 RAGAS 的结构化 LLM 接口完成，属于项目扩展。"
                   "Faithfulness 不等于引用支持率，Context Precision 也不等于 Precision@K。")
        with st.expander("全部指标"):
            st.json(summary)
    elif summary:
        st.info("当前展示的是历史自定义评测记录。点击“运行 / 继续评测”后，将对已有回答重新计算 RAGAS 分数。")
    rows = data.get("results", [])
    if rows:
        selected = st.selectbox("查看单题结果", [row["id"] for row in rows], key="evaluation_case")
        row = next(item for item in rows if item["id"] == selected)
        if row.get("error"):
            st.warning(f"该题未完成：{row['error']}")
        if "output" in row:
            show_result(row["output"])
            with st.expander("RAGAS 评测输入与参考答案"):
                case = next(case for case in data["cases"] if case["id"] == selected)
                st.json(evaluation.to_sample(case, row["output"]))
        with st.expander("评审与相关性记录"):
            st.json({key: value for key, value in row.items() if key != "output"})


def main() -> None:
    st.set_page_config(page_title="HonorGuide · 王者荣耀攻略助手", page_icon="🎮", layout="wide")
    st.title("🎮 HonorGuide")
    st.caption("王者荣耀攻略问答 · 艾琳 / 虞姬")
    st.session_state.setdefault("history", [])
    receive_browser_timing()
    try:
        with st.spinner("正在检查图数据库……"):
            ensure_graph_database()
    except (Neo4jError, ServiceUnavailable, ValueError, OSError) as exc:
        st.warning(error_message(exc) + " 自动或联合模式可在图查询失败时回退到文本检索。")
    with st.sidebar:
        page = st.radio("功能", ("攻略问答", "效果测评"), key="page")
        show_knowledge_update()
    ready = knowledge.update_state() == "ready"
    if ready and not (bm25.load_index.cache_info().currsize and vector.load_index.cache_info().currsize):
        try:
            index_warnings = prepare_search_indexes()
        except knowledge.KnowledgeUnavailableError:
            ready = False
        else:
            for warning in index_warnings:
                st.warning(warning)
    if not ready:
        st.info("知识库更新完成后可继续问答和测评。")
    if page == "攻略问答":
        show_chat(ready)
    else:
        show_evaluation(ready)


if __name__ == "__main__":
    main()
