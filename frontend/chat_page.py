"""运行问答页面，保存会话并接收浏览器计时。"""

import math
from pathlib import Path
import sqlite3
from uuid import uuid4

import streamlit as st
from streamlit.components.v2 import component

from agent.run_qa_agent import ask
from evaluation.record_request_timing import RequestTrace, save_timing
from memory import load_history, save_turn, select_context

from .render_answer import REQUEST_ERRORS, error_message, show_result, show_timings


@st.cache_resource
def browser_clock_component():
    """采集浏览器首段渲染时间，并在生成时仅锁定提交，保留输入和草稿。"""
    script = Path(__file__).with_name("browser_clock.js").read_text(encoding="utf-8")
    return component("honorguide_browser_clock", js=script)


def record_request_start():
    st.session_state["request_trace"] = RequestTrace()


def entry_timings(entry: dict) -> dict:
    return entry["result"]["timings"] if "result" in entry else entry["timings"]


def persist_chat_turn(entry: dict, *, index_vector: bool = False) -> None:
    try:
        indexed = save_turn(entry_timings(entry)["request_id"], entry, index_vector=index_vector)
    except (OSError, sqlite3.Error):
        st.warning("本轮对话暂未写入本地记忆，重启后可能无法恢复。请检查 data/memory 的写入权限。")
        return
    if not indexed:
        st.warning("本轮问答已保存，记忆向量暂未写入。后续问答会自动补写。")


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
        timings = entry_timings(entry)
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
                persist_chat_turn(entry)
            break


def persist_timing(timings, event="server_complete", status="ok"):
    try:
        save_timing(timings, event, status)
    except OSError:
        st.warning("耗时日志写入失败，当前会话仍可查看计时结果。")


def initialize_chat() -> None:
    if "history" not in st.session_state:
        try:
            st.session_state.history = load_history()
        except (OSError, ValueError, sqlite3.Error):
            st.session_state.history = []
            st.warning("暂时无法读取本地对话记忆，刷新页面可重试。")
    receive_browser_timing()


def show_chat(ready: bool) -> None:
    st.caption("Agent 会先选一个工具查资料，根据结果决定是否补查一次。")
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
                trace.step("记忆召回与上下文选择")
                entry["result"] = ask(
                    question, on_token=display_token, trace=trace,
                    conversation_history=select_context(question, st.session_state.history),
                )
        except REQUEST_ERRORS as exc:
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
    persist_chat_turn(entry, index_vector=True)
    persist_timing(entry_timings(entry), status="error" if "error" in entry else "ok")
    # 完成标记在历史保存之后下发；浏览器此时回传不会打断正在生成的回答。
    with st.container(key=f"hg_done_{trace.request_id}"):
        st.empty()
