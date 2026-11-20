"""提供 RAGAS 测评及结果查看页面。"""

import streamlit as st

from evaluation import evaluate_with_ragas as evaluation
from .render_answer import REQUEST_ERRORS, error_message, show_result


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
    st.caption("使用 RAGAS 0.4.4 测评问答 Agent。首次测评 Agent 或修改回答 Skill 后，请重新评测。"
               "继续评测复用同一版本的答案和有效断点。")
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
        except REQUEST_ERRORS as exc:
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
        st.info("当前是历史评测结果。点击“重新评测（覆盖结果）”可按当前 Agent 生成答案并评分。")
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
