"""统一问答模块：规则路由、查询规划及检索到生成的执行流程。"""

import re
import json
from datetime import datetime, timezone
from time import perf_counter
from uuid import uuid4

from config import STORAGE

from neo4j.exceptions import Neo4jError, ServiceUnavailable

from .fusion import hybrid_search
from .generation import build_evidence, generate_answer
from .graph import retriever as graph_retrieval

MODES = ("auto", "hybrid", "graph", "combined")


class RequestTrace:
    """用同一个单调时钟记录顺序阶段；阶段互不重叠，可以直接相加。"""

    def __init__(self):
        self.request_id = uuid4().hex
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.started = perf_counter()
        self.stages = []
        self.marks = {}
        self.current = ("页面准备与索引检查", 0.0)

    def elapsed(self):
        return (perf_counter() - self.started) * 1000

    def step(self, name):
        now = self.elapsed()
        if self.current:
            label, start = self.current
            self.stages.append({"stage": label, "start_ms": start, "end_ms": now})
        self.current = (name, now) if name else None

    def mark(self, name):
        self.marks.setdefault(name, self.elapsed())

    def snapshot(self):
        now = self.elapsed()
        stages = list(self.stages)
        if self.current:
            label, start = self.current
            stages.append({"stage": label, "start_ms": start, "end_ms": now})
        cutoff = self.marks.get("server_first_dispatch", self.marks.get("first_text"))
        before_first = []
        if cutoff is not None:
            for row in stages:
                end = min(row["end_ms"], cutoff)
                if end > row["start_ms"]:
                    before_first.append({**row, "end_ms": end,
                                         "duration_ms": end - row["start_ms"]})
        model_start = self.marks.get("model_request")
        model_first = self.marks.get("model_first_text")
        return {
            "request_id": self.request_id, "created_at": self.created_at,
            "first_text_ms": self.marks.get("first_text"),
            "server_first_dispatch_ms": self.marks.get("server_first_dispatch"),
            "model_ttft_ms": model_first - model_start
                if model_first is not None and model_start is not None else None,
            "complete_ms": self.marks.get("complete"),
            "failed_ms": self.marks.get("failed"),
            "first_token_stages": before_first,
            "stages": [{**r, "duration_ms": r["end_ms"] - r["start_ms"]} for r in stages],
        }


def save_timing(timings: dict, event: str = "server_complete", status: str = "ok") -> None:
    """每次请求写一次服务端记录；浏览器回报后追加同一 request_id 的渲染记录。"""
    STORAGE.mkdir(parents=True, exist_ok=True)
    with (STORAGE / "request_timings.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps({"event": event, "status": status, **timings}, ensure_ascii=False) + "\n")


RELATION_WORDS = ("克制", "克星", "相克")


STRATEGY_WORDS = (
    "出装", "装备", "铭文", "连招", "打法", "站位", "发育", "清线", "补刀",
    "团战", "怎么玩", "怎么打", "怎么应对", "如何应对", "应对策略", "使用技巧",
    "技能", "被动", "普攻", "位移",
)


def choose_mode(question: str, requested: str = "auto") -> tuple[str, str]:
    """手动指定优先；auto 根据关系词和攻略主题词选择，路由本身不调用 LLM。"""
    if requested not in MODES:
        raise ValueError(f"不支持的模式：{requested}")
    if requested != "auto":
        return requested, "用户手动指定"
    has_relation = any(word in question for word in RELATION_WORDS)
    has_strategy = any(word in question for word in STRATEGY_WORDS)
    if has_relation and has_strategy:
        return "combined", "问题同时涉及克制关系与出装、打法等攻略内容"
    if has_relation:
        return "graph", "问题涉及克制关系，可查询知识图谱中的方向、原因和条件"
    return "hybrid", "问题不含已支持的关系关键词，使用文本混合检索"


def build_text_query(question: str, graph_facts: list[dict]) -> str:
    clauses = [part.strip() for part in re.split(r"[，,；;。？！!?]", question)]
    strategy_query = "，".join(
        part for part in clauses if any(word in part for word in STRATEGY_WORDS)
    )
    heroes = {name for fact in graph_facts for name in (fact["head"], fact["tail"])}
    if strategy_query and any(hero in strategy_query for hero in heroes):
        return strategy_query
    return question


def ask(
    question: str, mode: str = "auto", top_k: int = 3, recall_k: int = 10,
    *, on_token=None, trace: RequestTrace | None = None,
) -> dict:
    """选择模式、检索、组织证据、生成答案，并返回可供调试的实际执行信息。"""
    if not question.strip() or min(top_k, recall_k) <= 0:
        raise ValueError("问题不能为空，top_k 和 recall_k 必须大于 0")
    question = question.strip()
    started = perf_counter()
    trace = trace or RequestTrace()
    trace.step("问题路由")
    selected, reason = choose_mode(question, mode)
    used = selected
    text_results, graph_facts, graph_plan, notes = [], [], None, []
    graph_status = "not_requested"
    text_query = question
    timings = {}

    if selected in ("graph", "combined"):
        stage = perf_counter()
        try:
            graph_plan, graph_facts = graph_retrieval.retrieve(question, trace=trace)
            graph_status = "ok" if graph_facts else "no_match"
        except (ServiceUnavailable, Neo4jError, ValueError, OSError) as exc:
            # 手动指定 graph 时直接报告错误；auto/combined 保留文本检索能力。
            if mode == "graph":
                raise
            graph_status = "unsupported" if isinstance(exc, ValueError) else "unavailable"
            detail = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
            notes.append(f"图检索未完成（{detail}），本次回退到 hybrid")
            used = "hybrid"
        timings["graph_seconds"] = round(perf_counter() - stage, 3)

    if used in ("hybrid", "combined"):
        stage = perf_counter()
        trace.step("文本查询规划")
        if used == "combined":
            text_query = build_text_query(question, graph_facts)
        text_results = hybrid_search(text_query, top_k, recall_k, trace=trace)
        timings["text_seconds"] = round(perf_counter() - stage, 3)

    trace.step("证据整理")
    evidence = build_evidence(text_results, graph_facts)
    stage = perf_counter()
    answer = generate_answer(question, evidence, used, graph_status, on_token=on_token, trace=trace)
    timings["generation_seconds"] = round(perf_counter() - stage, 3)
    timings["total_seconds"] = round(perf_counter() - started, 3)
    trace.mark("complete")
    trace.step(None)
    timings.update(trace.snapshot())
    timings.update(requested_mode=mode, used_mode=used)
    return {
        "requested_mode": mode, "selected_mode": selected, "used_mode": used,
        "route_reason": reason, "notes": notes,
        "graph_status": graph_status, "graph_used": bool(graph_facts),
        "graph_plan": graph_plan, "graph_relations": len(graph_facts),
        "text_query": text_query if used in ("hybrid", "combined") else None,
        "text_candidates": [
            {"chunk_id": row["metadata"]["chunk_id"], "rrf_score": row["score"],
             "ranks": row.get("ranks", {}), "raw_scores": row.get("raw_scores", {})}
            for row in text_results
        ],
        "evidence": evidence, "answer": answer, "timings": timings,
    }
