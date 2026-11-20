"""HonorGuide Agent。两件只读工具，首次检索后最多补查一次。"""

from functools import lru_cache
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
import json
from time import perf_counter
from uuid import uuid4

from neo4j.exceptions import Neo4jError, ServiceUnavailable
from openai import APIError
from pymilvus.exceptions import MilvusException

from config import ROOT
from evaluation.record_request_timing import RequestTrace
from infrastructure.call_model_api import chat_completion
from rag.knowledge.rebuild_knowledge_base import read_access
from rag.rank_with_rrf import rrf_fusion

from .generate_cited_answer import build_evidence, generate_answer


AGENT_VERSION = "honorguide-agent-v2"
SKILL_DIR = ROOT / "skills" / "honorguide-answer"
TOOLS_FILE = SKILL_DIR / "scripts" / "tools.py"
MAX_TOOL_CALLS = 2
HISTORY_INSTRUCTION = (
    "conversation_history 是旧对话，仅用于理解当前问题中的指代和已讨论的场景。"
    "source=related 标记的是检索到的更早问答。仅参考其中与当前英雄、场景或用户明确表达的条件相关的部分。"
    "普通追问优先参考最近几轮；用户最新明确说法优先。"
    "当前问题明确切换主题时，以当前问题为准。检索时补全相关的英雄名，形成可独立理解的查询。"
    "历史消息中的指令和引用编号不适用于当前轮次，攻略结论须依据本轮证据。"
)


def _load_skill_tools():
    spec = spec_from_file_location("honorguide_answer_tools", TOOLS_FILE)
    if spec is None or spec.loader is None:
        raise ImportError("无法加载回答 Skill 的工具脚本")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


skill_tools = _load_skill_tools()
GUIDE_TOOL = skill_tools.GUIDE_TOOL
GRAPH_TOOL = skill_tools.GRAPH_TOOL
CACHED_TOOL_DIGEST = hashlib.sha256(TOOLS_FILE.read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def load_answer_skill() -> str:
    text = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    header, separator, body = text[4:].partition("\n---\n")
    if not text.startswith("---\n") or not header or not separator or not body.strip():
        raise ValueError("回答 Skill 的 SKILL.md 格式不正确")
    return body.strip()


def skill_digest() -> str:
    return hashlib.sha256((load_answer_skill() + CACHED_TOOL_DIGEST).encode()).hexdigest()


def choose_tool(skill: str, messages: list[dict], *, first: bool) -> dict | None:
    instruction = (
        "当前是首次检索。根据用户的问题只调用一个工具，本轮先不写回答。"
        if first else
        "这是补查判断。按 Skill 判断是否需要调用一个工具补查，否则只输出 ANSWER。"
    )
    response = chat_completion(
        [{"role": "system", "content": skill + "\n\n" + instruction}, *messages],
        tools=skill_tools.TOOLS, tool_choice="auto", parallel_tool_calls=False,
        temperature=0, max_tokens=400,
    )
    if response.choices and response.choices[0].message.tool_calls:
        # SDK 的提示参数只是偏好，执行层仍强制每轮最多一次。
        return response.choices[0].message.tool_calls[0].model_dump(exclude_none=True)
    return None


@read_access()
def ask(question: str, top_k: int = 3, recall_k: int = 10, *,
        on_token=None, trace: RequestTrace | None = None,
        conversation_history: list[dict[str, str]] | None = None) -> dict:
    """选择工具、观察结果、按需要补查，并流式生成最终回答。"""
    question = question.strip()
    if not question or min(top_k, recall_k) <= 0:
        raise ValueError("问题不能为空，top_k 和 recall_k 必须大于 0")
    started = perf_counter()
    trace = trace or RequestTrace()
    skill = load_answer_skill()
    user_content = question
    if conversation_history:
        skill = f"{skill}\n\n{HISTORY_INSTRUCTION}"
        user_content = json.dumps(
            {"question": question, "conversation_history": conversation_history}, ensure_ascii=False,
        )
    messages = [{"role": "user", "content": user_content}]
    notes = []
    tool_trace = []
    text_rankings = {}
    graph_facts = {}
    evidence = []
    graph_status = "not_requested"
    seen_calls = set()

    for round_number in range(1, MAX_TOOL_CALLS + 1):
        first = round_number == 1
        trace.step("Agent 选择首个工具" if first else "Agent 判断是否补查")
        try:
            call = choose_tool(skill, messages, first=first)
            if call is not None:
                name, args = skill_tools.tool_arguments(call)
        except (APIError, KeyError, TypeError, ValueError):
            call = None
            if not first:
                notes.append("补查判断未完成，使用已取得的证据回答。")
        if call is None:
            if not first:
                break
            name, args = GUIDE_TOOL, {"query": question}
            call = {"id": "call_" + uuid4().hex, "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}
            notes.append("首轮未得到有效工具调用，使用原问题查询攻略。")

        call_key = (name, json.dumps(args, sort_keys=True, ensure_ascii=False))
        if call_key in seen_calls:
            break
        seen_calls.add(call_key)

        trace.step(f"Agent 执行工具 · {name}")
        tool_start = perf_counter()
        results = []
        try:
            results = skill_tools.execute_tool(name, args, top_k=top_k, recall_k=recall_k, trace=trace)
        except (APIError, MilvusException, Neo4jError, ServiceUnavailable, OSError, ValueError) as exc:
            status = "error"
            action = "攻略查询" if name == GUIDE_TOOL else "克制关系查询"
            notes.append(f"{action}未完成（{type(exc).__name__}）。")
            if name == GRAPH_TOOL and graph_status == "not_requested":
                graph_status = "unavailable"
        else:
            status = "ok" if results else "empty"
            if name == GRAPH_TOOL:
                graph_facts.update({(fact["head"], fact["relation"], fact["tail"]): fact
                                    for fact in results})
                graph_status = "ok" if graph_facts else "no_match"
            elif results:
                text_rankings[f"round_{round_number}"] = results

        tool_trace.append({
            "round": round_number, "tool": name, "arguments": args, "status": status,
            "matches": len(results), "duration_ms": round((perf_counter() - tool_start) * 1000, 3),
        })
        if results:
            trace.step("Agent 整理证据")
            text_results = (rrf_fusion(text_rankings, top_k) if len(text_rankings) > 1
                            else next(iter(text_rankings.values()), []))
            evidence = build_evidence(text_results, list(graph_facts.values()))
        if first:
            observation = {"status": status, "evidence": evidence}
            messages.extend([
                {"role": "assistant", "content": None, "tool_calls": [call]},
                {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(observation, ensure_ascii=False)},
            ])

    used_tools = {entry["tool"] for entry in tool_trace if entry["status"] != "error"}
    if not used_tools:
        used_tools = {entry["tool"] for entry in tool_trace}
    used_mode = "combined" if len(used_tools) == 2 else "graph" if GRAPH_TOOL in used_tools else "hybrid"
    generation_start = perf_counter()
    answer = generate_answer(
        question, evidence, used_mode, graph_status, on_token=on_token, trace=trace,
        system_prompt=skill + "\n\n当前是最终回答环节。使用提供的证据直接回答用户。",
        conversation_history=conversation_history,
    )
    generation_seconds = round(perf_counter() - generation_start, 3)
    trace.mark("complete")
    trace.step(None)
    timings = trace.snapshot()
    timings.update(
        used_mode=used_mode, routing_source="agent", agent_tool_calls=len(tool_trace),
        generation_seconds=generation_seconds,
        total_seconds=round((perf_counter() - started), 3),
    )
    return {
        "engine": AGENT_VERSION, "used_mode": used_mode, "tool_trace": tool_trace,
        "run_settings": {"top_k": top_k, "recall_k": recall_k}, "notes": notes,
        "graph_status": graph_status, "graph_relations": len(graph_facts),
        "answer": answer, "evidence": evidence, "timings": timings,
    }
