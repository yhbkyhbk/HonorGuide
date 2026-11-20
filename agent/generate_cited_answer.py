"""回答生成模块：模型调用、证据组织、提示词与引用编号检查。"""

import json
import re

from infrastructure.call_model_api import chat_completion
from rag.tokenize_search_text import strip_reference_lines

CITATION = re.compile(r"\[([TG]\d+)\]")


def build_evidence(text_results: list[dict], graph_facts: list[dict]) -> list[dict]:
    """给两类证据分配独立引用 ID；去重后在内存中组织，不保存中间文件。"""
    evidence = []
    seen = set()
    for result in text_results:
        metadata = result["metadata"]
        key = ("text", metadata["chunk_id"])
        if key in seen:
            continue
        seen.add(key)
        text = strip_reference_lines(result["page_content"]).strip()
        evidence.append({
            "id": f"T{len(evidence) + 1}",
            "type": "text", "text": text,
            "source_doc_id": metadata["doc_id"], "source_file": metadata["source"],
            "chunk_id": metadata["chunk_id"], "source_urls": metadata.get("source_urls", []),
        })
    graph_count = 0
    for fact in graph_facts:
        sources = zip(
            fact["source_doc_ids"], fact["evidence_quotes"], fact["reasons"],
            fact["conditions"], fact["source_files"], strict=True,
        )
        for doc_id, quote, reason, condition, source_file in sources:
            key = ("graph", fact["head"], fact["tail"], doc_id, quote)
            if key in seen:
                continue
            seen.add(key)
            graph_count += 1
            evidence.append({
                "id": f"G{graph_count}", "type": "graph",
                "head": fact["head"], "relation": fact["relation"], "tail": fact["tail"],
                "reason": reason, "condition": condition, "evidence_quote": quote,
                "source_doc_id": doc_id, "source_file": source_file,
                "source_urls": fact["source_urls"],
            })
    return evidence


def generate_answer(
    question: str, evidence: list[dict], mode: str, graph_status: str,
    *, system_prompt: str, on_token=None, trace=None,
    conversation_history: list[dict[str, str]] | None = None,
) -> str:
    """流式下发回答并校验引用编号。"""
    def emit(text):
        if trace:
            trace.mark("first_text")
        if on_token:
            on_token(text)
            if trace:
                trace.mark("server_first_dispatch")

    if not evidence:
        if trace:
            trace.step("本地无证据提示下发（不调用模型）")
        answer = "当前检索未找到支持该问题的证据。"
        if graph_status == "no_match":
            answer += "图谱暂未收录该关系。"
        emit(answer)
    else:
        if trace:
            trace.step("提示词组装")
        payload = {
            "question": question, "mode": mode, "graph_status": graph_status, "evidence": evidence,
        }
        if conversation_history:
            payload["conversation_history"] = conversation_history
            allowed_ids = "、".join(f"[{item['id']}]" for item in evidence)
            system_prompt += (
                f"\n本轮可引用的编号只有 {allowed_ids}。"
                "用旧对话理解用户的场景和偏好，引用只能指向支持当前陈述的本轮证据。"
            )
        user_content = json.dumps(payload, ensure_ascii=False)
        parts = []
        first_sent = False
        if trace:
            trace.step("模型请求至首段文本（含网络与重试）")
            trace.mark("model_request")
        with chat_completion(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}],
            stream=True,
        ) as stream:
            for chunk in stream:
                if chunk.choices and chunk.choices[0].finish_reason == "length":
                    raise ValueError("回答达到输出长度限制，内容尚未完整生成，请缩小问题范围后重试")
                # 角色、用量及推理字段不算回答首 token；纯空白不触发首次显示。
                text = chunk.choices[0].delta.content if chunk.choices else None
                if not text:
                    continue
                parts.append(text)
                if not first_sent:
                    if not "".join(parts).strip():
                        continue
                    if trace:
                        trace.mark("model_first_text")
                        trace.step("首段文本下发")
                    emit("".join(parts))
                    first_sent = True
                    if trace:
                        trace.step("后续文本生成与下发")
                else:
                    emit(text)
        answer = "".join(parts).strip()
        if not answer:
            raise ValueError("模型返回了空内容")
    if trace:
        trace.step("回答引用校验")
    allowed = {item["id"] for item in evidence}
    if set(CITATION.findall(answer)) - allowed:
        raise ValueError("模型引用了本次未提供的证据编号")
    return answer
