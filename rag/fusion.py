"""混合检索：两路召回后使用 RRF 合并排名。"""

from .bm25.retriever import search as bm25_search
from .vector.retriever import search as vector_search


def rrf_fusion(
    rankings: dict[str, list[dict]], top_k: int = 3, rank_constant: int = 60
) -> list[dict]:
    """每路从第 1 名起计算 1/(60+rank)，按 chunk_id 合并并累加贡献。"""
    if top_k <= 0 or rank_constant <= 0:
        raise ValueError("top_k 和 rank_constant 必须大于 0")
    fused = {}
    for method, results in rankings.items():
        seen = set()
        for rank, result in enumerate(results, 1):
            chunk_id = result["metadata"]["chunk_id"]
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            if chunk_id not in fused:
                fused[chunk_id] = {**result, "score": 0.0, "ranks": {}, "raw_scores": {}}
            item = fused[chunk_id]
            item["score"] += 1 / (rank_constant + rank)
            item["ranks"][method] = rank
            item["raw_scores"][method] = result["score"]
    return sorted(
        fused.values(), key=lambda item: (-item["score"], item["metadata"]["chunk_id"])
    )[:top_k]


def hybrid_search(question: str, top_k: int = 3, recall_k: int = 10, *, trace=None) -> list[dict]:
    """两路分别召回候选，再做 RRF；只复用已有向量索引，不自动重建。"""
    if not question.strip() or min(top_k, recall_k) <= 0:
        raise ValueError("问题不能为空，top_k 和 recall_k 必须大于 0")
    recall_k = max(top_k, recall_k)
    rankings = {
        "bm25": bm25_search(question, recall_k, trace=trace),
        "vector": vector_search(question, recall_k, trace=trace),
    }
    if trace:
        trace.step("RRF 排名融合")
    return rrf_fusion(rankings, top_k)
