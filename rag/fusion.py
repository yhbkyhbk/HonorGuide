"""混合检索：并行召回后使用 RRF 合并排名。"""

from concurrent.futures import ThreadPoolExecutor, wait

from .bm25.retriever import search as bm25_search
from .vector.retriever import search as vector_search


_retrieval_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="honorguide-retrieval")


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


def _search_branch(search, question: str, recall_k: int, trace):
    try:
        return search(question, recall_k, trace=trace)
    finally:
        if trace:
            trace.step(None)


def hybrid_search(question: str, top_k: int = 3, recall_k: int = 10, *, trace=None) -> list[dict]:
    """并行查 BM25 和向量索引，收齐两路候选后做 RRF。"""
    if not question.strip() or min(top_k, recall_k) <= 0:
        raise ValueError("问题不能为空，top_k 和 recall_k 必须大于 0")
    recall_k = max(top_k, recall_k)
    if trace:
        trace.step("BM25 与向量并行召回")
    futures = {
        "bm25": _retrieval_pool.submit(
            _search_branch, bm25_search, question, recall_k, trace.branch("BM25") if trace else None,
        ),
        "vector": _retrieval_pool.submit(
            _search_branch, vector_search, question, recall_k, trace.branch("向量检索") if trace else None,
        ),
    }
    # 即使某一路失败，也让两路结束后再归还知识库读取权限。
    wait(futures.values())
    rankings = {name: future.result() for name, future in futures.items()}
    if trace:
        trace.step("RRF 排名融合")
    return rrf_fusion(rankings, top_k)
