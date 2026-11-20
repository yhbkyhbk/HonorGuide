"""用 RRF 合并多路检索的排名。"""

from typing import Callable


def _chunk_id(result: dict) -> str:
    return result["metadata"]["chunk_id"]


def rrf_fusion(
    rankings: dict[str, list[dict]], top_k: int = 3, rank_constant: int = 60,
    *, key: Callable[[dict], str | int] = _chunk_id,
) -> list[dict]:
    """每路从第 1 名起计算 1/(60+rank)，按结果标识合并并累加贡献。"""
    if top_k <= 0 or rank_constant <= 0:
        raise ValueError("top_k 和 rank_constant 必须大于 0")
    fused = {}
    for method, results in rankings.items():
        seen = set()
        for rank, result in enumerate(results, 1):
            item_id = key(result)
            if item_id in seen:
                continue
            seen.add(item_id)
            if item_id not in fused:
                fused[item_id] = {**result, "score": 0.0, "ranks": {}, "raw_scores": {}}
            item = fused[item_id]
            item["score"] += 1 / (rank_constant + rank)
            item["ranks"][method] = rank
            item["raw_scores"][method] = result["score"]
    return sorted(
        fused.values(), key=lambda item: (-item["score"], key(item))
    )[:top_k]
