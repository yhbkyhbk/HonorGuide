"""记录问答及并行检索的分段耗时，并写入测评目录。"""

import json
from datetime import datetime, timezone
from threading import Lock
from time import perf_counter
from uuid import uuid4

from config import TIMINGS_FILE


_timing_log_lock = Lock()


class StageTrace:
    """记录一条执行路径内的顺序阶段。"""

    def __init__(self, started: float):
        self.started = started
        self.stages = []
        self.current = None

    def elapsed(self):
        return (perf_counter() - self.started) * 1000

    def step(self, name):
        now = self.elapsed()
        if self.current:
            label, start = self.current
            self.stages.append({"stage": label, "start_ms": start, "end_ms": now})
        self.current = (name, now) if name else None

    def stage_rows(self) -> list[dict]:
        stages = list(self.stages)
        if self.current:
            label, start = self.current
            stages.append({"stage": label, "start_ms": start, "end_ms": self.elapsed()})
        return [{**row, "duration_ms": row["end_ms"] - row["start_ms"]} for row in stages]


class RequestTrace(StageTrace):
    """主路径计总耗时，并行分支各自记录阶段明细。"""

    def __init__(self):
        self.request_id = uuid4().hex
        self.created_at = datetime.now(timezone.utc).isoformat()
        super().__init__(perf_counter())
        self.marks = {}
        self.branches = {}
        self.current = ("页面准备", 0.0)

    def mark(self, name):
        self.marks.setdefault(name, self.elapsed())

    def branch(self, name: str) -> StageTrace:
        branch = StageTrace(self.started)
        key, number = name, 1
        while key in self.branches:
            number += 1
            key = f"{name}（第 {number} 次）"
        self.branches[key] = branch
        return branch

    def snapshot(self):
        stages = self.stage_rows()
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
            "stages": stages,
            "parallel_branches": {name: branch.stage_rows() for name, branch in self.branches.items()},
        }


def save_timing(timings: dict, event: str = "server_complete", status: str = "ok") -> None:
    """每次请求写一次服务端记录；浏览器回报后追加同一 request_id 的渲染记录。"""
    TIMINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({**timings, "event": event, "status": status}, ensure_ascii=False) + "\n"
    with _timing_log_lock, TIMINGS_FILE.open("a", encoding="utf-8") as file:
        file.write(line)
