"""回答 Skill 提供给 Agent 的两种只读检索工具。"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag.run_hybrid_text_search import hybrid_search
from rag.search_neo4j_graph import retrieve as search_matchups


GUIDE_TOOL = "search_guides"
GRAPH_TOOL = "search_matchups"


class GuideQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    query: str = Field(min_length=1, max_length=2000)


class GraphQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    kind: Literal["incoming", "outgoing", "pair"]
    hero: str = Field(min_length=1, max_length=50)
    target: str | None = Field(default=None, min_length=1, max_length=50)

    @model_validator(mode="after")
    def check_target(self):
        if self.kind == "pair" and (not self.target or self.target == self.hero):
            raise ValueError("配对查询需要两个不同英雄")
        if self.kind != "pair" and self.target is not None:
            raise ValueError("方向查询只需要一个英雄")
        return self


TOOL_INPUTS = {GUIDE_TOOL: GuideQuery, GRAPH_TOOL: GraphQuery}
TOOL_DESCRIPTIONS = {
    GUIDE_TOOL: "检索攻略文本，查技能、装备、铭文、连招和打法。",
    GRAPH_TOOL: "查 Neo4j 的英雄克制关系。pair 查双方，incoming 查谁克制 hero，outgoing 查 hero 克制谁。",
}
TOOLS = [
    {"type": "function", "function": {
        "name": name, "description": TOOL_DESCRIPTIONS[name], "parameters": schema.model_json_schema(),
    }}
    for name, schema in TOOL_INPUTS.items()
]


def tool_arguments(call: dict) -> tuple[str, dict]:
    if call.get("type") != "function" or not isinstance(call.get("id"), str) or not call["id"]:
        raise ValueError("工具调用格式无效")
    function = call["function"]
    name = function["name"]
    if name not in TOOL_INPUTS:
        raise ValueError("未注册的工具")
    args = TOOL_INPUTS[name].model_validate_json(function["arguments"]).model_dump(exclude_none=True)
    return name, args


def execute_tool(name: str, args: dict, *, top_k: int, recall_k: int, trace=None) -> list[dict]:
    if name == GUIDE_TOOL:
        return hybrid_search(args["query"], top_k, recall_k, trace=trace)
    if name == GRAPH_TOOL:
        return search_matchups(args, trace=trace)
    raise ValueError("未注册的工具")
