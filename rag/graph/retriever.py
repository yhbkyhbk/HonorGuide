"""GraphRAG 模块：连接、启动、抽取、校验、入库、关系检索。"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import time
from urllib.parse import urlsplit

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

from config import (
    NEO4J_URI as URI, NEO4J_USERNAME as USERNAME, NEO4J_PASSWORD as PASSWORD,
    NEO4J_DATABASE as DATABASE, NEO4J_HOME, JAVA_HOME, NEO4J_DATA_DIR,
)
from ..documents import load_parent_records as load_documents
from ..generation import call_llm

EXTRACT_PROMPT = """只依据输入攻略，抽取英雄之间有证据的机制克制关系。
COUNTERS 表示技能或伤害机制上的相对优势，不表示任何条件下都能获胜。
例如正文明确说明乙的物理免疫不能处理甲的法术输出，可提取甲 COUNTERS 乙，
但必须保留对局条件与不保证胜负的限制。不要仅凭单个技能描述或模型记忆推断克制。
只允许使用给定的英雄名单。输出 JSON 对象，格式：
{"relations":[{"head":"优势方英雄","relation":"COUNTERS","tail":"被克制方英雄",
"reason":"依据资料解释原因","condition":"适用条件及局限",
"evidence_quote":"连续复制一段原文，包含双方英雄名，不拼接、不改写"}]}
没有证据时返回 {"relations":[]}。"""


def connect():
    """返回驱动，连接信息来自 .env；密码不会打印或写入关系数据。"""
    if not PASSWORD:
        raise ValueError("请在 .env 配置 NEO4J_PASSWORD")
    return GraphDatabase.driver(
        URI, auth=(USERNAME, PASSWORD), connection_timeout=5,
        connection_acquisition_timeout=5, max_transaction_retry_time=5,
    )


def prepare_local_config() -> Path:
    """为本项目生成 Neo4j 配置，安装程序与数据库数据目录分开管理。"""
    directory = NEO4J_DATA_DIR.resolve()
    conf = directory / "conf"
    for folder in (conf, directory / "logs", directory / "run", directory / "transactions"):
        folder.mkdir(parents=True, exist_ok=True)
    template = NEO4J_HOME / "conf"
    overrides = {
        "server.directories.data": directory.as_posix(),
        "server.directories.transaction.logs.root": (directory / "transactions").as_posix(),
        "server.directories.logs": (directory / "logs").as_posix(),
        "server.directories.run": (directory / "run").as_posix(),
    }
    lines = (template / "neo4j.conf").read_text(encoding="utf-8-sig").splitlines()
    lines = [line for line in lines if line.split("=", 1)[0].strip() not in overrides]
    lines.extend(f"{key}={value}" for key, value in overrides.items())
    (conf / "neo4j.conf").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for name in ("server-logs.xml", "user-logs.xml", "neo4j-admin.conf"):
        source = template / name
        if source.is_file():
            (conf / name).write_bytes(source.read_bytes())
    return conf


def start_database() -> None:
    """启动已安装的本地便携实例；不修改系统服务或全局环境变量。"""
    try:
        with connect() as driver:
            driver.verify_connectivity()
            if urlsplit(URI).hostname in {"localhost", "127.0.0.1", "::1"}:
                records, _, _ = driver.execute_query(
                    "SHOW SETTINGS YIELD name, value WHERE name = 'server.directories.data' RETURN value",
                    database_="system",
                )
                if not records or Path(records[0]["value"]).resolve() != NEO4J_DATA_DIR.resolve():
                    raise ValueError("当前 Neo4j 实例未使用本项目数据目录，请停止旧实例后再从页面启动")
        print("Neo4j 已在运行")
        return
    except ServiceUnavailable:
        pass
    home = NEO4J_HOME
    java_home = JAVA_HOME
    java = java_home / "bin" / ("java.exe" if os.name == "nt" else "java")
    if not java.is_file() or not (home / "lib").is_dir():
        raise ValueError("请配置 .env 中的 JAVA_HOME 和 NEO4J_HOME，或手动启动 Neo4j")
    conf = prepare_local_config()
    environment = {**os.environ, "NEO4J_HOME": str(home), "JAVA_HOME": str(java_home), "NEO4J_CONF": str(conf)}
    with (NEO4J_DATA_DIR / "logs" / "console.log").open("ab") as log:
        process = subprocess.Popen(
            [str(java), "-cp", str(home / "lib" / "*"), f"-Dbasedir={home}",
             "org.neo4j.server.startup.Neo4jCommand", "console"],
            cwd=home, env=environment, stdout=log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise ValueError(f"Neo4j 启动失败，请查看 {NEO4J_DATA_DIR / 'logs' / 'console.log'}")
        try:
            with connect() as driver:
                driver.verify_connectivity()
            print("Neo4j 已启动")
            return
        except ServiceUnavailable:
            time.sleep(1)
    raise ValueError(f"Neo4j 启动超时，请查看 {NEO4J_DATA_DIR / 'logs' / 'console.log'}")


def validate_relations(rows: list[dict], documents: dict[str, dict]) -> list[dict]:
    """校验关系类型、英雄、来源与原文；同一英雄对在每篇来源中保留一条证据。"""
    if not isinstance(rows, list):
        raise ValueError("关系数据必须是列表")
    heroes = {doc["metadata"]["hero"] for doc in documents.values()}
    validated = {}
    required = ("head", "tail", "relation", "reason", "condition", "evidence_quote", "source_doc_id")
    for row in rows:
        if not isinstance(row, dict) or any(
            not isinstance(row.get(key), str) or not row[key].strip() for key in required
        ):
            raise ValueError("关系字段缺失或为空")
        row = {**row, **{key: row[key].strip() for key in required}}
        if row["relation"] != "COUNTERS":
            raise ValueError("关系类型只能是 COUNTERS")
        if row["head"] not in heroes or row["tail"] not in heroes or row["head"] == row["tail"]:
            raise ValueError("英雄不在资料名单中，或关系指向自身")
        doc = documents.get(row["source_doc_id"])
        if doc is None or row["evidence_quote"] not in doc["page_content"]:
            raise ValueError("关系证据无法在对应原文中找到")
        if not all(hero in row["evidence_quote"] for hero in (row["head"], row["tail"])):
            raise ValueError("证据必须明确包含关系两端的英雄名")
        allowed_urls = re.findall(r"\[[^\]]+\]\((https?://[^)\s]+)\)", doc["page_content"])
        source_urls = list(dict.fromkeys(row.get("source_urls", allowed_urls)))
        if any(url not in allowed_urls for url in source_urls):
            raise ValueError("来源链接不在对应文档中")
        row = {key: row[key] for key in required}
        row.update(source_file=doc["metadata"]["source"], source_urls=source_urls)
        validated[(row["head"], row["tail"], row["source_doc_id"])] = row
    return list(validated.values())


def extract_relations() -> list[dict]:
    """抽取完整克制/对抗小节，校验后返回内存数据，不保存中间文件。"""
    documents = load_documents()
    heroes = sorted({doc["metadata"]["hero"] for doc in documents.values()})
    rows = []
    for doc_id, doc in documents.items():
        sections = re.findall(
            r"^## ([^\n]+)\n(.*?)(?=^## |\Z)", doc["page_content"], flags=re.M | re.S
        )
        text = "\n\n".join(
            title + "\n" + body for title, body in sections if re.search("克制|对抗", title)
        )
        if not text.strip():
            continue
        result = json.loads(call_llm(
            EXTRACT_PROMPT,
            f"英雄名单：{json.dumps(heroes, ensure_ascii=False)}\n资料：\n{text}",
            json_mode=True,
        ))
        extracted = result.get("relations")
        if not isinstance(extracted, list):
            raise ValueError(f"{doc_id} 的模型输出缺少 relations 列表")
        urls = list(dict.fromkeys(re.findall(r"\[[^\]]+\]\((https?://[^)\s]+)\)", text)))
        for row in extracted:
            rows.append({**row, "source_doc_id": doc_id, "source_urls": urls})
        print(f"{doc_id}：抽取 {len(extracted)} 条待校验关系", flush=True)
    relations = validate_relations(rows, documents)
    if not relations:
        raise ValueError("没有抽取到有效关系，请检查资料和模型输出")
    print(f"已校验 {len(relations)} 条来源证据")
    return relations


def load_graph_relations(driver, documents: dict[str, dict]) -> list[dict]:
    """读取已有来源证据，仅保留仍能通过当前原文校验的记录。"""
    records, _, _ = driver.execute_query(
        "MATCH (a:Hero)-[r:COUNTERS]->(b:Hero) "
        "RETURN a.name AS head, b.name AS tail, properties(r) AS data",
        database_=DATABASE,
    )
    rows = []
    for record in records:
        data = record["data"]
        ids = data.get("source_doc_ids", [data.get("source_doc_id")])
        quotes = data.get("evidence_quotes", [data.get("evidence_quote")])
        reasons = data.get("reasons", [data.get("reason", "")] * len(ids))
        conditions = data.get("conditions", [data.get("condition", "")] * len(ids))
        for doc_id, quote, reason, condition in zip(ids, quotes, reasons, conditions):
            doc = documents.get(doc_id)
            if doc is None:
                continue
            urls = re.findall(r"\[[^\]]+\]\((https?://[^)\s]+)\)", doc["page_content"])
            row = {
                "head": record["head"], "tail": record["tail"], "relation": "COUNTERS",
                "source_doc_id": doc_id, "evidence_quote": quote,
                "reason": reason, "condition": condition,
                "source_urls": [url for url in data.get("source_urls", []) if url in urls],
            }
            try:
                rows.extend(validate_relations([row], documents))
            except ValueError:
                continue
    return rows


def build_graph(rows: list[dict] | None = None) -> dict:
    """同一方向保留一条 COUNTERS 边；新证据合并已有有效来源，不因漏抽而丢失。"""
    documents = load_documents()
    rows = validate_relations(extract_relations() if rows is None else rows, documents)
    if not rows:
        raise ValueError("关系列表为空，请先抽取有效关系")
    heroes = sorted({doc["metadata"]["hero"] for doc in documents.values()})
    with connect() as driver:
        driver.verify_connectivity()
        rows = validate_relations(load_graph_relations(driver, documents) + rows, documents)
        grouped = {}
        for row in rows:
            grouped.setdefault((row["head"], row["tail"]), []).append(row)
        relationships = []
        for (head, tail), evidence in grouped.items():
            relationships.append({
                "head": head, "tail": tail,
                # 五个列表使用相同下标，将原因、条件和原文关联到各自来源。
                "reasons": [item["reason"] for item in evidence],
                "conditions": [item["condition"] for item in evidence],
                "source_doc_ids": [item["source_doc_id"] for item in evidence],
                "evidence_quotes": [item["evidence_quote"] for item in evidence],
                "source_files": [item["source_file"] for item in evidence],
                "source_urls": list(dict.fromkeys(url for item in evidence for url in item["source_urls"])),
            })
        driver.execute_query(
            "CREATE CONSTRAINT hero_name_unique IF NOT EXISTS "
            "FOR (h:Hero) REQUIRE h.name IS UNIQUE", database_=DATABASE,
        )
        driver.execute_query(
            "UNWIND $heroes AS name MERGE (:Hero {name: name})",
            heroes=heroes, database_=DATABASE,
        )
        driver.execute_query(
            """UNWIND $relationships AS row
            MATCH (a:Hero {name: row.head}), (b:Hero {name: row.tail})
            OPTIONAL MATCH (a)-[old:COUNTERS]->(b)
            WITH a, b, row, collect(old) AS existing
            FOREACH (duplicate IN tail(existing) | DELETE duplicate)
            MERGE (a)-[r:COUNTERS]->(b)
            SET r.reasons = row.reasons, r.conditions = row.conditions,
                r.evidence_quotes = row.evidence_quotes,
                r.source_doc_ids = row.source_doc_ids, r.source_files = row.source_files,
                r.source_urls = row.source_urls
            REMOVE r.source_doc_id, r.evidence_quote, r.source_file, r.reason, r.condition
            RETURN count(r) AS processed""",
            relationships=relationships, database_=DATABASE,
        )
    return {"heroes": len(heroes), "relationships": len(relationships), "source_evidences": len(rows)}


def parse_question(question: str, heroes: list[str]) -> dict:
    """识别精确英雄名与三类问题。两位英雄的问题查询双方，避免预设克制方向。"""
    mentioned = sorted({name for name in heroes if name in question}, key=question.find)
    if len(mentioned) == 2:
        return {"kind": "pair", "hero": mentioned[0], "target": mentioned[1]}
    if len(mentioned) != 1:
        raise ValueError("请明确输入一至两个已入库的英雄名")
    hero = mentioned[0]
    verb = question.find("克制")
    if verb < 0 and "克星" not in question:
        raise ValueError("目前支持：谁克制某英雄、某英雄克制谁、两位英雄之间的克制关系")
    incoming = (
        "克星" in question or verb < question.find(hero)
        or "被" in question[question.find(hero) + len(hero):]
    )
    return {"kind": "incoming" if incoming else "outgoing", "hero": hero}


def retrieve(question: str, *, trace=None) -> tuple[dict, list[dict]]:
    """使用固定参数化 Cypher 查询一跳关系，返回实际图谱证据。"""
    if trace:
        trace.step("Neo4j 连接与认证")
    with connect() as driver:
        driver.verify_connectivity()
        if trace:
            trace.step("图谱英雄名单读取")
        records, _, _ = driver.execute_query(
            "MATCH (h:Hero) RETURN h.name AS name", database_=DATABASE
        )
        if trace:
            trace.step("图问题解析与 Cypher 组装")
        plan = parse_question(question, [record["name"] for record in records])
        conditions = {
            "incoming": "b.name = $hero",
            "outgoing": "a.name = $hero",
            "pair": "((a.name = $hero AND b.name = $target) OR "
                    "(a.name = $target AND b.name = $hero))",
        }
        cypher = (
            "MATCH (a:Hero)-[r:COUNTERS]->(b:Hero) WHERE " + conditions[plan["kind"]]
            + " RETURN a.name AS head, b.name AS tail, type(r) AS relation, "
            "r.reasons AS reasons, r.conditions AS conditions, "
            "r.evidence_quotes AS evidence_quotes, r.source_doc_ids AS source_doc_ids, "
            "r.source_files AS source_files, r.source_urls AS source_urls "
            "ORDER BY head, tail LIMIT 20"
        )
        if trace:
            trace.step("图关系查询与结果整理")
        records, _, _ = driver.execute_query(
            cypher, parameters_=plan, database_=DATABASE
        )
    return {**plan, "cypher": cypher}, [record.data() for record in records]
