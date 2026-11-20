"""项目配置：统一读取 .env，并定义所有本地数据路径。"""

import os
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent
DOCS_DIR = ROOT / "data" / "docs"
STORAGE = ROOT / "data" / "storage"
EVALUATION_DIR = ROOT / "data" / "evaluation"
QUESTIONS_FILE = EVALUATION_DIR / "questions.json"
RESULTS_FILE = EVALUATION_DIR / "results.json"
TIMINGS_FILE = EVALUATION_DIR / "request_timings.jsonl"

_settings = {
    **{key: value for key, value in dotenv_values(ROOT / ".env", encoding="utf-8-sig").items()
       if value is not None},
    **os.environ,
}
API_KEY = _settings.get("API_KEY") or _settings.get("DASHSCOPE_API_KEY")
BASE_URL = _settings.get("BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
LLM_MODEL = _settings.get("LLM_MODEL", "qwen3.8-flash")
EMBEDDING_MODEL = _settings.get("EMBEDDING_MODEL", "qwen3.7-text-embedding")
MEMORY_EMBEDDING_MODEL = _settings.get("MEMORY_EMBEDDING_MODEL", "qwen3.7-text-embedding")

NEO4J_URI = _settings.get("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USERNAME = _settings.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = _settings.get("NEO4J_PASSWORD", "")
NEO4J_DATABASE = _settings.get("NEO4J_DATABASE", "neo4j")
NEO4J_DATA_DIR = STORAGE / "neo4j"
NEO4J_HOME = Path(_settings.get("NEO4J_HOME", ""))
JAVA_HOME = Path(_settings.get("JAVA_HOME", ""))

# 使用项目专用名称，避免 PyMilvus 把本地路径当作 SDK 的远程地址。
MILVUS_URI = _settings.get("HONORGUIDE_MILVUS_URI") or _settings.get("MILVUS_URI") or str(STORAGE / "milvus.db")
MILVUS_TOKEN = _settings.get("MILVUS_TOKEN", "")
MILVUS_COLLECTION = _settings.get("MILVUS_COLLECTION", "honorguide_chunks")
