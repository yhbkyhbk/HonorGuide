"""项目配置：统一读取 .env，并定义所有本地数据路径。"""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
DOCS_DIR = ROOT / "data" / "docs"
STORAGE = ROOT / "data" / "storage"
EVALUATION_DIR = ROOT / "data" / "evaluation"
QUESTIONS_FILE = EVALUATION_DIR / "questions.json"
RESULTS_FILE = EVALUATION_DIR / "results.json"

load_dotenv(ROOT / ".env", encoding="utf-8-sig")
API_KEY = os.getenv("API_KEY") or os.getenv("DASHSCOPE_API_KEY")
BASE_URL = os.getenv("BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.8-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3.7-text-embedding")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
NEO4J_DATA_DIR = STORAGE / "neo4j"
NEO4J_HOME = Path(os.getenv("NEO4J_HOME", ""))
JAVA_HOME = Path(os.getenv("JAVA_HOME", ""))

# 默认使用本地 Milvus Lite；填写 http(s) 地址可连接独立 Milvus 服务。
MILVUS_URI = os.getenv("MILVUS_URI", str(STORAGE / "milvus.db"))
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "honorguide_chunks")
