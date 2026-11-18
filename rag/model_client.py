"""为同步模型请求复用 HTTP 客户端。"""

import atexit
from threading import Lock

from openai import OpenAI

from config import API_KEY, BASE_URL


_client: OpenAI | None = None
_client_lock = Lock()


def get_model_client() -> OpenAI:
    """首次使用时创建，后续请求共享连接池。"""
    global _client
    with _client_lock:
        if _client is None:
            if not API_KEY:
                raise ValueError("请先在 .env 配置 API_KEY 或 DASHSCOPE_API_KEY")
            _client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=60, max_retries=1)
        return _client


@atexit.register
def close_model_client() -> None:
    global _client
    with _client_lock:
        if _client is not None:
            _client.close()
            _client = None
