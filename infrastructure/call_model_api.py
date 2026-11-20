"""复用 HTTP 客户端，为同步模型调用提供统一配置。"""

import atexit
from threading import Lock

import numpy as np
from openai import OpenAI

from config import API_KEY, BASE_URL, EMBEDDING_MODEL, LLM_MODEL


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


def chat_completion(messages: list[dict], *, temperature: float = 0.1,
                    max_tokens: int = 2048, **options):
    return get_model_client().chat.completions.create(
        model=LLM_MODEL, messages=messages, temperature=temperature, max_tokens=max_tokens,
        extra_body={"enable_thinking": False}, **options,
    )


@atexit.register
def close_model_client() -> None:
    global _client
    with _client_lock:
        if _client is not None:
            _client.close()
            _client = None


def embed_texts(texts: list[str], *, model: str = EMBEDDING_MODEL,
                timeout: float | None = None) -> np.ndarray:
    """每批最多 20 段文本；返回归一化后的 float32 向量。"""
    if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("待编码的文本不能为空")
    vectors = []
    client = get_model_client()
    if timeout is not None:
        client = client.with_options(timeout=timeout, max_retries=0)
    for start in range(0, len(texts), 20):
        batch = texts[start:start + 20]
        response = client.embeddings.create(
            model=model, input=batch, encoding_format="float"
        )
        items = sorted(response.data, key=lambda item: item.index)
        if [item.index for item in items] != list(range(len(batch))):
            raise ValueError("接口返回的向量数量或顺序编号异常")
        vectors.extend(item.embedding for item in items)
    array = np.asarray(vectors, dtype="float32")
    if array.ndim != 2 or not np.isfinite(array).all() or np.any(np.linalg.norm(array, axis=1) == 0):
        raise ValueError("接口返回的向量包含无效数值或零向量")
    array /= np.linalg.norm(array, axis=1, keepdims=True)
    return array
