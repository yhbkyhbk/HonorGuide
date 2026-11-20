"""复用攻略库的 Milvus 客户端和已加载的集合。"""

import atexit
from functools import lru_cache

from pymilvus import MilvusClient

from config import MILVUS_URI, MILVUS_TOKEN
from infrastructure.open_milvus_client import open_client


@lru_cache(maxsize=1)
def get_client() -> MilvusClient:
    return open_client(MILVUS_URI, MILVUS_TOKEN)


@lru_cache(maxsize=8)
def load_collection(name: str) -> None:
    get_client().load_collection(collection_name=name)


@atexit.register
def close_client() -> None:
    if get_client.cache_info().currsize:
        get_client().close()
    get_client.cache_clear()
    load_collection.cache_clear()
