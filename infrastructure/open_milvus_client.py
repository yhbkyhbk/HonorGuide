"""打开本地 Milvus Lite 或远程 Milvus 的连接。"""

from pathlib import Path

from pymilvus import MilvusClient

from config import ROOT


def open_client(uri: str, token: str = "") -> MilvusClient:
    """本地 .db 路径使用 Milvus Lite，http(s) 地址使用 Milvus 服务。"""
    if not uri.startswith(("http://", "https://")):
        path = Path(uri)
        path = path if path.is_absolute() else ROOT / path
        if path.suffix != ".db":
            raise ValueError("本地 Milvus 路径应以 .db 结尾，服务端连接应使用 http(s) 地址")
        path.parent.mkdir(parents=True, exist_ok=True)
        # Lite 与应用处于同一进程，避免空闲时的 gRPC 心跳触发 too_many_pings。
        return MilvusClient(
            uri=str(path.resolve()), token=token, timeout=10,
            grpc_options={"grpc.keepalive_permit_without_calls": False},
        )
    return MilvusClient(uri=uri, token=token, timeout=10)

