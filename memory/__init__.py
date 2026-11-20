"""对外提供本地问答记忆的读取、保存和上下文选择。"""

from .search_conversation_memory import MEMORY_VECTOR_ERRORS, index_saved_turn, select_context
from .store_conversation_history import load_history, save_turn as _save_turn

__all__ = ["load_history", "save_turn", "select_context"]


def save_turn(request_id: str, entry: dict, *, index_vector: bool = False) -> bool:
    """先保存对话，有完整回答时按需更新向量。返回向量写入是否成功。"""
    turn_id = _save_turn(request_id, entry)
    if index_vector and "result" in entry:
        try:
            index_saved_turn(turn_id, entry)
        except MEMORY_VECTOR_ERRORS:
            return False
    return True
