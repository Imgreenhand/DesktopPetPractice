"""长期记忆模块：记忆图 + 激活扩散 + 记忆服务门面。"""

from .memory_graph import MemNode, MemoryGraph

__all__ = ["MemNode", "MemoryGraph", "Embedder", "EmbeddingChangedError",
           "MemoryService", "get_primary_nickname"]

# 服务层按需懒加载：既能 from long_memory import MemoryService，
# 又不会因为包初始化时提前导入，干扰 python -m long_memory.memory_service 自检
_LAZY_EXPORTS = {
    "Embedder": "memory_service",
    "EmbeddingChangedError": "memory_service",
    "MemoryService": "memory_service",
    "get_primary_nickname": "memory_service",
}


def __getattr__(name: str):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(f".{module_name}", __name__), name)
