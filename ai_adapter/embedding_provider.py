"""Embedding（文本向量）接入层：抽象基类 + 按名字创建实现的工厂。

和 llm_provider.py 的分工完全一致：这里只定义「接口」，具体模型放进各自的适配器文件。
上层的 long_memory 只认识 EmbeddingProvider，所以换向量模型不需要动任何记忆逻辑。

目前提供：
    local  ->  LocalEmbeddingAdapter（sentence-transformers，本机离线运行）

以后要接在线接口（OpenAI / 硅基流动 / 智谱 / Ollama……），只要新增一个继承本基类的
适配器文件，再在下面的 build_embedder 里加一条分支即可。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence


class EmbeddingProvider(ABC):
    """文本向量化接口。

    必须实现的只有 embed；name 与 dimension 用来做「存档兼容性」校验，
    embed_batch 与 embed_query 都有默认实现，支持批量的后端覆写它能明显提速。
    """

    #: 模型标识，会写进记忆存档，用来发现「换了模型」这件事
    name: str = "unknown"

    #: 向量维度；未知时为 None，此时跳过维度校验
    dimension: int | None = None

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """把一段文本编码成向量（记忆「文档」侧）"""

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """批量编码；默认逐条调用 embed，后端若支持批量请覆写"""
        return [self.embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """把「检索词」编码成向量。

        部分模型（如 bge 中文 v1.5 系列）要求查询侧加指令前缀才能发挥最佳效果，
        这类后端覆写本方法；其余模型直接复用 embed 即可。
        """
        return self.embed(text)


def build_embedder(kind: str = "local", **kwargs) -> EmbeddingProvider:
    """按名字创建 embedder。

    具体实现延迟导入，这样没装 sentence-transformers 也不会影响别的后端。
    """
    if kind == "local":
        from .local_embedding_adapter import LocalEmbeddingAdapter

        return LocalEmbeddingAdapter(**kwargs)
    # 方案 3 预留位：以后在这里补 "openai" / "siliconflow" / "ollama" 等分支
    raise ValueError(f"未知的 embedding 后端：{kind!r}（目前可用：local）")
