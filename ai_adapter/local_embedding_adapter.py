"""本地离线 embedding：sentence-transformers 实现。

优点：不要 API Key、不联网、聊天内容不出本机；
代价：需要 `pip install sentence-transformers`（会连带安装 torch，约 2GB 磁盘），
      并且首次运行要下载模型权重（默认模型约 100MB）。

自检（纯向量 + 接入 long_memory 的端到端验证）：
    python -m ai_adapter.local_embedding_adapter
"""
from __future__ import annotations

import os
from typing import Sequence

from .embedding_provider import EmbeddingProvider

#: 默认模型：中文、512 维、约 100MB，在效果/体积/速度之间比较均衡。
#: 想更准可以换 BAAI/bge-base-zh-v1.5（768 维）或 BAAI/bge-m3（1024 维，多语言）。
DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"

#: bge 中文 v1.5 系列的官方建议：检索词侧加上这段指令，召回效果更好
BGE_ZH_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


def default_query_instruction(model_name: str) -> str:
    """只有 bge 中文 v1.5 系列需要查询指令，其余模型留空"""
    lowered = model_name.lower()
    if "bge" in lowered and "zh" in lowered and "v1.5" in lowered:
        return BGE_ZH_QUERY_INSTRUCTION
    return ""


class LocalEmbeddingAdapter(EmbeddingProvider):
    """用 sentence-transformers 在本机跑向量模型"""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None,
                 cache_folder: str | None = None, batch_size: int = 32,
                 query_instruction: str | None = None, show_progress: bool = False,
                 local_files_only: bool | None = None, quiet: bool = False):
        # HuggingFace 在 Windows 上没法用软链接时会刷一条无害的缓存警告，这里静音掉
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # 让报错直接告诉用户怎么修
            raise ImportError(
                "没有找到 sentence-transformers，本地向量模型无法使用。\n"
                "请先安装：pip install sentence-transformers\n"
                "（只是想跑记忆模块的零依赖快速自检，可以用：python -m long_memory.memory_service）"
            ) from exc

        self.model_name = model_name
        self.name = f"local:{model_name}"
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.local_files_only = local_files_only
        # None 表示按模型自动决定；显式传 "" 可以关掉查询指令
        self.query_instruction = (default_query_instruction(model_name)
                                  if query_instruction is None else query_instruction)

        if not quiet:
            print(f"正在加载本地向量模型 {model_name}...")
        self._model = self._load_model(model_name, device, cache_folder, quiet)
        # 新版本把这个方法改名成了 get_embedding_dimension，这里兼容新旧两种写法
        get_dimension = (getattr(self._model, "get_embedding_dimension", None)
                         or getattr(self._model, "get_sentence_embedding_dimension", None))
        dimension = get_dimension() if callable(get_dimension) else None
        self.dimension = int(dimension) if dimension else None
        if not quiet:
            print(f"向量模型就绪：{model_name}（{self.dimension} 维，运行在 {self._model.device}）")

    def _load_model(self, model_name: str, device: str | None, cache_folder: str | None,
                    quiet: bool):
        """加载模型：默认先用本机缓存，没有缓存才联网下载

        不能每次都让 HuggingFace 去检查更新——国内网络下那一串 SSL 超时重试
        会让桌宠启动白白多等十几秒，而权重其实早就缓存在本机了。
        想强制走网络（比如要拉模型更新），传 local_files_only=False 即可。
        """
        from sentence_transformers import SentenceTransformer

        kwargs = {"device": device, "cache_folder": cache_folder}
        if self.local_files_only is not None:
            return SentenceTransformer(model_name, local_files_only=self.local_files_only, **kwargs)
        try:
            return SentenceTransformer(model_name, local_files_only=True, **kwargs)
        except Exception:
            if not quiet:
                print(f"本机没有 {model_name} 的缓存，正在联网下载模型权重（首次会比较慢）...")
            return SentenceTransformer(model_name, **kwargs)

    # ---------- 编码 ----------

    def _encode(self, texts: list[str]) -> list[list[float]]:
        """真正调用模型；余弦相似度由 VectorCalculation 负责，所以这里保留原始向量"""
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return [[float(value) for value in vector] for vector in vectors]

    def embed(self, text: str) -> list[float]:
        return self._encode([text])[0]

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        texts = list(texts)
        if not texts:
            return []
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        if self.query_instruction:
            text = self.query_instruction + text
        return self.embed(text)


if __name__ == "__main__":
    # 这一段只是给开发者跑端到端验证用的，正式代码不会走到这里
    import tempfile
    from pathlib import Path

    import numpy as np

    from long_memory.memory_service import EmbeddingChangedError, MemoryService
    from long_memory.vector_calculation import VectorCalculation

    embedder = LocalEmbeddingAdapter()
    print(f"\n模型={embedder.name}  维度={embedder.dimension}\n")

    def similarity(text_a: str, text_b: str) -> float:
        vec_a, vec_b = embedder.embed(text_a), embedder.embed(text_b)
        return VectorCalculation.cos_calculation(
            float(np.linalg.norm(vec_a)), float(np.linalg.norm(vec_b)), vec_a, vec_b)

    # 1) 纯向量自检：语义相近的一对，相似度必须高于不相关的一对
    related = similarity("我喜欢吃螺蛳粉", "螺蛳粉加酸笋才够味")
    unrelated = similarity("我喜欢吃螺蛳粉", "今天股市又跌了")
    print(f"语义相近 => {related:.4f}")
    print(f"毫不相关 => {unrelated:.4f}")
    assert related > unrelated, f"语义相近的句子反而更不像：{related} <= {unrelated}"

    # 2) 端到端：接进记忆服务，看语义检索能不能命中
    with tempfile.TemporaryDirectory() as tmp:
        store = Path(tmp) / "memory.json"
        service = MemoryService(embedder, store_path=store)
        service.remember("我最爱吃螺蛳粉，尤其是柳州的")
        service.remember("柳州是个好地方，螺蛳粉很出名")
        service.remember("我最近在学 Python，想写一个桌宠")

        hits = service.recall("我喜欢吃什么小吃？", top_k=2)
        print("\n回忆：", [hit["text"] for hit in hits])
        assert hits, "换了真实向量却什么都没回忆起来"
        assert "螺蛳粉" in hits[0]["text"], f"语义检索没把螺蛳粉排第一：{hits}"

        # 3) 存档里记下了向量信息，重新打开同一份存档不该报错
        reopened = MemoryService(embedder, store_path=store)
        assert reopened.graph.meta["embedding_dim"] == embedder.dimension
        print("存档元信息：", reopened.graph.meta)

        # 4) 换成另一套向量时必须明确报错，而不是把两种向量混在一起
        class _OtherEmbedder:
            name = "fake:other"
            dimension = 999

            def embed(self, text: str) -> list[float]:
                return [0.0] * 999

        try:
            MemoryService(_OtherEmbedder(), store_path=store)
        except EmbeddingChangedError as exc:
            print("\n换模型时按预期拦下：", str(exc).splitlines()[0])
        else:
            raise AssertionError("换了维度不同的向量模型，却没能拦下来")

        # 5) 显式允许重建时，应该清空旧记忆继续跑
        reset_service = MemoryService(_OtherEmbedder(), store_path=store,
                                      on_embedding_change="reset")
        assert not reset_service.graph.nodes, "reset 之后旧记忆应该被清空"

    print("\n自检通过：本地向量模型可用，语义检索与存档兼容性都正常喵～")
