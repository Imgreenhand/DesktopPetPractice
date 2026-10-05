"""记忆服务门面：把“文本 → 向量 → 检索 → 扩散 → 加权 → 注入提示词”串起来

自检：python -m long_memory.memory_service（零依赖，用假向量器跑通流程）
真实向量：见 ai_adapter/local_embedding_adapter.py，
         python -m ai_adapter.local_embedding_adapter 可跑端到端自检

本模块不绑定任何向量供应商，只认下面的 Embedder 协议，真实实现由调用方注入。
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol, Sequence

from memory_graph import MemoryGraph


class EmbeddingChangedError(ValueError):
    """存档里的向量和当前 embedder 不是同一套（换了模型或维度）时抛出"""


class Embedder(Protocol):
    """向量化接口（真实实现见 ai_adapter/ 下的适配器）

    只有 embed 是必须的，下面几个是「可选增强」，实现了就会自动被用上：
        name / dimension  用于存档兼容性校验，及时发现换了模型
        embed_batch       批量编码，本地模型批量比逐条快得多
        embed_query       检索词单独编码，部分模型需要加查询指令前缀
    """

    def embed(self, text: str) -> list[float]:
        ...


def get_primary_nickname(props: dict) -> str | None:
    """取出权重最高的那个外号"""
    nicknames = props.get("nicknames", {})
    if not nicknames:
        return None
    return max(nicknames, key=nicknames.get)


class MemoryService:
    """桌宠的长期记忆服务"""

    #: on_embedding_change 的合法取值
    EMBEDDING_CHANGE_POLICIES = ("raise", "reset", "ignore")

    #: 默认起点阈值：真实向量模型（bge 系列）实测的相似度分布是
    #: 「相关」约 0.35~0.55、「无关」约 0.15~0.30，所以取 0.30——
    #: 既能滤掉噪声，又不会把「换种说法的同一件事」挡在门外。
    #: 换了向量模型后，建议用自己的记忆数据重新标定这两个阈值。
    DEFAULT_START_THRESHOLD = 0.30

    #: 默认连边阈值：记忆之间同话题实测约 0.6 以上，无关的最高能到 0.4 左右
    DEFAULT_LINK_THRESHOLD = 0.50

    def __init__(self, embedder: Embedder, store_path: str | Path | None = None,
                 start_threshold: float = DEFAULT_START_THRESHOLD,
                 link_threshold: float = DEFAULT_LINK_THRESHOLD,
                 forget_tau: float | None = None, forget_prune_floor: float | None = 0.05,
                 on_embedding_change: str = "raise"):
        if on_embedding_change not in self.EMBEDDING_CHANGE_POLICIES:
            raise ValueError(f"on_embedding_change 只能是 {self.EMBEDDING_CHANGE_POLICIES}，"
                             f"收到 {on_embedding_change!r}")
        self.embedder = embedder
        self.store_path = Path(store_path) if store_path else None
        self.graph = MemoryGraph.load(self.store_path) if self.store_path else MemoryGraph()
        self.start_threshold = start_threshold   # 检索起点所需的最低相似度
        self.link_threshold = link_threshold     # 两条记忆之间要有多像才连边
        self.forget_tau = forget_tau             # None 表示用 MemoryGraph 的默认值
        self.forget_prune_floor = forget_prune_floor
        self.on_embedding_change = on_embedding_change
        self._check_embedding_change()

    # ---------- 向量兼容性 ----------

    @property
    def embedding_name(self) -> str | None:
        """当前向量的模型标识；拿不到就返回 None，此时跳过模型校验"""
        name = getattr(self.embedder, "name", None)
        return name if name and name != "unknown" else None

    @property
    def embedding_dimension(self) -> int | None:
        """当前向量的维度；未知时返回 None"""
        return getattr(self.embedder, "dimension", None)

    def _embed_query(self, text: str) -> list[float]:
        """检索词单独编码：部分模型（如 bge 中文）查询侧需要加指令前缀"""
        embed_query = getattr(self.embedder, "embed_query", None)
        return embed_query(text) if callable(embed_query) else self.embedder.embed(text)

    def _check_embedding_change(self) -> None:
        """启动时检查：这份存档是不是用当前这套向量建立的

        换向量模型等于换了一套语义坐标系，新旧向量混在一起检索只会越来越乱，
        所以在这里就挑明，而不是等回忆结果变得莫名其妙。
        """
        saved_model = self.graph.meta.get("embedding_model")
        saved_dim = self.graph.meta.get("embedding_dim")
        if saved_model is None and saved_dim is None:
            return  # 空存档或旧版存档：没记录，无从校验

        current_model = self.embedding_name
        current_dim = self.embedding_dimension
        model_changed = bool(saved_model and current_model and saved_model != current_model)
        dim_changed = bool(saved_dim and current_dim and saved_dim != current_dim)
        if not (model_changed or dim_changed):
            return

        detail = (f"存档是用 {saved_model or '未知模型'}（{saved_dim or '?'} 维）建立的，"
                  f"当前是 {current_model or '未知模型'}（{current_dim or '?'} 维）")
        if self.on_embedding_change == "ignore":
            print(f"警告：{detail}，两者向量空间不一致，回忆结果可能不准。")
            return
        if self.on_embedding_change == "reset":
            print(f"警告：{detail}，已按 on_embedding_change='reset' 清空旧记忆重建。")
            self.graph = MemoryGraph()
            self._save()
            return
        raise EmbeddingChangedError(
            f"{detail}。新旧向量不在同一个语义空间，混用会让回忆变乱。请二选一：\n"
            f"  - 换回原来的向量模型；\n"
            f"  - 用 MemoryService(..., on_embedding_change='reset') 清空旧记忆重新开始。"
        )

    # ---------- 写入 ----------

    def remember(self, text: str, node_type: str = "fact",
                 tags: list[str] | None = None, props: dict | None = None,
                 link_top_k: int = 5) -> str:
        """记下一句话：向量化 → 建点 → 与已有记忆按相似度连边 → 落盘"""
        return self._insert(text, self.embedder.embed(text), node_type, tags, props,
                            link_top_k, save=True)

    def remember_many(self, texts: Sequence[str], node_type: str = "fact",
                      tags: list[str] | None = None, props: dict | None = None,
                      link_top_k: int = 5) -> list[str]:
        """一次记下多句话：批量编码后逐条建点连边，全程只落盘一次

        本地向量模型批量编码比逐条快得多，导入历史对话之类的场景用这个。
        """
        texts = list(texts)
        if not texts:
            return []
        embeddings = self.embedder.embed_batch(texts)
        if len(embeddings) != len(texts):
            raise ValueError(f"embed_batch 返回 {len(embeddings)} 条向量，"
                             f"但输入了 {len(texts)} 段文本")
        ids = [self._insert(text, embedding, node_type, tags, props, link_top_k, save=False)
               for text, embedding in zip(texts, embeddings)]
        self._save()
        return ids

    def _insert(self, text: str, embedding: list[float], node_type: str,
                tags: list[str] | None, props: dict | None, link_top_k: int,
                save: bool) -> str:
        """建点 + 连边 + 记录向量信息；save=False 时由调用方决定何时落盘"""
        nid = self.graph.add_node(text, embedding, tags=tags, node_type=node_type, props=props)

        # 与已有记忆建立相似度边（自己除外）
        for other_id, similarity in self.graph.retrieve(embedding, threshold=self.link_threshold,
                                                        top_k=link_top_k + 1):
            if other_id != nid:
                self.graph.add_edge(nid, other_id, similarity)

        # 记下这份记忆是用哪套向量建立的，换模型时能第一时间发现
        self.graph.meta["embedding_model"] = self.embedding_name or "unknown"
        self.graph.meta["embedding_dim"] = len(embedding)

        if save:
            self._save()
        return nid

    # ---------- 回忆 ----------

    def recall(self, query: str, top_k: int = 3, max_depth: int = 3,
               result_limit: int = 5) -> list[dict]:
        """回忆：先结算遗忘 → 检索起点 → 激活扩散 → 加权 → 返回权重最高的若干条记忆"""
        self.graph.apply_forgetting(tau=self.forget_tau, prune_floor=self.forget_prune_floor)

        starts = self.graph.retrieve(self._embed_query(query), threshold=self.start_threshold,
                                     top_k=top_k)
        if not starts:
            return []

        now = time.time()
        touched: dict[str, None] = {}
        for start_id, _ in starts:
            touched[start_id] = None
            activated = self.graph.bfs_activate(start_id, max_depth=max_depth)
            self.graph.update_weights(start_id, activated, current_time=now)
            for nid in activated:
                touched[nid] = None

        self._save()

        memories = [
            {
                "id": nid,
                "text": node.text,
                "weight": round(node.weight, 4),
                "retention": round(self.graph.retention(nid, now, tau=self.forget_tau), 4),
                "mention_count": node.mention_count,
                "node_type": node.node_type,
                "tags": node.tags,
                "depth": 0,
            }
            for nid in touched
            if (node := self.graph.nodes.get(nid)) is not None
        ]
        memories.sort(key=lambda m: m["weight"], reverse=True)
        return memories[:result_limit]

    def format_for_prompt(self, memories: list[dict], header: str = "【你脑海中浮现的记忆】") -> str:
        """把回忆结果拼成可直接注入提示词的片段"""
        if not memories:
            return ""
        lines = [f"- {m['text']}" for m in memories]
        return header + "\n" + "\n".join(lines)

    def _save(self) -> None:
        if self.store_path:
            self.graph.save(self.store_path)


if __name__ == "__main__":
    # 自检：建点、去重、连边、扩散、加权、存盘再加载
    import tempfile


    class _DemoEmbedder:
        """演示用的假向量器：按关键词生成 3 维向量，真实使用请注入真向量服务"""

        TABLE = {"螺蛳粉": 0, "柳州": 0, "猫": 1, "狗": 1, "python": 2, "代码": 2}

        def embed(self, text: str) -> list[float]:
            vector = [0.0, 0.0, 0.0]
            for keyword, index in self.TABLE.items():
                if keyword in text:
                    vector[index] += 1.0
            return vector if any(vector) else [0.1, 0.1, 0.1]


    with tempfile.TemporaryDirectory() as tmp:
        store = Path(tmp) / "memory.json"
        service = MemoryService(_DemoEmbedder(), store_path=store)

        service.remember("我最喜欢吃螺蛳粉，尤其是柳州的")
        service.remember("柳州是个好地方")
        service.remember("我喜欢写 python 代码")
        assert len(service.graph.nodes) == 3, "三段不同文本应该建三个节点"

        service.remember("柳州是个好地方")  # 完全相同的文本
        assert len(service.graph.nodes) == 3, "相同文本不应重复建点"

        hits = service.recall("螺蛳粉", top_k=1)
        assert hits and "螺蛳粉" in hits[0]["text"], f"最相关的记忆没排第一：{hits}"
        assert len(hits) == 2, f"扩散应把「柳州」一起带出来：{hits}"

        reloaded = MemoryGraph.load(store)
        assert len(reloaded.nodes) == len(service.graph.nodes), "存盘再加载，节点数应一致"
        assert set(reloaded.edges) == set(service.graph.edges), "存盘再加载，边应完整恢复"
        assert reloaded.retrieve([1.0, 0.0, 0.0], threshold=0.5, top_k=1), "重建索引后应能正常检索"

        print("自检通过：", [m["text"] for m in hits])
