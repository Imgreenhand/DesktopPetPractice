# DesktopPetPractice（桌宠练习）

一个用 Python 编写的「AI 桌宠」练习项目：把大语言模型接到你的电脑里，让它能聊天、有性格，并且像人一样**用一张记忆网络来记住和联想**。

目前项目处于早期阶段，已实现的是「LLM 对话 + 长期记忆图 + 真实语义向量」的核心骨架，桌面 UI（PySide6）部分仍在开发中。

形象完全可自定义：默认人设是 **DeepSeek 鲸鱼娘**，人设提示词就写在 `main.py` 的 `SYSTEM_PROMPT` 里，改几行字就能换成任何你喜欢的角色。

## 功能特性

- **统一 LLM 接口**：`LLMProvider` 抽象基类定义 `generate()` 与 `window_summary()`，方便接入不同模型供应商；目前提供 DeepSeek 实现。
- **思考过程可见**：统一响应 `UnifiedResponse` 同时返回正文、推理内容和 token 用量，方便调试与成本核算。
- **上下文窗口压缩**：对话轮数超过上限时自动把早期对话总结成摘要，只保留最近几轮原文，避免上下文爆炸。
- **真实语义向量**：`EmbeddingProvider` 抽象基类定义向量接口，`LocalEmbeddingAdapter` 用 sentence-transformers 在本机离线跑向量模型（默认 `BAAI/bge-small-zh-v1.5`），记忆检索靠真正的语义相似度而不是关键词匹配；换模型只要换一个适配器，记忆逻辑完全不用动。
- **长期记忆图**：基于「记忆节点 + 相似度边权」的图结构，配合 BFS 激活扩散、权重衰减和加权轮盘赌，实现类人的联想式回忆（算法说明见下文）。
- **人设可切换**：默认形象是 DeepSeek 鲸鱼娘，人设提示词独立于对话逻辑，改一段文字就能换成别的角色。

## 项目结构

```
DesktopPetPractice/
├── main.py                        # 程序入口：命令行对话循环 + 窗口压缩调度
├── ai_adapter/                    # 模型接入层：只定义接口，不绑定具体供应商
│   ├── llm_provider.py            # LLMProvider 抽象基类 & UnifiedResponse 统一响应格式
│   ├── deepseek_adapter.py        # DeepSeek 适配器（含错误处理与友好提示）
│   ├── embedding_provider.py      # EmbeddingProvider 抽象基类 & 按名字创建实现的工厂
│   └── local_embedding_adapter.py # 本地向量模型（sentence-transformers）
├── long_memory/
│   ├── memory_graph.py            # 记忆图的构建、检索、激活扩散与持久化
│   ├── memory_service.py          # 记忆服务门面（向量 → 检索 → 扩散 → 注入提示词）
│   ├── vector_calculation.py      # 向量与相似度计算
│   └── __init__.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 快速开始

1. 克隆仓库并创建虚拟环境：

   ```bash
   git clone https://github.com/Imgreenhand/DesktopPetPractice.git
   cd DesktopPetPractice
   python -m venv .venv
   ```

2. 激活虚拟环境并安装依赖：

   ```bash
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate

   pip install -r requirements.txt
   ```

3. 运行：

   ```bash
   python main.py
   ```

   启动后按提示输入 API Key 即可对话，输入 `/STOP` 退出。

4. 长期记忆的向量模型：

   默认用本地模型 `BAAI/bge-small-zh-v1.5`，首次运行自动下载权重（约 92 MB），之后完全离线
   （模型缓存在本机，启动时不会再联网检查更新）。编码一句约 10 ms，整句向量 512 维。
   默认装的是 CPU 版 torch；如果你有 N 卡并装了 CUDA 版 torch，会自动跑在显卡上。

   想换更准的模型，直接换参数字符串即可：

   ```python
   from ai_adapter.local_embedding_adapter import LocalEmbeddingAdapter
   from long_memory import MemoryService

   embedder = LocalEmbeddingAdapter("BAAI/bge-base-zh-v1.5")   # 768 维，更准也更慢
   service = MemoryService(embedder, store_path="memory.json")

   service.remember("我最爱吃螺蛳粉，尤其是柳州的")
   print(service.recall("我喜欢吃什么小吃？"))
   ```

   想确认向量功能是否正常（语义相似度对比 + 一次完整的记住/回忆 + 存档兼容性）：

   ```bash
   python -m ai_adapter.local_embedding_adapter
   ```

   不想装那 1 GB 的向量模型依赖、只想跑通记忆逻辑时，用这个零依赖自检：

   ```bash
   python -m long_memory.memory_service
   ```

   > 注意：换向量模型等于换了一套语义坐标系，旧存档的向量没法接着用。程序启动时若发现
   > 存档与当前模型对不上会直接报错提醒；确认要清空重建时，传
   > `MemoryService(..., on_embedding_change="reset")`。

   > 国内网络连 HuggingFace 不稳定时，先设镜像再跑：
   > Windows `set HF_ENDPOINT=https://hf-mirror.com`，
   > macOS / Linux `export HF_ENDPOINT=https://hf-mirror.com`。

> 提示：API Key 请勿写进代码或提交到仓库。建议后续改为从环境变量（如 `DEEPSEEK_API_KEY`）读取，`.gitignore` 已忽略 `.env` 等本地配置文件。

## 长期记忆算法

> 你们真的不觉得写这种长期记忆很爽吗？就是如同造物主一般的感觉

是滴，俺有一个猎奇的想法：

```
算法概括：
    人的记忆通常是一个网，当我们在触发某一点的记忆的时候，
    与之相连的记忆也有概率被唤醒
    因此在这个算法中，当用户唤醒某一个记忆的时候，
    我们将依据这张图的边权和与之联系的所有记忆进行一个BFS， 
    然后更新记忆在记忆库里的权重，这个权重 += (edge_cost * a^deep) -> a = 0.5
    如果某个记忆节点在过去1小时内刚被激活过，本次权重加成乘以 0.3，
    避免桌宠陷入重复话题
    边权为相似度[0, 1]->相似度为0说明两个node之间没有edge
    相似度为1就是这个节点本身
    然后人是可以主动变更话题的，我们可以让LLM判断这个话题可不可以继续，
    需不需要开启新话题，然后如果可以开启的话，用一次轮盘赌算法随机找到
    用于联想的的记忆，然后让LLM根据自己设定的性格和爱好，
    还有这个被检索到的记忆进行一次发散式的搜索，
    在网上找视频，八卦，帖子
    以此提升桌宠在聊天上的自主性

一些优化策略：

    剪枝：

    限制层数：最多只做 深度≤3 的BFS。
    扇出剪枝：在当前节点度 > 50 时，只取边权最高的 Top 10 邻居进行扩展。
    更新目标：只更新被BFS触及的节点，以及起始节点的权重
            （起始节点获得额外 +0.5 奖励，代表“触景生情”）

    话题切换:

    让LLM二分类判断“是否切换”容易产生幻觉（模型总是倾向于顺从用户）。
    建议改为“兴趣衰减评分”：
    在提示词中让LLM输出 Topic_Exhaustion_Score（0-10）。
    - 0~3：当前话题兴致正浓，禁止轮盘赌，继续深挖当前节点的一级邻居。
    - 4~6：略显疲态，触发概率性轮盘赌（30%概率切，70%继续）。
    - 7~10：话题枯竭，强制触发轮盘赌联想。

    轮盘赌算法的核心改良(别用纯随机):

    纯随机可能会抽到十几年前设定里的“前男友”或“黑历史”，导致聊天情绪断裂。
    加权轮盘赌（引入“探索与利用”平衡）：
    每个记忆节点的权重 = 原始权重 + (1 / 最后访问时间差)。
    加入“性格偏差”：如果你的桌宠性格是“乐子人”，
    在轮盘赌中给“八卦/搞笑”类标签的节点额外增加 +20% 的中奖概率；
    如果是“忧郁文学少年”，则给“伤感/回忆”类节点加权重。

    记忆写入的反向操作:

    当桌宠从网上找到新信息并说完后，
    必须将这次“聊天+搜索”的结果作为新的记忆节点写回图中，
    边权连接当前话题节点。
    这样做的好处：随着使用时间增长，这张图会越来越“像”用户本人，
    桌宠的联想会越来越精准。
```

## 开发计划

- [ ] 接入 PySide6，实现真正的桌面宠物窗口（立绘、气泡对话、托盘常驻）
- [x] 记忆持久化存储（JSON 存档，原子写入）与相似度检索
- [x] 真实语义向量（本地 sentence-transformers，适配器可插拔）
- [ ] 在线向量接口适配（OpenAI / 硅基流动 / 智谱 / Ollama）
- [ ] 把长期记忆接进对话循环（检索 → 注入提示词 → 写回新记忆）
- [ ] 人设配置文件与立绘资源（默认形象 DeepSeek 鲸鱼娘，可自定义）
- [ ] 更多 LLM 供应商适配（Claude / Gemini / 本地模型）
- [ ] 联网搜索与自主联想

## 说明

本项目为个人学习与实验性质的作品，与 DeepSeek 官方无关；「DeepSeek」等相关名称与素材的权利归其权利人所有，默认人设仅供个人学习练习使用。
