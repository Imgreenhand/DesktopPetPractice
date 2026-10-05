import os
from ai_adapter.deepseek_adapter import DeepSeekAdapter
from ai_adapter.llm_provider import UnifiedResponse 

def main_while(Key):
    # 默认人设：DeepSeek 鲸鱼娘。想换个形象，改这段文字就行
    SYSTEM_PROMPT = (
        "你是 DeepSeek 的鲸鱼娘，名字叫小鲸。"
        "性格温柔、聪明，有点黏人，说话自然不生硬，句尾偶尔带个「~」。"
        "偶尔会提到深海、气泡、鲸歌这些和自己有关的东西，但不要刻意卖萌。"
        "回答保持简洁，就像朋友聊天一样。"
    )
    MAX_HISTORY :int = 10 # 最大历史对话轮数，超过就压缩
    KEEP_HISTORY :int = 4 # 保留的最近对话
    history: list[dict] = []
    ds = DeepSeekAdapter(Key)
    while True:
        # 窗口压缩
        if len(history) > MAX_HISTORY:
            ds.window_summary(keep_history=KEEP_HISTORY, the_history=history)
        # 压缩完成
        pr = input("你想对我说什么呀: ")
        if pr == "/STOP":
            return
        current_messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ] + history + [{"role": "user", "content": pr}]
        a: UnifiedResponse = ds.generate(
            messages=current_messages
        )
        print(f"思考过程：\n{a.reasoning}\n") 
        print(f"输出：\n{a.content}")
        print(f"—— {a.model}\n")
        print(f"input_tokens: {a.usage['input_tokens']}")
        print(f"output: {a.usage['output_tokens']}")
        history.append({"role": "user", "content": pr})      
        history.append({"role": "assistant", "content": a.content})

if __name__ == "__main__":
    KEY = input("APIKEY= ")
    main_while(Key=KEY)
