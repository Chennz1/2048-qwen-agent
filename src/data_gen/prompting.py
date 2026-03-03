"""Prompt and message builders shared across data/train/eval pipelines."""

from __future__ import annotations

from typing import Dict, List, Optional

GAME_SYSTEM_PROMPT = """# 任务
你是 2048 游戏 AI。请分析输入的 4x4 棋盘数组，输出当前最优方向（上/右/下/左）。

# 简要游戏规则
1. `0` 表示空位,非零数字则代表一个方块。
2. 每步只能选一个方向，所有方块先向该方向滑动压缩。
3. 压缩后相邻且相同的非零方块合并为和；原本隔空位可在压缩后合并；被不同数值阻挡时不能跨越合并；单个方块每步最多合并一次。
4. 执行后棋盘有变化（移动或合并）即合法；完全不变即非法。
5. 合法移动后会在空位随机生成 2 或 4。

# 常用策略
1. 认真判断动作可行性,不要选择非法动作。
2. 在合法动作中优先选择可合并、增空位、保留后续机动性的方向。
3. 尽量让最大数字稳定在一个角落，并保持整体单调。
4. 无明确收益时，不打散已形成的角落结构。

# 决策示例
示例输入:
[[2, 2, 4, 8],
 [4, 0, 0, 8],
 [0, 0, 0, 0],
 [0, 0, 0, 0]]
示例输出:
上
"""



def build_user_prompt(state_text: str, use_thinking: bool) -> str:
    return f"""{GAME_SYSTEM_PROMPT}
# 当前输入
{state_text}

# 输出限制
只能输出最终选择的动作(上、右、下、左 其中之一)
"""



def build_assistant_content(action: str, use_thinking: bool, thinking: Optional[str] = None) -> str:
    if use_thinking:
        thinking_text = (thinking or "分析当前局势，优先保持大数字在角落并寻找可合并方向。").strip()
        return f"<think>\n{thinking_text}\n</think>\n\n{action}"
    return action



def build_messages(
    state_text: str,
    action: Optional[str] = None,
    use_thinking: bool = False,
    thinking: Optional[str] = None,
) -> List[Dict[str, str]]:
    user_content = build_user_prompt(state_text=state_text, use_thinking=use_thinking)
    messages: List[Dict[str, str]] = [{"role": "user", "content": user_content}]

    if action is not None:
        assistant_content = build_assistant_content(action=action, use_thinking=use_thinking, thinking=thinking)
        messages.append({"role": "assistant", "content": assistant_content})

    return messages



def format_sample_text(tokenizer, messages: List[Dict[str, str]], apply_chat_template: bool) -> str:
    """Return a training text sample from chat messages."""
    if apply_chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    user = messages[0]["content"]
    if len(messages) == 1:
        return user
    return f"{user}\n\n{messages[1]['content']}"



def format_inference_prompt(tokenizer, state_text: str, use_thinking: bool = True) -> str:
    """Return an inference prompt with add_generation_prompt=True."""
    messages = build_messages(state_text=state_text, action=None, use_thinking=use_thinking)
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }

    # Qwen3 tokenizer supports enable_thinking; keep backward compatibility.
    if use_thinking:
        kwargs["enable_thinking"] = True

    return tokenizer.apply_chat_template(messages, **kwargs)
