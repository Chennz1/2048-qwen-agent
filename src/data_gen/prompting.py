"""Prompt and message builders shared across data/train/eval pipelines."""

from __future__ import annotations

from typing import Dict, List, Optional

GAME_SYSTEM_PROMPT = """# 任务
你是 2048 游戏的最强 AI 求解器。请分析输入的 4x4 棋盘数组，并输出当前的最优移动方向。

# 核心启发式策略 (Heuristics)
1. 蛇形/单调性：将最大数字固定在棋盘角落（首选左上），其余数字按贪心策略呈蛇形递减排列。
2. 空间控制：优先选择能产生合并、最大化空位数量的移动。
3. 危险规避：绝对禁止向最大数所在的对立方向移动，防止最大数被迫移出角落。

# 决策示例
示例输入:
[[2, 2, 4, 8],
 [4, 0, 0, 0],
 [0, 0, 0, 0],
 [0, 0, 0, 0]]
示例输出:
左

示例输入:
[[1024, 512, 64, 16],
 [8, 4, 2, 2],
 [0, 0, 0, 0],
 [0, 0, 0, 0]]
示例输出:
左
"""



def build_user_prompt(state_text: str, use_thinking: bool) -> str:
    return f"""{GAME_SYSTEM_PROMPT}
# 当前输入
{state_text}

# 严格输出限制
禁止输出任何解释或标点符号。必须且只能输出以下四个字中的一个：
上、右、下、左
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
