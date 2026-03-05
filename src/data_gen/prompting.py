"""Prompt and message builders shared across data/train/eval pipelines."""

from __future__ import annotations

import ast
import json
from typing import Dict, List, Optional, Set

import numpy as np

from src.envs.game_2048 import ACTION_MAP

GAME_SYSTEM_PROMPT = """
你是 2048 游戏 AI。请分析输入的 4x4 棋盘数组，并给出当前最优方向（上/右/下/左）。

# 简明游戏规则
1. `0` 表示空位,非零数字则代表一个方块。
2. 每步只能选一个方向，所有方块先向该方向滑动压缩。
3. 压缩后相邻且相同的非零方块合并为和；原本隔空位可在压缩后合并；被不同数值阻挡时不能跨越合并；单个方块每步最多合并一次。
4. 执行后棋盘有变化（移动或合并）即合法；完全不变即非法。

# 输出要求
输出一个 JSON 对象，格式如下：
1. "局面": object
   - "最大数字": int，表示当前棋盘中的最大数值。
   - "位置": list，表示所有最大数字坐标；每个坐标格式为 [行, 列]（0-based）。
   - "在角落": bool，表示是否至少有一个最大数字位于四个角之一。
2. "判断": object
   - 固定包含四个键："上"、"右"、"下"、"左"。
   - 每个键的值为 bool，表示该方向本步是否为合法动作。
3. "选择": string
   - 表示最终动作方向，只能是 "上"、"右"、"下"、"左" 之一。

# 示例
输入：
[[2, 2, 4, 8],
[4, 0, 0, 8],
[0, 0, 0, 0],
[0, 0, 0, 0]]
输出：
{
  "局面": {"最大数字": 8, "位置": [[0, 3], [1, 3]], "在角落": true},
  "判断": {"上": false, "右": true, "下": true, "左": true},
  "选择": "右"
}
"""



def build_user_prompt(state_text: str, use_thinking: bool) -> str:
    return f"""{GAME_SYSTEM_PROMPT}
# 当前输入
{state_text}

# 输出限制
只输出一个 JSON 对象，不要输出代码块、解释或额外文本。
"""


def _parse_grid(state_text: str) -> Optional[List[List[int]]]:
    try:
        grid = ast.literal_eval(state_text)
    except Exception:
        return None
    if not isinstance(grid, list) or len(grid) != 4:
        return None
    parsed: List[List[int]] = []
    for row in grid:
        if not isinstance(row, list) or len(row) != 4:
            return None
        row_vals: List[int] = []
        for v in row:
            if not isinstance(v, int):
                return None
            row_vals.append(int(v))
        parsed.append(row_vals)
    return parsed


def _compress_and_merge_line(line: np.ndarray) -> np.ndarray:
    """Return merged line following 2048 move rules."""
    non_zero = line[line > 0]
    merged: List[int] = []
    i = 0
    while i < len(non_zero):
        if i + 1 < len(non_zero) and int(non_zero[i]) == int(non_zero[i + 1]):
            merged.append(int(non_zero[i]) * 2)
            i += 2
        else:
            merged.append(int(non_zero[i]))
            i += 1

    result = np.zeros(4, dtype=int)
    result[: len(merged)] = merged
    return result


def _simulate_move(grid: np.ndarray, action: int) -> np.ndarray:
    """Simulate one move on a copied board without RNG side effects."""
    moved = grid.copy()

    if action == 0:  # up
        for j in range(4):
            moved[:, j] = _compress_and_merge_line(moved[:, j])
    elif action == 1:  # right
        for i in range(4):
            moved[i, :] = _compress_and_merge_line(moved[i, ::-1])[::-1]
    elif action == 2:  # down
        for j in range(4):
            moved[:, j] = _compress_and_merge_line(moved[::-1, j])[::-1]
    elif action == 3:  # left
        for i in range(4):
            moved[i, :] = _compress_and_merge_line(moved[i, :])

    return moved


def _valid_action_ids_from_grid(grid: np.ndarray) -> Set[int]:
    """Compute valid action ids directly from grid, side-effect free."""
    valid_ids: Set[int] = set()
    for action in range(4):
        if not np.array_equal(grid, _simulate_move(grid=grid, action=action)):
            valid_ids.add(action)
    return valid_ids


def build_action_json_text(state_text: str, action: str) -> str:
    action_name = str(action or "").strip()
    grid = _parse_grid(state_text)
    if grid is None:
        payload = {
            "局面": {"最大数字": 0, "位置": [], "在角落": False},
            "判断": {"上": False, "右": False, "下": False, "左": False},
            "选择": action_name,
        }
        return json.dumps(payload, ensure_ascii=False)

    arr = np.array(grid, dtype=int)
    max_tile = int(np.max(arr))
    positions: List[List[int]] = []
    for i in range(4):
        for j in range(4):
            if int(arr[i, j]) == max_tile:
                positions.append([int(i), int(j)])
    in_corner = any((i, j) in {(0, 0), (0, 3), (3, 0), (3, 3)} for i, j in positions)

    valid_ids = _valid_action_ids_from_grid(arr)
    judgment = {ACTION_MAP[i]: (i in valid_ids) for i in range(4)}

    payload = {
        "局面": {
            "最大数字": max_tile,
            "位置": positions,
            "在角落": bool(in_corner),
        },
        "判断": judgment,
        "选择": action_name,
    }
    return json.dumps(payload, ensure_ascii=False)



def build_assistant_content(
    action: str,
    state_text: str,
    use_thinking: bool,
    thinking: Optional[str] = None,
    action_json_text: Optional[str] = None,
) -> str:
    action_json = (action_json_text or "").strip() or build_action_json_text(
        state_text=state_text,
        action=action,
    )
    if use_thinking:
        thinking_text = (thinking or "分析当前局势，优先保持大数字在角落并寻找可合并方向。").strip()
        return f"<think>\n{thinking_text}\n</think>\n\n{action_json}"
    return action_json



def build_messages(
    state_text: str,
    action: Optional[str] = None,
    use_thinking: bool = False,
    thinking: Optional[str] = None,
    action_json_text: Optional[str] = None,
) -> List[Dict[str, str]]:
    user_content = build_user_prompt(state_text=state_text, use_thinking=use_thinking)
    messages: List[Dict[str, str]] = [{"role": "user", "content": user_content}]

    if action is not None:
        assistant_content = build_assistant_content(
            action=action,
            state_text=state_text,
            use_thinking=use_thinking,
            thinking=thinking,
            action_json_text=action_json_text,
        )
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
