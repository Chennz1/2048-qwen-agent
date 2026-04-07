"""Prompt and message builders shared across data/train/eval pipelines."""

from __future__ import annotations

import ast
import json
from typing import Dict, List, Optional, Set

import numpy as np

from src.envs.game_2048 import ACTION_MAP, ACTION_MAP_ENGLISH

GAME_SYSTEM_PROMPT = """
You are a 2048 game AI. Analyze the input 4x4 board and choose the best direction (UP/RIGHT/DOWN/LEFT).

# Rules
1. `0` means an empty cell and any non-zero value is a tile.
2. Each move chooses exactly one direction, and all tiles slide in that direction first.
3. After sliding, adjacent equal non-zero tiles merge into their sum. Tiles separated by zeros may merge after compression. Tiles blocked by different values cannot merge across them. A tile can merge at most once per move.
4. A move is legal if the board changes after execution, either by sliding or merging. If the board stays exactly the same, the move is illegal.

# Output Format
Return one JSON object with this schema:
1. "board": object
   - "max_tile": int, the largest value on the current board.
   - "positions": list, all coordinates of the largest value; each coordinate is [row, col] with 0-based indexing.
   - "in_corner": bool, whether at least one largest tile is in a corner.
2. "judgment": object
   - Must contain exactly four keys: "UP", "RIGHT", "DOWN", "LEFT".
   - Each value is a bool indicating whether that direction is legal for this move.
3. "choice": string
   - The final chosen direction, and it must be one of "UP", "RIGHT", "DOWN", "LEFT".

# Example
Input:
[[2, 2, 4, 8],
[4, 0, 0, 8],
[0, 0, 0, 0],
[0, 0, 0, 0]]
Output:
{
  "board": {"max_tile": 8, "positions": [[0, 3], [1, 3]], "in_corner": true},
  "judgment": {"UP": false, "RIGHT": true, "DOWN": true, "LEFT": true},
  "choice": "RIGHT"
}
"""



def build_user_prompt(state_text: str, use_thinking: bool) -> str:
    return f"""{GAME_SYSTEM_PROMPT}
# Current Input
{state_text}

# Output Restriction
Return exactly one JSON object. Do not output code fences, explanations, or any extra text.
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
    action_lookup = {name: idx for idx, name in ACTION_MAP.items()}
    english_choice = ACTION_MAP_ENGLISH.get(action_lookup.get(str(action or "").strip()))
    action_name = english_choice or str(action or "").strip().upper()
    grid = _parse_grid(state_text)
    if grid is None:
        payload = {
            "board": {"max_tile": 0, "positions": [], "in_corner": False},
            "judgment": {"UP": False, "RIGHT": False, "DOWN": False, "LEFT": False},
            "choice": action_name,
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
    judgment = {ACTION_MAP_ENGLISH[i]: (i in valid_ids) for i in range(4)}

    payload = {
        "board": {
            "max_tile": max_tile,
            "positions": positions,
            "in_corner": bool(in_corner),
        },
        "judgment": judgment,
        "choice": action_name,
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
        thinking_text = (
            thinking
            or "Analyze the board, keep the largest tile stable, and prefer a direction that preserves structure or creates merges."
        ).strip()
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
