"""Build short-chain SFT data from processed next-board datasets."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datasets import Dataset, load_from_disk

from src.data_gen.next_board_processor import validate_next_board_sample


SHORT_THINKING_SYSTEM_PROMPT = """Simulate exactly one 2048 move.

The board is 4 rows by 4 columns: board[r][c].
Row 0 is top, row 3 is bottom, column 0 is left, column 3 is right.

Apply the rule to each affected line only:
1. remove zeros
2. merge adjacent equal tiles once
3. pad zeros on the far side

UP/DOWN operate on columns.
LEFT/RIGHT operate on rows.
Rebuild the final answer as 4 rows.

If you think, keep it short:
- line 1: which lines are affected
- line 2: one or two example transformations
- line 3: rebuild rows

Output the final answer as strict JSON after </think>.
"""


def _extract_board_from_user_prompt(user_text: str) -> Optional[List[List[int]]]:
    if "Current board:\n" not in user_text or "\n\nMove direction:" not in user_text:
        return None
    board_text = user_text.split("Current board:\n", 1)[1].split("\n\nMove direction:", 1)[0]
    try:
        board = ast.literal_eval(board_text)
    except Exception:
        return None
    if not isinstance(board, list) or len(board) != 4:
        return None
    parsed: List[List[int]] = []
    for row in board:
        if not isinstance(row, list) or len(row) != 4:
            return None
        parsed_row: List[int] = []
        for value in row:
            if not isinstance(value, int):
                return None
            parsed_row.append(int(value))
        parsed.append(parsed_row)
    return parsed


def _extract_action_from_user_prompt(user_text: str) -> Optional[str]:
    if "Move direction:" not in user_text or "\n\nCompute the next board exactly." not in user_text:
        return None
    action_text = user_text.split("Move direction:", 1)[1].split("\n\nCompute the next board exactly.", 1)[0]
    action = str(action_text).strip().upper()
    return action if action in {"UP", "RIGHT", "DOWN", "LEFT"} else None


def _compress_merge_line(line: List[int]) -> List[int]:
    non_zero = [int(v) for v in line if int(v) != 0]
    merged: List[int] = []
    idx = 0
    while idx < len(non_zero):
        if idx + 1 < len(non_zero) and non_zero[idx] == non_zero[idx + 1]:
            merged.append(non_zero[idx] * 2)
            idx += 2
        else:
            merged.append(non_zero[idx])
            idx += 1
    return merged + [0] * (4 - len(merged))


def _line_before_after(board: List[List[int]], action: str, line_idx: int) -> Tuple[List[int], List[int]]:
    if action == "UP":
        before = [board[r][line_idx] for r in range(4)]
        after = _compress_merge_line(before)
        return before, after
    if action == "DOWN":
        before = [board[r][line_idx] for r in range(4)]
        after = list(reversed(_compress_merge_line(list(reversed(before)))))
        return before, after
    if action == "LEFT":
        before = list(board[line_idx])
        after = _compress_merge_line(before)
        return before, after
    before = list(board[line_idx])
    after = list(reversed(_compress_merge_line(list(reversed(before)))))
    return before, after


def build_short_thinking(board: List[List[int]], action: str, next_board: List[List[int]]) -> str:
    axis = "columns" if action in {"UP", "DOWN"} else "rows"
    changed: List[str] = []
    for idx in range(4):
        before, after = _line_before_after(board, action, idx)
        if before != after:
            prefix = f"c{idx}" if axis == "columns" else f"r{idx}"
            changed.append(f"{prefix}: {before}->{after}")

    if not changed:
        changed.append("no line changes")

    example = "; ".join(changed[:2])
    return (
        f"{action} uses {axis}.\n"
        f"{example}\n"
        f"Rebuild 4 rows -> {next_board}"
    )


def build_sft_sample(sample: Dict) -> Dict:
    prompt = sample.get("prompt", [])
    completion = sample.get("completion", [])
    if not isinstance(prompt, list) or len(prompt) != 2:
        raise ValueError("source sample prompt must contain two messages")
    if not isinstance(completion, list) or len(completion) != 1:
        raise ValueError("source sample completion must contain one message")

    user_text = str(prompt[1].get("content", ""))
    board = _extract_board_from_user_prompt(user_text)
    action = _extract_action_from_user_prompt(user_text)
    try:
        target_payload = json.loads(str(completion[0].get("content", "")))
    except json.JSONDecodeError as exc:
        raise ValueError("source completion must be valid JSON") from exc
    next_board = target_payload.get("next_board")
    if board is None or action is None or not isinstance(next_board, list):
        raise ValueError("failed to parse source sample")

    thinking = build_short_thinking(board=board, action=action, next_board=next_board)
    assistant_text = f"<think>\n{thinking}\n</think>\n\n{json.dumps({'next_board': next_board}, ensure_ascii=False)}"

    return {
        "prompt": [
            {"role": "system", "content": SHORT_THINKING_SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        "completion": [
            {"role": "assistant", "content": assistant_text},
        ],
        "schema_version": sample.get("schema_version", "1.0"),
        "task_type": "next_board_prediction_short_cot",
        "source_type": sample.get("source_type", "processed_next_board"),
        "action": sample.get("action"),
        "action_id": sample.get("action_id"),
    }


def transform_dataset(dataset: Dataset) -> Dataset:
    rows: List[Dict] = []
    for sample in dataset:
        rows.append(build_sft_sample(sample))
    return Dataset.from_list(rows)


def validate_short_thinking_sample(sample: Dict) -> Tuple[bool, Optional[str]]:
    ok, err = validate_next_board_sample(sample, require_thinking=True)
    if not ok:
        return ok, err
    assistant_text = sample["completion"][0]["content"]
    think_text = assistant_text.split("<think>\n", 1)[1].split("\n</think>", 1)[0]
    if len([line for line in think_text.splitlines() if line.strip()]) > 3:
        return False, "thinking should contain at most 3 non-empty lines"
    return True, None


def process_split(input_dir: str, output_dir: str, split: str) -> None:
    src = Path(input_dir) / split
    dst = Path(output_dir) / split
    dataset = load_from_disk(str(src))
    transformed = transform_dataset(dataset)
    for idx, sample in enumerate(transformed):
        ok, err = validate_short_thinking_sample(sample)
        if not ok:
            raise ValueError(f"{split} sample {idx} invalid: {err}")
    transformed.save_to_disk(str(dst))
    print(f"Saved {split}: {len(transformed)} samples -> {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build short-chain next-board SFT datasets")
    parser.add_argument("--input_dir", type=str, default="data/processed_next_board")
    parser.add_argument("--output_dir", type=str, default="data/processed_next_board_sft")
    args = parser.parse_args()

    for split in ("train", "val", "test"):
        process_split(args.input_dir, args.output_dir, split)


if __name__ == "__main__":
    main()
