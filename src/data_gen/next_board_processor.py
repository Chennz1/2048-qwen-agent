"""Build synthetic supervised data for predicting the next 2048 board."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from datasets import Dataset

from src.data_gen.contracts import summarize_validation_failures
from src.data_gen.processor import save_datasets, split_dataset
from src.envs.game_2048 import ACTION_MAP_ENGLISH, Game2048


OUTPUT_SCHEMA_VERSION = "1.0"


SYSTEM_PROMPT = """Simulate exactly one 2048 move.

The board is 4 rows by 4 columns: board[r][c].
Row 0 is top, row 3 is bottom, column 0 is left, column 3 is right.

Rule for each affected line:
1. remove zeros
2. merge adjacent equal tiles once
3. fill remaining cells with zeros on the far side

UP/DOWN operate on columns.
LEFT/RIGHT operate on rows.
Rebuild the final answer as 4 rows. Never treat processed columns as final rows.

Output only:
```json
{"next_board": [[...], [...], [...], [...]]}
```"""


def format_board_text(board: List[List[int]]) -> str:
    """Render a board in the same multiline format used by the prompts."""
    lines = []
    for row in board:
        lines.append("  [" + ", ".join(str(int(v)) for v in row) + "]")
    return "[\n" + ",\n".join(lines) + "\n]"


def simulate_next_board(board: List[List[int]], action_id: int) -> List[List[int]]:
    """Return the deterministic post-move board without spawning a new tile."""
    game = Game2048(seed=0)
    game.grid = np.array(board, dtype=int)
    game.score = 0
    game.game_over = False
    game._move(int(action_id))
    return game.grid.astype(int).tolist()


def get_valid_action_ids(board: List[List[int]]) -> List[int]:
    """Return legal move ids for a board."""
    game = Game2048(seed=0)
    game.grid = np.array(board, dtype=int)
    game.score = 0
    game.game_over = False
    return [int(action_id) for action_id in game.get_valid_actions()]


def evaluate_action(board: List[List[int]], action_id: int) -> Tuple[float, List[List[int]]]:
    """Lightweight heuristic to prefer legal, non-trivial directions."""
    next_board = simulate_next_board(board, action_id)
    arr = np.array(next_board, dtype=int)
    empty_cells = int(np.sum(arr == 0))
    max_tile = int(np.max(arr))
    corners = [int(arr[0, 0]), int(arr[0, 3]), int(arr[3, 0]), int(arr[3, 3])]
    max_in_corner = any(tile == max_tile for tile in corners)
    merge_gain = int(np.sum(arr) - np.sum(np.array(board, dtype=int)))
    score = float(empty_cells * 5 + (20 if max_in_corner else 0) + merge_gain)
    return score, next_board


def choose_reasonable_action(board: List[List[int]], rng: random.Random) -> Tuple[int, List[List[int]]]:
    """Choose a legal direction with mild heuristic bias and some randomness."""
    valid_actions = get_valid_action_ids(board)
    if not valid_actions:
        raise ValueError("board has no legal actions")

    scored: List[Tuple[float, int, List[List[int]]]] = []
    for action_id in valid_actions:
        score, next_board = evaluate_action(board, action_id)
        scored.append((score, action_id, next_board))

    scored.sort(key=lambda item: item[0], reverse=True)
    top_k = scored[: min(2, len(scored))]
    picked_score, picked_action, picked_board = rng.choice(top_k)
    _ = picked_score
    return int(picked_action), picked_board


def build_user_prompt(state_text: str, action_name: str) -> str:
    return (
        "Current board:\n"
        f"{state_text}\n\n"
        f"Move direction: {action_name}\n\n"
        "Compute the next board exactly.\n"
    )


def build_assistant_content(next_board: List[List[int]]) -> str:
    return json.dumps({"next_board": next_board}, ensure_ascii=False)


def build_messages(state_text: str, action_name: str, next_board: List[List[int]]) -> Dict[str, List[Dict[str, str]]]:
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(state_text, action_name)},
        ],
        "completion": [
            {"role": "assistant", "content": build_assistant_content(next_board)},
        ],
    }


def validate_next_board_sample(sample: Dict, require_thinking: bool = False) -> Tuple[bool, Optional[str]]:
    prompt = sample.get("prompt")
    completion = sample.get("completion")
    if not isinstance(prompt, list) or len(prompt) != 2:
        return False, "prompt must contain exactly two messages"
    if not isinstance(completion, list) or len(completion) != 1:
        return False, "completion must contain exactly one message"

    system_msg = prompt[0]
    user_msg = prompt[1]
    assistant_msg = completion[0]
    if system_msg.get("role") != "system" or not isinstance(system_msg.get("content"), str):
        return False, "prompt[0] must be a system message"
    if user_msg.get("role") != "user" or not isinstance(user_msg.get("content"), str):
        return False, "prompt[1] must be a user message"
    if assistant_msg.get("role") != "assistant" or not isinstance(assistant_msg.get("content"), str):
        return False, "completion[0] must be an assistant message"

    if require_thinking and "<think>" not in assistant_msg["content"]:
        return False, "thinking samples must contain <think>"

    payload_text = assistant_msg["content"].strip()
    if require_thinking:
        close_idx = payload_text.rfind("</think>")
        if close_idx < 0:
            return False, "thinking samples must close </think>"
        payload_text = payload_text[close_idx + len("</think>"):].strip()

    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        return False, "assistant content must be valid JSON"

    if set(payload.keys()) != {"next_board"}:
        return False, "assistant JSON must contain only next_board"

    next_board = payload["next_board"]
    if not isinstance(next_board, list) or len(next_board) != 4:
        return False, "next_board must be a 4x4 list"
    for row in next_board:
        if not isinstance(row, list) or len(row) != 4:
            return False, "next_board must be a 4x4 list"
        for value in row:
            if not isinstance(value, int):
                return False, "next_board values must be ints"

    return True, None


def validate_dataset(dataset: Dataset, name: str = "Dataset", require_thinking: bool = False) -> Dict:
    """Validate processed next-board dataset quality."""
    print(f"\n=== 验证 {name} ===")
    issues: List[str] = []
    valid_count = 0

    for i, sample in enumerate(dataset):
        is_valid, error = validate_next_board_sample(sample, require_thinking=require_thinking)
        if is_valid:
            valid_count += 1
        else:
            issues.append(f"样本 {i}: {error}")

    stats = {
        "total_samples": len(dataset),
        "valid_samples": valid_count,
        "invalid_samples": len(dataset) - valid_count,
        "issues": issues[:10],
    }

    print(f"总样本数: {stats['total_samples']}")
    print(f"有效样本: {stats['valid_samples']}")
    print(f"无效样本: {stats['invalid_samples']}")
    if issues:
        print(summarize_validation_failures(issues, max_items=10))
    else:
        print("✅ 数据验证通过")
    return stats


def generate_synthetic_next_board_dataset(
    num_samples: int,
    seed: int = 42,
    include_metadata: bool = True,
    max_rollout_steps: int = 200,
    min_steps_before_collect: int = 0,
    strict_validation: bool = True,
) -> Dataset:
    """Generate synthetic next-board supervision from random 2048 rollouts."""
    rng = random.Random(seed)
    data_samples: List[Dict] = []
    errors: List[str] = []
    game_index = 0

    while len(data_samples) < int(num_samples):
        game = Game2048(seed=rng.randint(0, 10**9))
        rollout_steps = rng.randint(max(0, min_steps_before_collect), max(min_steps_before_collect, max_rollout_steps))

        for step_idx in range(rollout_steps + 1):
            board = game.grid.astype(int).tolist()
            valid_actions = get_valid_action_ids(board)
            if not valid_actions:
                break

            if step_idx >= int(min_steps_before_collect):
                try:
                    action_id, next_board = choose_reasonable_action(board, rng=rng)
                    action_name = ACTION_MAP_ENGLISH[action_id]
                    sample: Dict = build_messages(
                        state_text=format_board_text(board),
                        action_name=action_name,
                        next_board=next_board,
                    )
                    if include_metadata:
                        sample.update(
                            {
                                "schema_version": OUTPUT_SCHEMA_VERSION,
                                "task_type": "next_board_prediction",
                                "source_type": "synthetic_rollout",
                                "source_game_index": game_index,
                                "source_step": step_idx,
                                "action": action_name,
                                "action_id": action_id,
                                "num_valid_actions": len(valid_actions),
                                "board_sum": int(np.sum(np.array(board, dtype=int))),
                                "max_tile": int(np.max(np.array(board, dtype=int))),
                            }
                        )

                    is_valid, error = validate_next_board_sample(sample)
                    if not is_valid:
                        raise ValueError(error)
                    data_samples.append(sample)
                except Exception as exc:
                    message = f"synthetic game={game_index} step={step_idx}: {exc}"
                    if strict_validation:
                        raise ValueError(message) from exc
                    errors.append(message)

                if len(data_samples) >= int(num_samples):
                    break

            valid_for_rollout = game.get_valid_actions()
            if not valid_for_rollout:
                break
            rollout_action = rng.choice(valid_for_rollout)
            game.step(int(rollout_action))
            if game.game_over:
                break

        game_index += 1

    if errors:
        print(summarize_validation_failures(errors, max_items=10))

    print(f"Created {len(data_samples)} synthetic next-board samples")
    return Dataset.from_list(data_samples)


def games_to_next_board_training_format(
    games_dir: str = "",
    min_score: int = 0,
    sample_rate: float = 1.0,
    include_metadata: bool = True,
    strict_validation: bool = True,
    max_games: Optional[int] = None,
    max_samples: Optional[int] = None,
    seed: int = 42,
    max_rollout_steps: int = 200,
    min_steps_before_collect: int = 0,
) -> Dataset:
    """Backward-compatible wrapper that now generates synthetic samples."""
    _ = games_dir
    _ = min_score
    _ = sample_rate
    _ = max_games
    target_samples = int(max_samples) if max_samples is not None else 1000
    return generate_synthetic_next_board_dataset(
        num_samples=target_samples,
        seed=seed,
        include_metadata=include_metadata,
        max_rollout_steps=max_rollout_steps,
        min_steps_before_collect=min_steps_before_collect,
        strict_validation=strict_validation,
    )


def analyze_dataset(dataset: Dataset, name: str = "Dataset") -> None:
    """Print sample previews and action distribution."""
    print(f"\n=== {name} ===")
    num_examples = min(3, len(dataset))
    print(f"\n前 {num_examples} 个样本:")

    for i in range(num_examples):
        sample = dataset[i]
        print(f"\n--- 样本 {i + 1} ---")
        preview = {
            "prompt": sample.get("prompt", []),
            "completion": sample.get("completion", []),
            "action": sample.get("action"),
        }
        print(str(preview)[:700] + "...")

    action_counts: Dict[str, int] = {}
    for item in dataset:
        action = str(item.get("action", ""))
        if action:
            action_counts[action] = action_counts.get(action, 0) + 1

    print("\n动作分布:")
    for action_id in range(4):
        action_name = ACTION_MAP[action_id]
        count = action_counts.get(action_name, 0)
        pct = 100 * count / len(dataset) if len(dataset) else 0.0
        print(f"  {action_name}: {count:,} ({pct:.1f}%)")


def save_processing_manifest(
    *,
    output_dir: str,
    args: argparse.Namespace,
    total_samples: int,
    train_size: int,
    val_size: int,
    test_size: int,
    manifest_file: Optional[str] = None,
) -> Path:
    """Save manifest for synthetic next-board datasets."""
    output_path = Path(output_dir)

    manifest = {
        "manifest_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "synthetic_next_board_dataset",
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "output_dir": output_dir,
        "generator": "random_rollout",
        "config": {
            "num_samples": args.num_samples,
            "max_rollout_steps": args.max_rollout_steps,
            "min_steps_before_collect": args.min_steps_before_collect,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "seed": args.seed,
            "strict_validation": not args.non_strict,
            "include_metadata": not args.no_metadata,
        },
        "sizes": {
            "total": total_samples,
            "train": train_size,
            "val": val_size,
            "test": test_size,
        },
    }

    target = Path(manifest_file) if manifest_file else output_path / "manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Manifest saved to {target}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="生成2048下一棋盘预测数据")
    parser.add_argument("--output_dir", type=str, default="data/processed_next_board")
    parser.add_argument("--num_samples", type=int, default=100000)
    parser.add_argument("--max_rollout_steps", type=int, default=200)
    parser.add_argument("--min_steps_before_collect", type=int, default=0)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--val_ratio", type=float, default=0.08)
    parser.add_argument("--test_ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--non_strict", action="store_true", help="遇到脏数据跳过而不是报错")
    parser.add_argument("--no_metadata", action="store_true", help="不写入生成元数据")
    parser.add_argument("--manifest_file", type=str, default=None)
    args = parser.parse_args()

    print("\n配置:")
    print(f"  输出目录: {args.output_dir}")
    print(f"  样本数: {args.num_samples}")
    print(f"  rollout最大步数: {args.max_rollout_steps}")
    print(f"  最早采样步数: {args.min_steps_before_collect}")
    print(f"  严格校验: {not args.non_strict}")
    print(f"  包含元数据: {not args.no_metadata}")

    dataset = generate_synthetic_next_board_dataset(
        num_samples=args.num_samples,
        seed=args.seed,
        include_metadata=not args.no_metadata,
        max_rollout_steps=args.max_rollout_steps,
        min_steps_before_collect=args.min_steps_before_collect,
        strict_validation=not args.non_strict,
    )

    train, val, test = split_dataset(
        dataset,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    if args.validate:
        validate_dataset(train, "训练集")
        validate_dataset(val, "验证集")
        validate_dataset(test, "测试集")

    if args.analyze:
        analyze_dataset(train, "训练集")

    save_datasets(train, val, test, args.output_dir)
    save_processing_manifest(
        output_dir=args.output_dir,
        args=args,
        total_samples=len(dataset),
        train_size=len(train),
        val_size=len(val),
        test_size=len(test),
        manifest_file=args.manifest_file,
    )
    print("\n完成!")


if __name__ == "__main__":
    main()

# python -m src.data_gen.next_board_processor \
#     --output_dir data/processed_next_board \
#     --num_samples 1000 \
#     --validate

#   几个主要参数是：

#   - --num_samples：要生成多少条样本
#   - --max_rollout_steps：每局随机 rollout 最多走多少步，控制棋盘复杂度
#   - --min_steps_before_collect：至少走到第几步后才开始采样，避免太多特别简单的早期棋盘

#   我已经跑过一次小规模 smoke：

#   python -m src.data_gen.next_board_processor \
#     --output_dir /tmp/processed_next_board_synth \
#     --num_samples 16 \
#     --max_rollout_steps 20 \
#     --validate --analyze

#   可以正常生成、校验、切分并落盘。
