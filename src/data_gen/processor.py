"""Data processor: raw trajectory JSON -> processed SFT dataset."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datasets import Dataset

# Hugging Face mirror defaults (honor existing env if user already set it).
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

from src.data_gen.contracts import (
    SCHEMA_VERSION,
    summarize_validation_failures,
    validate_processed_sample,
    validate_raw_game,
)
from src.data_gen.prompting import build_messages
from src.envs.game_2048 import ACTION_MAP_ENGLISH, parse_action_from_non_think_text



def validate_game_data(game: Dict) -> Tuple[bool, Optional[str]]:
    """Backward-compatible validator wrapper for one raw game."""
    result = validate_raw_game(game)
    return result.is_valid, result.error


def _list_raw_game_files(games_path: Path) -> List[Path]:
    """List raw game files, skipping manifest/metadata JSON."""
    preferred = sorted(games_path.glob("game_*.json"))
    if preferred:
        return preferred

    all_json = sorted(games_path.glob("*.json"))
    skip_names = {"manifest.json", "metadata.json", "summary.json", "stats.json"}
    return [p for p in all_json if p.name.lower() not in skip_names]



def validate_dataset(dataset: Dataset, name: str = "Dataset", require_thinking: bool = False) -> Dict:
    """Validate processed training dataset quality."""
    print(f"\n=== 验证 {name} ===")

    issues: List[str] = []
    valid_count = 0

    for i, sample in enumerate(dataset):
        result = validate_processed_sample(sample, require_thinking=require_thinking)
        if result.is_valid:
            valid_count += 1
        else:
            issues.append(f"样本 {i}: {result.error}")

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


def _sample_response_text(sample: Dict) -> str:
    """Extract assistant response text from processed sample."""
    completion = sample.get("completion")
    if isinstance(completion, list):
        assistant_msgs = [
            m for m in completion
            if isinstance(m, dict) and m.get("role") == "assistant" and isinstance(m.get("content"), str)
        ]
        if assistant_msgs:
            return assistant_msgs[-1]["content"]
    return ""



def games_to_training_format(
    games_dir: str = "data/raw",
    min_score: int = 0,
    sample_rate: float = 1.0,
    use_thinking: bool = False,
    apply_chat_template: bool = True,
    base_model: str = "Qwen/Qwen3-1.7B",
    strict_validation: bool = True,
    include_metadata: bool = True,
) -> Dataset:
    """
    Convert raw game trajectories to SFT dataset.

    Output contract:
    - required: prompt, completion (conversational format)
    - optional: schema_version, source_game_id, source_step
    """
    games_path = Path(games_dir)
    json_files = _list_raw_game_files(games_path)
    print(f"Found {len(json_files)} game files")

    _ = apply_chat_template
    _ = base_model

    data_samples: List[Dict] = []
    errors: List[str] = []

    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            game = json.load(f)

        # Support old files by filling missing schema_version before validation.
        if "schema_version" not in game:
            game["schema_version"] = SCHEMA_VERSION

        is_valid, error = validate_game_data(game)
        if not is_valid:
            message = f"{json_file.name}: {error}"
            if strict_validation:
                raise ValueError(message)
            errors.append(message)
            continue

        if game["final_score"] < min_score:
            continue

        for step_data in game["states"]:
            action_json_text: Optional[str] = None
            if "action_json" in step_data:
                action_json_value = step_data.get("action_json")
                if isinstance(action_json_value, dict):
                    action_json_text = json.dumps(action_json_value, ensure_ascii=False)
                elif isinstance(action_json_value, str):
                    action_json_text = action_json_value.strip()
                else:
                    message = (
                        f"Invalid action_json type from {json_file.name}, "
                        f"step={step_data.get('step')}: {type(action_json_value).__name__}"
                    )
                    if strict_validation:
                        raise ValueError(message)
                    errors.append(message)
                    continue

            messages = build_messages(
                state_text=step_data["state"],
                action=step_data["action"],
                use_thinking=use_thinking,
                thinking=step_data.get("thinking"),
                action_json_text=action_json_text,
            )

            sample: Dict = {
                "prompt": [messages[0]],
                "completion": [messages[1]],
            }
            if include_metadata:
                sample.update(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "source_game_id": game["game_id"],
                        "source_step": step_data["step"],
                    }
                )

            validation = validate_processed_sample(sample, require_thinking=use_thinking)
            if not validation.is_valid:
                message = (
                    f"Invalid processed sample from {json_file.name}, "
                    f"step={step_data['step']}: {validation.error}"
                )
                if strict_validation:
                    raise ValueError(message)
                errors.append(message)
                continue

            data_samples.append(sample)

    if errors:
        print(summarize_validation_failures(errors, max_items=10))

    if sample_rate < 1.0:
        n_samples = int(len(data_samples) * sample_rate)
        data_samples = data_samples[:n_samples]

    print(f"Created {len(data_samples)} training samples")
    return Dataset.from_list(data_samples)



def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.98,
    val_ratio: float = 0.01,
    test_ratio: float = 0.01,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset]:
    """Split dataset into train/val/test sets."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    total = len(dataset)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)

    shuffled = dataset.shuffle(seed=seed)

    train = shuffled.select(range(train_size))
    val = shuffled.select(range(train_size, train_size + val_size))
    test = shuffled.select(range(train_size + val_size, total))

    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return train, val, test



def save_datasets(train: Dataset, val: Dataset, test: Dataset, output_dir: str = "data/processed") -> None:
    """Save dataset splits to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train.save_to_disk(str(output_path / "train"))
    val.save_to_disk(str(output_path / "val"))
    test.save_to_disk(str(output_path / "test"))

    print(f"Datasets saved to {output_dir}")


def save_processing_manifest(
    *,
    output_dir: str,
    input_dir: str,
    args: argparse.Namespace,
    total_samples: int,
    train_size: int,
    val_size: int,
    test_size: int,
    manifest_file: Optional[str] = None,
) -> Path:
    """Save processed dataset manifest for versioning and reproducibility."""
    output_path = Path(output_dir)
    input_manifest = Path(input_dir) / "manifest.json"

    manifest = {
        "manifest_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "processed_dataset",
        "schema_version": SCHEMA_VERSION,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "input_manifest": str(input_manifest) if input_manifest.exists() else None,
        "config": {
            "min_score": args.min_score,
            "sample_rate": args.sample_rate,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "seed": args.seed,
            "use_thinking": args.use_thinking,
            "apply_chat_template": not args.no_chat_template,
            "base_model": args.base_model,
            "strict_validation": not args.non_strict,
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
        }
        print(str(preview)[:500] + "...")

    actions = []
    for item in dataset:
        text = _sample_response_text(item).rstrip()
        action_id = parse_action_from_non_think_text(text)
        if action_id is not None:
            actions.append(ACTION_MAP_ENGLISH[action_id])

    unique_actions = sorted(set(actions))
    print(f"\n唯一动作: {len(unique_actions)}")
    print("动作分布:")
    for action in unique_actions:
        count = sum(1 for a in actions if a == action)
        pct = 100 * count / len(actions) if actions else 0
        print(f"  {action}: {count:,} ({pct:.1f}%)")



def main() -> None:
    parser = argparse.ArgumentParser(description="处理2048游戏数据（统一数据契约）")
    parser.add_argument("--input_dir", type=str, default="data/raw")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--min_score", type=int, default=0)
    parser.add_argument("--sample_rate", type=float, default=1.0)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--val_ratio", type=float, default=0.08)
    parser.add_argument("--test_ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_thinking", action="store_true", help="使用Chain-of-Thought格式")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-1.7B", help="用于chat template")
    parser.add_argument("--no_chat_template", action="store_true", help="不应用chat template（不推荐）")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--non_strict", action="store_true", help="遇到脏数据跳过而不是报错")
    parser.add_argument("--manifest_file", type=str, default=None, help="可选：保存manifest到指定路径")

    args = parser.parse_args()

    print("\n配置:")
    print(f"  输入目录: {args.input_dir}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  使用CoT: {args.use_thinking}")
    print(f"  应用Chat Template: {not args.no_chat_template}")
    print(f"  基座模型: {args.base_model}")
    print(f"  严格校验: {not args.non_strict}")

    dataset = games_to_training_format(
        games_dir=args.input_dir,
        min_score=args.min_score,
        sample_rate=args.sample_rate,
        use_thinking=args.use_thinking,
        apply_chat_template=not args.no_chat_template,
        base_model=args.base_model,
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
        validate_dataset(train, "训练集", require_thinking=args.use_thinking)
        validate_dataset(val, "验证集", require_thinking=args.use_thinking)
        validate_dataset(test, "测试集", require_thinking=args.use_thinking)

    if args.analyze:
        analyze_dataset(train, "训练集")

    save_datasets(train, val, test, args.output_dir)
    save_processing_manifest(
        output_dir=args.output_dir,
        input_dir=args.input_dir,
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
