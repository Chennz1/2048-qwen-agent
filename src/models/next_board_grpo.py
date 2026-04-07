"""TRL-native GRPO trainer for the 2048 next-board prediction task."""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import os
import re
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer

from src.data_gen.next_board_processor import simulate_next_board
from src.data_gen.next_board_sft_processor import SHORT_THINKING_SYSTEM_PROMPT
from src.models.grpo import (
    _THINK_BLOCK_RE,
    _THINK_CLOSE_RE,
    _build_grpo_config,
    build_grpo_generation_kwargs,
    _load_trl_grpo_symbols,
    _normalize_completion_text,
    _patch_grpo_trainer_sampler_compat,
    _resolve_grpo_model_input,
    _validate_batch_generation_compatibility,
    resolve_save_steps,
)
from src.utils.action_stats import ActionWindowStats
from src.utils.monitoring import normalize_monitor_backend, report_to_list


os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


_DIRECTION_TO_ID = {
    "UP": 0,
    "RIGHT": 1,
    "DOWN": 2,
    "LEFT": 3,
}
_FENCED_JSON_RE = re.compile(
    r"^\s*```(?:json)?\s*([\s\S]*?)\s*```\s*$",
    flags=re.I,
)


@dataclass
class NextBoardRewardConfig:
    json_invalid_penalty: float = -20.0
    top_level_ok_bonus: float = 1.0
    top_level_bad_penalty: float = -2.0
    shape_ok_bonus: float = 1.0
    shape_bad_penalty: float = -3.0
    value_type_ok_bonus: float = 1.0
    value_type_bad_penalty: float = -2.0
    exact_cell_bonus: float = 0.25
    wrong_cell_penalty: float = -0.10
    exact_row_bonus: float = 0.5
    exact_board_bonus: float = 6.0
    compact_exact_bonus_max: float = 1.0
    compact_extra_char_penalty: float = 0.01
    consistent_with_env_bonus: float = 2.0
    inconsistent_with_env_penalty: float = -2.0
    cot_length_threshold: int = 80
    cot_length_bonus: float = 0.3


class NextBoardGRPODatasetBuilder:
    """Build prompt-only GRPO dataset from processed next-board datasets."""

    def __init__(self, model_name: str, use_thinking: bool = True):
        self.model_name = model_name
        self.use_thinking = use_thinking
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if getattr(self.tokenizer, "padding_side", None) != "left":
            self.tokenizer.padding_side = "left"

    def build(self, data_dir: str, num_samples: int, seed: int = 42) -> Dataset:
        ds = load_from_disk(data_dir)
        rows: List[Dict[str, Any]] = []
        for item in ds:
            prompt_messages = item.get("prompt")
            completion_messages = item.get("completion")
            if not isinstance(prompt_messages, list) or len(prompt_messages) != 2:
                continue
            if not isinstance(completion_messages, list) or len(completion_messages) != 1:
                continue

            user_text = str(prompt_messages[1].get("content", ""))
            board = _extract_board_from_user_prompt(user_text)
            action_id = _extract_action_id_from_item(item, user_text)
            target_next_board = _extract_target_next_board(completion_messages[0].get("content", ""))
            if board is None or action_id is None or target_next_board is None:
                continue

            aligned_prompt_messages = [
                {"role": "system", "content": SHORT_THINKING_SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ]
            prompt = self.tokenizer.apply_chat_template(
                aligned_prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=bool(self.use_thinking),
            )
            rows.append(
                {
                    "prompt": prompt,
                    "current_board": board,
                    "action_id": int(action_id),
                    "target_next_board": target_next_board,
                }
            )

        if not rows:
            raise ValueError(f"No valid next-board samples found in {data_dir}")

        rng = np.random.default_rng(seed)
        if len(rows) > int(num_samples):
            indices = rng.choice(len(rows), size=int(num_samples), replace=False)
            rows = [rows[int(i)] for i in indices]
        return Dataset.from_list(rows)


class NextBoardGRPORewards:
    _cfg: NextBoardRewardConfig = NextBoardRewardConfig()
    _window_stats: Optional[ActionWindowStats] = None

    @classmethod
    def configure(cls, cfg: NextBoardRewardConfig) -> None:
        cls._cfg = cfg

    @classmethod
    def attach_window_stats(cls, stats: Optional[ActionWindowStats]) -> None:
        cls._window_stats = stats

    @classmethod
    def _record_window_stats(
        cls,
        *,
        parsed: bool,
        legal: bool,
        correct: bool,
        reward_format: float = 0.0,
        reward_legal: float = 0.0,
        reward_facts: float = 0.0,
        reward_expert: float = 0.0,
        reward_all_correct: float = 0.0,
    ) -> None:
        if cls._window_stats is None:
            return
        cls._window_stats.update(
            parsed=parsed,
            legal=legal,
            correct=correct,
            reward_format=reward_format,
            reward_legal=reward_legal,
            reward_facts=reward_facts,
            reward_expert=reward_expert,
            reward_all_correct=reward_all_correct,
        )

    @classmethod
    def _extract_non_think_region(cls, text: str) -> str:
        normalized = _normalize_completion_text(text)
        closes = list(_THINK_CLOSE_RE.finditer(normalized))
        if closes:
            return normalized[closes[-1].end():].strip()
        return normalized.strip()

    @classmethod
    def _extract_think_region(cls, text: str) -> str:
        normalized = _normalize_completion_text(text)
        match = _THINK_BLOCK_RE.search(normalized)
        if match is None:
            return ""
        return str(match.group(1) or "").strip()

    @classmethod
    def _unwrap_json_candidate(cls, text: str) -> str:
        stripped = str(text or "").strip()
        if not stripped:
            return ""
        fenced = _FENCED_JSON_RE.match(stripped)
        if fenced is not None:
            return str(fenced.group(1) or "").strip()
        return stripped

    @classmethod
    def _compact_json_text(cls, next_board: List[List[int]]) -> str:
        return json.dumps({"next_board": next_board}, ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def _parse_predicted_next_board(
        cls,
        text: str,
    ) -> Tuple[Optional[List[List[int]]], float, str]:
        format_score = 0.0
        candidate = cls._unwrap_json_candidate(cls._extract_non_think_region(text))
        if not candidate:
            return None, float(cls._cfg.json_invalid_penalty), ""

        try:
            payload = json.loads(candidate)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None, float(cls._cfg.json_invalid_penalty), candidate

        if set(payload.keys()) == {"next_board"}:
            format_score += float(cls._cfg.top_level_ok_bonus)
        else:
            return None, float(cls._cfg.json_invalid_penalty + cls._cfg.top_level_bad_penalty), candidate

        board = payload.get("next_board")
        if not isinstance(board, list) or len(board) != 4:
            return None, float(format_score + cls._cfg.shape_bad_penalty), candidate
        format_score += float(cls._cfg.shape_ok_bonus)

        typed_rows: List[List[int]] = []
        for row in board:
            if not isinstance(row, list) or len(row) != 4:
                return None, float(format_score + cls._cfg.shape_bad_penalty), candidate
            typed_row: List[int] = []
            for value in row:
                if not isinstance(value, int) or isinstance(value, bool):
                    return None, float(format_score + cls._cfg.value_type_bad_penalty), candidate
                typed_row.append(int(value))
            typed_rows.append(typed_row)
        format_score += float(cls._cfg.value_type_ok_bonus)
        return typed_rows, float(format_score), candidate

    @classmethod
    def next_board_accuracy(cls, completions: List[Any], **kwargs) -> List[float]:
        rewards: List[float] = []
        boards = kwargs.get("current_board")
        action_ids = kwargs.get("action_id")
        targets = kwargs.get("target_next_board")

        for idx, completion in enumerate(completions):
            text = _completion_to_text(completion)
            think_text = cls._extract_think_region(text)
            predicted_board, format_score, parsed_json_text = cls._parse_predicted_next_board(text)
            if len(think_text) > int(cls._cfg.cot_length_threshold):
                format_score += float(cls._cfg.cot_length_bonus)

            current_board = _coerce_board(boards[idx] if boards else None)
            action_id = _coerce_action_id(action_ids[idx] if action_ids else None)
            target_board = _coerce_board(targets[idx] if targets else None)

            parsed = predicted_board is not None
            legal = False
            correct = False
            legal_score = 0.0
            facts_score = 0.0
            all_correct_score = 0.0

            if predicted_board is not None and current_board is not None and action_id is not None:
                env_next_board = simulate_next_board(current_board, action_id)
                if predicted_board == env_next_board:
                    legal = True
                    legal_score += float(cls._cfg.consistent_with_env_bonus)
                else:
                    legal_score += float(cls._cfg.inconsistent_with_env_penalty)

            if predicted_board is not None and target_board is not None:
                exact_rows = 0
                exact_cells = 0
                for row_idx in range(4):
                    if predicted_board[row_idx] == target_board[row_idx]:
                        exact_rows += 1
                        facts_score += float(cls._cfg.exact_row_bonus)
                    for col_idx in range(4):
                        if predicted_board[row_idx][col_idx] == target_board[row_idx][col_idx]:
                            exact_cells += 1
                            facts_score += float(cls._cfg.exact_cell_bonus)
                        else:
                            facts_score += float(cls._cfg.wrong_cell_penalty)
                _ = exact_cells
                if predicted_board == target_board:
                    correct = True
                    facts_score += float(cls._cfg.exact_board_bonus)
                    compact_target = cls._compact_json_text(target_board)
                    extra_chars = max(0, len(parsed_json_text) - len(compact_target))
                    facts_score += max(
                        0.0,
                        float(cls._cfg.compact_exact_bonus_max)
                        - float(cls._cfg.compact_extra_char_penalty) * float(extra_chars),
                    )
                if correct and legal:
                    all_correct_score += 1.0
            total_score = float(format_score + legal_score + facts_score + all_correct_score)
            rewards.append(total_score)
            cls._record_window_stats(
                parsed=parsed,
                legal=legal,
                correct=correct,
                reward_format=format_score,
                reward_legal=legal_score,
                reward_facts=facts_score,
                reward_expert=0.0,
                reward_all_correct=all_correct_score,
            )

        return rewards

    @classmethod
    def get_reward_funcs(cls) -> List[Any]:
        return [cls.next_board_accuracy]


class TRLGRPONextBoardTrainer:
    """High-level trainer using TRL GRPOTrainer for next-board prediction."""

    def __init__(
        self,
        model_name_or_path: str,
        output_dir: str = "./checkpoints/grpo_next_board",
        use_wandb: bool = False,
        monitor_backend: str = "none",
    ):
        self.model_name_or_path = model_name_or_path
        self.output_dir = output_dir
        self.monitor_backend = normalize_monitor_backend(
            monitor_backend,
            use_wandb=use_wandb,
        )

    def train(
        self,
        dataset: Dataset,
        tokenizer: Optional[Any] = None,
        learning_rate: float = 1e-6,
        warmup_ratio: float = 0.03,
        lr_scheduler_type: str = "cosine",
        clip_eps: float = 0.28,
        kl_beta: float = 0.0,
        num_train_epochs: int = 1,
        batch_size: int = 2,
        gradient_accumulation_steps: int = 4,
        num_generations: int = 2,
        max_prompt_length: int = 512,
        max_completion_length: int = 256,
        save_steps: float = 200,
        logging_steps: int = 4,
        disable_tqdm: bool = False,
    ) -> Any:
        GRPOConfig, GRPOTrainer = _load_trl_grpo_symbols()
        report_to = report_to_list(self.monitor_backend)
        generation_kwargs = build_grpo_generation_kwargs(tokenizer)
        resolved_save_steps = resolve_save_steps(
            save_steps=save_steps,
            dataset_size=len(dataset),
            num_train_epochs=num_train_epochs,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        cfg = _build_grpo_config(
            GRPOConfig=GRPOConfig,
            output_dir=self.output_dir,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type=lr_scheduler_type,
            num_train_epochs=num_train_epochs,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_generations=num_generations,
            max_prompt_length=max_prompt_length,
            max_completion_length=max_completion_length,
            save_steps=resolved_save_steps,
            logging_steps=logging_steps,
            disable_tqdm=disable_tqdm,
            generation_kwargs=generation_kwargs,
            report_to=report_to,
            clip_eps=clip_eps,
            kl_beta=kl_beta,
        )

        model_input = _resolve_grpo_model_input(self.model_name_or_path)
        reward_funcs = NextBoardGRPORewards.get_reward_funcs()
        trainer_params = set(inspect.signature(GRPOTrainer.__init__).parameters.keys())
        trainer_kwargs: Dict[str, Any] = {
            "model": model_input,
            "reward_funcs": reward_funcs,
            "args": cfg,
            "train_dataset": dataset,
        }
        if tokenizer is not None:
            if "processing_class" in trainer_params:
                trainer_kwargs["processing_class"] = tokenizer
            elif "tokenizer" in trainer_params:
                trainer_kwargs["tokenizer"] = tokenizer

        trainer = GRPOTrainer(**trainer_kwargs)
        _patch_grpo_trainer_sampler_compat(trainer)

        stats = ActionWindowStats()
        NextBoardGRPORewards.attach_window_stats(stats)
        _patch_next_board_logging(trainer, stats)

        try:
            trainer.train()
            tail_metrics = stats.flush()
            if tail_metrics:
                trainer.log(tail_metrics)
        finally:
            NextBoardGRPORewards.attach_window_stats(None)

        trainer.save_model(self.output_dir)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(self.output_dir) / "training_meta.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "trainer": "trl_grpo_next_board",
                    "model_name_or_path": self.model_name_or_path,
                    "num_train_samples": len(dataset),
                    "reward_mode": "next_board_accuracy",
                    "reward_config": vars(NextBoardGRPORewards._cfg),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        return trainer


def _patch_next_board_logging(trainer: Any, stats: ActionWindowStats) -> None:
    original_log = trainer.log

    def _wrapped(self, logs, *args, **kwargs):
        merged = dict(logs) if isinstance(logs, dict) else {}
        merged.update(stats.flush())
        return original_log(merged, *args, **kwargs)

    trainer.log = types.MethodType(_wrapped, trainer)


def _completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict) and isinstance(completion.get("content"), str):
        return completion["content"]
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict) and isinstance(last.get("content"), str):
            return last["content"]
    return str(completion)


def _coerce_action_id(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        value_int = int(value)
        return value_int if 0 <= value_int <= 3 else None
    if isinstance(value, str):
        value_norm = value.strip().upper()
        if value_norm in _DIRECTION_TO_ID:
            return int(_DIRECTION_TO_ID[value_norm])
        if value_norm in {"0", "1", "2", "3"}:
            return int(value_norm)
    return None


def _coerce_board(value: Any) -> Optional[List[List[int]]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            try:
                value = ast.literal_eval(value)
            except Exception:
                return None
    if not isinstance(value, list) or len(value) != 4:
        return None
    board: List[List[int]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 4:
            return None
        typed_row: List[int] = []
        for cell in row:
            if not isinstance(cell, int) or isinstance(cell, bool):
                return None
            typed_row.append(int(cell))
        board.append(typed_row)
    return board


def _extract_board_from_user_prompt(user_text: str) -> Optional[List[List[int]]]:
    if "Current board:\n" not in user_text or "\n\nMove direction:" not in user_text:
        return None
    board_text = user_text.split("Current board:\n", 1)[1].split("\n\nMove direction:", 1)[0]
    return _coerce_board(board_text)


def _extract_action_id_from_item(item: Dict[str, Any], user_text: str) -> Optional[int]:
    action_id = _coerce_action_id(item.get("action_id"))
    if action_id is not None:
        return action_id
    if "Move direction:" not in user_text or "\n\nCompute the next board exactly." not in user_text:
        return None
    action_text = user_text.split("Move direction:", 1)[1].split("\n\nCompute the next board exactly.", 1)[0]
    return _coerce_action_id(action_text.strip())


def _extract_target_next_board(content: Any) -> Optional[List[List[int]]]:
    if not isinstance(content, str):
        return None
    try:
        payload = json.loads(content)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return _coerce_board(payload.get("next_board"))


def main() -> None:
    parser = argparse.ArgumentParser(description="TRL GRPO training for 2048 next-board prediction")
    parser.add_argument("--model", type=str, default="./checkpoints/sft_next_board")
    parser.add_argument("--base_model", type=str, default=None, help="Alias of --model")
    parser.add_argument("--input_dir", type=str, default="data/processed_next_board/train")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/grpo_next_board")
    parser.add_argument("--num_samples", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--num_generations", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--clip_eps", type=float, default=0.28)
    parser.add_argument("--kl_beta", type=float, default=0.0)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_completion_length", type=int, default=256)
    parser.add_argument("--save_steps", type=float, default=500)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--disable_tqdm", action="store_true")
    parser.add_argument("--reward_json_invalid_penalty", type=float, default=-20.0)
    parser.add_argument("--reward_exact_board_bonus", type=float, default=6.0)
    parser.add_argument("--reward_compact_exact_bonus_max", type=float, default=1.0)
    parser.add_argument("--reward_compact_extra_char_penalty", type=float, default=0.01)
    parser.add_argument("--reward_consistent_with_env_bonus", type=float, default=2.0)
    parser.add_argument("--reward_inconsistent_with_env_penalty", type=float, default=-2.0)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument(
        "--monitor_backend",
        type=str,
        default=None,
        choices=["wandb", "tensorboard", "none"],
    )
    parser.add_argument("--no_thinking", action="store_true")
    args = parser.parse_args()

    model_name_or_path = args.base_model or args.model
    monitor_backend = normalize_monitor_backend(
        args.monitor_backend,
        no_wandb=args.no_wandb,
    )
    _validate_batch_generation_compatibility(
        batch_size=args.batch_size,
        num_generations=args.num_generations,
    )

    builder = NextBoardGRPODatasetBuilder(
        model_name=model_name_or_path,
        use_thinking=not args.no_thinking,
    )
    train_dataset = builder.build(
        data_dir=args.input_dir,
        num_samples=args.num_samples,
        seed=args.seed,
    )

    NextBoardGRPORewards.configure(
        NextBoardRewardConfig(
            json_invalid_penalty=args.reward_json_invalid_penalty,
            exact_board_bonus=args.reward_exact_board_bonus,
            compact_exact_bonus_max=args.reward_compact_exact_bonus_max,
            compact_extra_char_penalty=args.reward_compact_extra_char_penalty,
            consistent_with_env_bonus=args.reward_consistent_with_env_bonus,
            inconsistent_with_env_penalty=args.reward_inconsistent_with_env_penalty,
        )
    )

    trainer = TRLGRPONextBoardTrainer(
        model_name_or_path=model_name_or_path,
        output_dir=args.output_dir,
        use_wandb=not args.no_wandb,
        monitor_backend=monitor_backend,
    )
    trainer.train(
        dataset=train_dataset,
        tokenizer=builder.tokenizer,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        clip_eps=args.clip_eps,
        kl_beta=args.kl_beta,
        num_train_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        disable_tqdm=args.disable_tqdm,
    )


if __name__ == "__main__":
    main()
