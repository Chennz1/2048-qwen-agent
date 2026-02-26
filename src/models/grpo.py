"""TRL-native GRPO trainer for 2048.

This module replaces the legacy hand-written GRPO loop with HuggingFace TRL
`GRPOTrainer` workflow:
- prompt-only dataset
- reward function callbacks
- standardized trainer config/save/log interface
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import inspect
import types
from importlib.metadata import PackageNotFoundError, version
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from datasets import Dataset

# Hugging Face mirror defaults (honor existing env if user already set it).
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

from transformers import AutoTokenizer

from src.data.prompting import format_inference_prompt
from src.envs.game_2048 import ACTION_MAP, Game2048, parse_action_from_text
from src.utils.monitoring import normalize_monitor_backend, report_to_list


def _load_trl_grpo_symbols():
    """Load TRL GRPO symbols with compatibility fallbacks."""
    import_errors: List[Exception] = []

    try:
        from trl import GRPOConfig, GRPOTrainer

        return GRPOConfig, GRPOTrainer
    except Exception as exc:
        import_errors.append(exc)

    try:
        from trl.trainer.grpo_config import GRPOConfig
        from trl.trainer.grpo_trainer import GRPOTrainer

        return GRPOConfig, GRPOTrainer
    except Exception as exc:
        import_errors.append(exc)

    try:
        trl_version = version("trl")
    except (PackageNotFoundError, Exception):
        trl_version = "not_installed"

    error_text = " | ".join(f"{type(e).__name__}: {e}" for e in import_errors)
    raise RuntimeError(
        "TRL GRPOTrainer is unavailable in current environment. "
        f"Detected trl={trl_version}. "
        "Please install/upgrade to a GRPO-capable version, e.g. "
        "`pip install -U \"trl>=0.15.0\"`. "
        f"Import errors: {error_text}"
    )


def _patch_grpo_trainer_sampler_compat(trainer: Any) -> None:
    """Patch TRL/Transformers sampler signature mismatch at runtime.

    Some version pairs call `_get_train_sampler(dataset)` while older GRPOTrainer
    only implements `_get_train_sampler(self)`. We patch instance methods to accept
    the newer call form and delegate to the original implementation.
    """
    patched: List[str] = []
    for name in ("_get_train_sampler", "_get_eval_sampler"):
        method = getattr(trainer, name, None)
        if method is None:
            continue
        try:
            param_count = len(inspect.signature(method).parameters)
        except (TypeError, ValueError):
            continue

        # Bound-method signature excludes `self`.
        if param_count == 0:
            orig = method

            def _wrapped(self, dataset=None, _orig=orig):
                return _orig()

            setattr(trainer, name, types.MethodType(_wrapped, trainer))
            patched.append(name)

    if patched:
        print(
            "[GRPO] Applied sampler compatibility patch for methods: "
            + ", ".join(patched)
        )


def _build_grpo_config(
    *,
    GRPOConfig: Any,
    output_dir: str,
    learning_rate: float,
    num_train_epochs: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    num_generations: int,
    max_prompt_length: int,
    max_completion_length: int,
    save_steps: int,
    logging_steps: int,
    report_to: List[str],
):
    """Build GRPOConfig with runtime compatibility across TRL versions."""
    params = set(inspect.signature(GRPOConfig.__init__).parameters.keys())

    kwargs: Dict[str, Any] = {
        "output_dir": output_dir,
        "learning_rate": learning_rate,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_generations": num_generations,
        "max_prompt_length": max_prompt_length,
        "max_completion_length": max_completion_length,
        "save_steps": save_steps,
        "logging_steps": logging_steps,
        "report_to": report_to,
    }

    # Length args vary across TRL versions. Provide fallbacks when canonical
    # fields are absent so older/newer versions can still receive effective limits.
    if "max_prompt_length" not in params:
        if "max_seq_length" in params:
            kwargs["max_seq_length"] = max_prompt_length + max_completion_length
        elif "max_length" in params:
            kwargs["max_length"] = max_prompt_length + max_completion_length

    if "max_completion_length" not in params:
        if "max_new_tokens" in params:
            kwargs["max_new_tokens"] = max_completion_length
        elif "response_length" in params:
            kwargs["response_length"] = max_completion_length

    filtered = {k: v for k, v in kwargs.items() if k in params}
    return GRPOConfig(**filtered)


def _resolve_grpo_model_input(model_name_or_path: str) -> Any:
    """Resolve model input for GRPOTrainer across full-model and PEFT checkpoints."""
    model_path = Path(model_name_or_path)
    if not model_path.is_dir():
        return model_name_or_path

    has_full_config = (model_path / "config.json").exists()
    has_adapter_config = (model_path / "adapter_config.json").exists()
    if has_full_config or not has_adapter_config:
        return model_name_or_path

    try:
        from peft import AutoPeftModelForCausalLM
    except Exception as exc:
        raise RuntimeError(
            "Detected PEFT adapter checkpoint without `config.json`, but `peft` "
            "is unavailable. Install `peft` or pass a full model path."
        ) from exc

    print(
        "[GRPO] Detected adapter-only checkpoint; loading trainable PEFT model "
        f"from: {model_name_or_path}"
    )
    load_params = set(inspect.signature(AutoPeftModelForCausalLM.from_pretrained).parameters.keys())
    load_kwargs: Dict[str, Any] = {
        "torch_dtype": "auto",
        "device_map": "auto",
        "trust_remote_code": True,
        "is_trainable": True,
    }
    filtered_kwargs = {k: v for k, v in load_kwargs.items() if k in load_params}
    return AutoPeftModelForCausalLM.from_pretrained(model_name_or_path, **filtered_kwargs)


@dataclass
class GRPO2048DataConfig:
    """Prompt dataset config for GRPO."""

    source: str = "raw"  # raw | processed
    input_dir: str = "data/raw"
    num_samples: int = 5000
    min_score: int = 0
    seed: int = 42


@dataclass
class ExpertRewardConfig:
    """Config for expert-shaped GRPO reward."""

    reward_mode: str = "expert_shaped"  # simple | expert_shaped
    illegal_penalty: float = -5.0
    legal_bonus: float = 0.2
    score_norm: float = 64.0
    potential_norm: float = 5.0
    w_score: float = 0.15
    w_potential: float = 0.35
    w_expert: float = 0.40
    w_risk: float = 0.10
    expert_depth: int = 2
    expert_max_empty: int = 8


class GRPO2048DatasetBuilder:
    """Build prompt-only GRPO dataset with metadata for reward calculation."""

    def __init__(self, model_name: str, use_thinking: bool = True):
        self.model_name = model_name
        self.use_thinking = use_thinking
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def build(self, cfg: GRPO2048DataConfig) -> Dataset:
        if cfg.source == "raw":
            rows = self._from_raw(Path(cfg.input_dir), cfg.min_score)
        elif cfg.source == "processed":
            rows = self._from_processed(Path(cfg.input_dir))
        else:
            raise ValueError(f"Unsupported source: {cfg.source}")

        if not rows:
            raise ValueError("No samples available for GRPO dataset")

        rng = np.random.default_rng(cfg.seed)
        if len(rows) > cfg.num_samples:
            indices = rng.choice(len(rows), size=cfg.num_samples, replace=False)
            rows = [rows[i] for i in indices]

        return Dataset.from_list(rows)

    def _from_raw(self, input_dir: Path, min_score: int) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []

        for file in sorted(input_dir.glob("*.json")):
            with open(file, "r", encoding="utf-8") as f:
                game = json.load(f)

            if int(game.get("final_score", 0)) < min_score:
                continue

            for state_item in game.get("states", []):
                state_text = state_item.get("state")
                if not state_text:
                    continue

                prompt = format_inference_prompt(
                    tokenizer=self.tokenizer,
                    state_text=state_text,
                    use_thinking=self.use_thinking,
                )

                valid_actions = self._compute_valid_actions_from_state(state_text)
                rows.append(
                    {
                        "prompt": prompt,
                        "state_text": state_text,
                        "valid_actions": valid_actions,
                        "target_action": state_item.get("action", "上"),
                    }
                )

        return rows

    def _from_processed(self, input_dir: Path) -> List[Dict[str, Any]]:
        ds = Dataset.load_from_disk(str(input_dir))
        rows: List[Dict[str, Any]] = []

        for item in ds:
            text = item["text"]
            # Fallback path: already templated text; no state metadata => weak reward.
            rows.append(
                {
                    "prompt": text,
                    "state_text": "",
                    "valid_actions": [],
                    "target_action": "上",
                }
            )

        return rows

    def _compute_valid_actions_from_state(self, state_text: str) -> List[int]:
        game = _game_from_state_text(state_text)
        if game is None:
            return []
        return game.get_valid_actions()


class BoardPotentialModel:
    """Board-value heuristic used by shaped rewards and expert evaluator."""

    @staticmethod
    def _smoothness_penalty(grid: np.ndarray) -> float:
        penalty = 0.0
        for r in range(4):
            for c in range(4):
                if grid[r, c] <= 0:
                    continue
                v = float(np.log2(grid[r, c]))
                if r + 1 < 4 and grid[r + 1, c] > 0:
                    penalty += abs(v - float(np.log2(grid[r + 1, c])))
                if c + 1 < 4 and grid[r, c + 1] > 0:
                    penalty += abs(v - float(np.log2(grid[r, c + 1])))
        return penalty

    @staticmethod
    def _monotonicity(grid: np.ndarray) -> float:
        totals = [0.0, 0.0, 0.0, 0.0]  # up, down, left, right
        for r in range(4):
            trend = 0.0
            for c in range(3):
                if grid[r, c] == 0 or grid[r, c + 1] == 0:
                    continue
                a = float(np.log2(grid[r, c]))
                b = float(np.log2(grid[r, c + 1]))
                trend += 1.0 if a >= b else -1.0
            totals[2] += trend
            totals[3] -= trend
        for c in range(4):
            trend = 0.0
            for r in range(3):
                if grid[r, c] == 0 or grid[r + 1, c] == 0:
                    continue
                a = float(np.log2(grid[r, c]))
                b = float(np.log2(grid[r + 1, c]))
                trend += 1.0 if a >= b else -1.0
            totals[0] += trend
            totals[1] -= trend
        return float(max(totals))

    @staticmethod
    def _merge_potential(grid: np.ndarray) -> int:
        merges = 0
        for r in range(4):
            for c in range(3):
                if grid[r, c] > 0 and grid[r, c] == grid[r, c + 1]:
                    merges += 1
        for r in range(3):
            for c in range(4):
                if grid[r, c] > 0 and grid[r, c] == grid[r + 1, c]:
                    merges += 1
        return merges

    @staticmethod
    def max_tile_in_corner(grid: np.ndarray) -> bool:
        max_tile = int(np.max(grid))
        corners = [(0, 0), (0, 3), (3, 0), (3, 3)]
        return any(int(grid[r, c]) == max_tile for r, c in corners)

    def score(self, grid: np.ndarray, valid_actions: List[int]) -> float:
        empty_cells = int(np.sum(grid == 0))
        max_tile = int(np.max(grid))
        smoothness = self._smoothness_penalty(grid)
        monotonicity = self._monotonicity(grid)
        merge_potential = self._merge_potential(grid)
        corner_bonus = 1.0 if self.max_tile_in_corner(grid) else 0.0
        mobility = len(valid_actions)

        return (
            empty_cells * 1.8
            + monotonicity * 0.9
            - smoothness * 0.8
            + merge_potential * 1.4
            + corner_bonus * 1.5
            + np.log2(max(max_tile, 2)) * 1.0
            + mobility * 0.4
        )


class ExpectimaxActionScorer:
    """Small expectimax evaluator to approximate expert action preference."""

    def __init__(
        self,
        potential_model: BoardPotentialModel,
        depth: int = 2,
        max_empty_branches: int = 8,
    ):
        self.potential_model = potential_model
        self.depth = max(1, int(depth))
        self.max_empty_branches = max(1, int(max_empty_branches))
        self._cache: Dict[Tuple[bytes, int, str], float] = {}

    @staticmethod
    def _compress_and_merge(line: np.ndarray) -> Tuple[np.ndarray, bool, int]:
        non_zero = line[line > 0]
        original = line.copy()
        merged: List[int] = []
        score_gain = 0
        i = 0
        while i < len(non_zero):
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                val = int(non_zero[i] * 2)
                merged.append(val)
                score_gain += val
                i += 2
            else:
                merged.append(int(non_zero[i]))
                i += 1

        result = np.zeros(4, dtype=np.int32)
        result[:len(merged)] = merged
        moved = not np.array_equal(original, result)
        return result, moved, score_gain

    def _simulate_move(self, grid: np.ndarray, action: int) -> Tuple[np.ndarray, bool, int]:
        new_grid = np.array(grid, dtype=np.int32, copy=True)
        moved = False
        score_gain = 0

        if action == 0:  # up
            for col in range(4):
                line = new_grid[:, col]
                out, line_moved, gain = self._compress_and_merge(line)
                new_grid[:, col] = out
                moved = moved or line_moved
                score_gain += gain
        elif action == 1:  # right
            for row in range(4):
                line = new_grid[row, :][::-1]
                out, line_moved, gain = self._compress_and_merge(line)
                new_grid[row, :] = out[::-1]
                moved = moved or line_moved
                score_gain += gain
        elif action == 2:  # down
            for col in range(4):
                line = new_grid[:, col][::-1]
                out, line_moved, gain = self._compress_and_merge(line)
                new_grid[:, col] = out[::-1]
                moved = moved or line_moved
                score_gain += gain
        elif action == 3:  # left
            for row in range(4):
                line = new_grid[row, :]
                out, line_moved, gain = self._compress_and_merge(line)
                new_grid[row, :] = out
                moved = moved or line_moved
                score_gain += gain

        return new_grid, moved, score_gain

    @staticmethod
    def _empty_priority(grid: np.ndarray, rc: Tuple[int, int]) -> float:
        r, c = rc
        score = 0.0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 4 and 0 <= nc < 4 and grid[nr, nc] > 0:
                score += float(np.log2(grid[nr, nc]))
        return score

    def _valid_actions(self, grid: np.ndarray) -> List[int]:
        valid: List[int] = []
        for action in range(4):
            _, moved, _ = self._simulate_move(grid, action)
            if moved:
                valid.append(action)
        return valid

    def _evaluate_grid(self, grid: np.ndarray) -> float:
        valid_actions = self._valid_actions(grid)
        return self.potential_model.score(grid, valid_actions)

    def _expectimax_max(self, grid: np.ndarray, depth: int) -> float:
        key = (grid.tobytes(), depth, "max")
        if key in self._cache:
            return self._cache[key]

        if depth <= 0:
            value = self._evaluate_grid(grid)
            self._cache[key] = value
            return value

        best = -float("inf")
        any_move = False
        for action in range(4):
            moved_grid, moved, gain = self._simulate_move(grid, action)
            if not moved:
                continue
            any_move = True
            value = float(np.tanh(gain / 64.0)) + self._expectimax_chance(moved_grid, depth - 1)
            if value > best:
                best = value

        if not any_move:
            best = self._evaluate_grid(grid)

        self._cache[key] = best
        return best

    def _expectimax_chance(self, grid: np.ndarray, depth: int) -> float:
        key = (grid.tobytes(), depth, "chance")
        if key in self._cache:
            return self._cache[key]

        empties = list(zip(*np.where(grid == 0)))
        if depth <= 0 or not empties:
            value = self._evaluate_grid(grid)
            self._cache[key] = value
            return value

        if len(empties) > self.max_empty_branches:
            empties = sorted(empties, key=lambda rc: self._empty_priority(grid, rc), reverse=True)[
                : self.max_empty_branches
            ]

        cell_prob = 1.0 / len(empties)
        expected = 0.0
        for r, c in empties:
            for tile, prob in ((2, 0.9), (4, 0.1)):
                next_grid = np.array(grid, copy=True)
                next_grid[r, c] = tile
                expected += cell_prob * prob * self._expectimax_max(next_grid, depth - 1)

        self._cache[key] = expected
        return expected

    def action_values(self, grid: np.ndarray, valid_actions: List[int]) -> Dict[int, float]:
        if len(self._cache) > 50000:
            self._cache.clear()
        values: Dict[int, float] = {}
        for action in valid_actions:
            moved_grid, moved, gain = self._simulate_move(grid, action)
            if not moved:
                continue
            values[action] = float(np.tanh(gain / 64.0)) + self._expectimax_chance(
                moved_grid, self.depth - 1
            )
        return values


class GRPO2048Rewards:
    """Reward functions compatible with TRL GRPOTrainer callbacks."""

    _cfg: ExpertRewardConfig = ExpertRewardConfig()
    _potential_model: BoardPotentialModel = BoardPotentialModel()
    _expert_scorer: ExpectimaxActionScorer = ExpectimaxActionScorer(
        potential_model=_potential_model,
        depth=_cfg.expert_depth,
        max_empty_branches=_cfg.expert_max_empty,
    )

    @classmethod
    def configure(cls, cfg: ExpertRewardConfig) -> None:
        cls._cfg = cfg
        cls._expert_scorer = ExpectimaxActionScorer(
            potential_model=cls._potential_model,
            depth=cfg.expert_depth,
            max_empty_branches=cfg.expert_max_empty,
        )

    @staticmethod
    def action_validity(completions: List[Any], **kwargs) -> List[float]:
        valid_actions_col = kwargs.get("valid_actions")
        rewards: List[float] = []

        for i, completion in enumerate(completions):
            text = _completion_to_text(completion)
            action_id = parse_action_from_text(text)

            valid_actions = _coerce_valid_actions(valid_actions_col[i] if valid_actions_col else None)
            if valid_actions:
                rewards.append(1.0 if action_id in valid_actions else -1.0)
            else:
                # Weak fallback when metadata is unavailable.
                rewards.append(0.0)

        return rewards

    @staticmethod
    def one_step_gain(completions: List[Any], **kwargs) -> List[float]:
        state_col = kwargs.get("state_text")
        rewards: List[float] = []

        for i, completion in enumerate(completions):
            text = _completion_to_text(completion)
            action_id = parse_action_from_text(text)

            state_text = state_col[i] if state_col else ""
            game = _game_from_state_text(state_text)
            if game is None:
                rewards.append(0.0)
                continue

            prev_score = game.score
            _, _, _, score = game.step(action_id)
            score_gain = score - prev_score
            rewards.append(float(score_gain))

        return rewards

    @classmethod
    def expert_shaped(cls, completions: List[Any], **kwargs) -> List[float]:
        state_col = kwargs.get("state_text")
        valid_actions_col = kwargs.get("valid_actions")
        rewards: List[float] = []

        for i, completion in enumerate(completions):
            text = _completion_to_text(completion)
            action_id = parse_action_from_text(text)

            state_text = state_col[i] if state_col else ""
            game = _game_from_state_text(state_text)
            if game is None:
                rewards.append(0.0)
                continue

            valid_actions = _coerce_valid_actions(valid_actions_col[i] if valid_actions_col else None)
            if not valid_actions:
                valid_actions = game.get_valid_actions()
            if not valid_actions:
                rewards.append(0.0)
                continue

            if action_id not in valid_actions:
                rewards.append(float(cls._cfg.illegal_penalty))
                continue

            prev_grid = np.array(game.grid, copy=True)
            prev_score = game.score
            prev_empty = int(np.sum(prev_grid == 0))
            prev_mobility = len(valid_actions)
            prev_corner_locked = cls._potential_model.max_tile_in_corner(prev_grid)
            prev_potential = cls._potential_model.score(prev_grid, valid_actions)

            _, _, _, score = game.step(action_id)
            score_gain = float(score - prev_score)
            next_grid = np.array(game.grid, copy=True)
            next_valid_actions = game.get_valid_actions()
            next_potential = cls._potential_model.score(next_grid, next_valid_actions)

            score_term = float(np.tanh(score_gain / max(cls._cfg.score_norm, 1e-6)))
            potential_term = float(
                np.tanh((next_potential - prev_potential) / max(cls._cfg.potential_norm, 1e-6))
            )
            expert_term = cls._expert_term(prev_grid, valid_actions, action_id)
            risk_term = cls._risk_term(
                prev_empty=prev_empty,
                prev_mobility=prev_mobility,
                prev_corner_locked=prev_corner_locked,
                next_grid=next_grid,
                next_valid_actions=next_valid_actions,
            )

            total_reward = (
                cls._cfg.legal_bonus
                + cls._cfg.w_score * score_term
                + cls._cfg.w_potential * potential_term
                + cls._cfg.w_expert * expert_term
                + cls._cfg.w_risk * risk_term
            )
            rewards.append(float(total_reward))

        return rewards

    @classmethod
    def get_reward_funcs(cls, reward_mode: str) -> List[Any]:
        if reward_mode == "simple":
            return [cls.action_validity, cls.one_step_gain]
        return [cls.expert_shaped]

    @classmethod
    def _expert_term(cls, grid: np.ndarray, valid_actions: List[int], action_id: int) -> float:
        values = cls._expert_scorer.action_values(grid, valid_actions)
        if action_id not in values or not values:
            return -1.0

        arr = np.array([values[a] for a in valid_actions if a in values], dtype=float)
        if arr.size <= 1:
            return 0.0
        std = float(arr.std())
        if std < 1e-6:
            return 0.0

        z = (values[action_id] - float(arr.mean())) / std
        return float(np.tanh(z))

    @classmethod
    def _risk_term(
        cls,
        prev_empty: int,
        prev_mobility: int,
        prev_corner_locked: bool,
        next_grid: np.ndarray,
        next_valid_actions: List[int],
    ) -> float:
        next_empty = int(np.sum(next_grid == 0))
        next_mobility = len(next_valid_actions)
        next_corner_locked = cls._potential_model.max_tile_in_corner(next_grid)

        risk = 0.0
        if next_empty <= 1:
            risk -= 1.0
        elif next_empty <= 2:
            risk -= 0.6

        if next_mobility <= 1:
            risk -= 1.0
        elif next_mobility == 2:
            risk -= 0.4

        if prev_corner_locked and not next_corner_locked:
            risk -= 0.8

        if next_empty >= 5 and next_mobility >= 3:
            risk += 0.3
        if next_empty > prev_empty and next_mobility >= prev_mobility:
            risk += 0.2

        return float(np.clip(risk, -2.0, 1.0))


class TRLGRPO2048Trainer:
    """High-level trainer using TRL GRPOTrainer."""

    def __init__(
        self,
        model_name_or_path: str,
        output_dir: str = "./checkpoints/grpo",
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
        learning_rate: float = 1e-6,
        num_train_epochs: int = 1,
        batch_size: int = 2,
        gradient_accumulation_steps: int = 4,
        num_generations: int = 2,
        max_prompt_length: int = 1024,
        max_completion_length: int = 128,
        save_steps: int = 200,
        logging_steps: int = 10,
        reward_mode: str = "expert_shaped",
    ):
        GRPOConfig, GRPOTrainer = _load_trl_grpo_symbols()

        report_to = report_to_list(self.monitor_backend)

        cfg = _build_grpo_config(
            GRPOConfig=GRPOConfig,
            output_dir=self.output_dir,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_generations=num_generations,
            max_prompt_length=max_prompt_length,
            max_completion_length=max_completion_length,
            save_steps=save_steps,
            logging_steps=logging_steps,
            report_to=report_to,
        )

        reward_funcs = GRPO2048Rewards.get_reward_funcs(reward_mode=reward_mode)
        model_input = _resolve_grpo_model_input(self.model_name_or_path)

        trainer = GRPOTrainer(
            model=model_input,
            reward_funcs=reward_funcs,
            args=cfg,
            train_dataset=dataset,
        )

        trainer.train()
        trainer.save_model(self.output_dir)

        # Save metadata for consistent checkpoint contract.
        meta = {
            "trainer": "trl_grpo",
            "model_name_or_path": self.model_name_or_path,
            "num_train_samples": len(dataset),
            "reward_mode": reward_mode,
            "reward_config": vars(GRPO2048Rewards._cfg),
        }
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(self.output_dir) / "training_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        return trainer


def _completion_to_text(completion: Any) -> str:
    """Normalize completion object to text.

    GRPO callbacks may provide plain strings or message-like objects.
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        if "content" in completion and isinstance(completion["content"], str):
            return completion["content"]
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict) and isinstance(last.get("content"), str):
            return last["content"]
    return str(completion)


def _coerce_valid_actions(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, list):
        return [int(v) for v in value]
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [int(v) for v in parsed]
        except Exception:
            return []
    return []


def _game_from_state_text(state_text: str) -> Optional[Game2048]:
    if not state_text:
        return None

    try:
        grid = np.array(ast.literal_eval(state_text), dtype=int)
        if grid.shape != (4, 4):
            return None
    except Exception:
        return None

    game = Game2048()
    game.grid = grid
    game.score = 0
    game.game_over = False
    return game


def _validate_batch_generation_compatibility(batch_size: int, num_generations: int) -> None:
    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    global_batch = int(batch_size) * world_size
    k = int(num_generations)
    if k <= 0:
        raise ValueError(f"num_generations must be > 0, got {k}")
    if global_batch % k != 0:
        valid = [v for v in range(1, global_batch + 1) if global_batch % v == 0]
        raise ValueError(
            f"Incompatible settings: global_train_batch_size={global_batch} "
            f"(batch_size={batch_size}, world_size={world_size}) is not divisible by "
            f"num_generations={k}. Valid num_generations: {valid}"
        )


def main():
    parser = argparse.ArgumentParser(description="TRL GRPO training for 2048")

    parser.add_argument("--model", type=str, default="./checkpoints/sft", help="Model path")
    parser.add_argument("--base_model", type=str, default=None, help="Alias of --model")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/grpo")

    parser.add_argument("--data_source", type=str, default="raw", choices=["raw", "processed"])
    parser.add_argument("--input_dir", type=str, default="data/raw")
    parser.add_argument("--num_samples", type=int, default=1200)
    parser.add_argument("--min_score", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument(
        "--num_generations",
        type=int,
        default=2,
        help="GRPO每个prompt采样条数(K)。要求 global_train_batch_size 可被该值整除。",
    )
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_completion_length", type=int, default=128)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument(
        "--reward_mode",
        type=str,
        default="expert_shaped",
        choices=["simple", "expert_shaped"],
        help="GRPO reward mode",
    )
    parser.add_argument("--reward_illegal_penalty", type=float, default=-5.0)
    parser.add_argument("--reward_legal_bonus", type=float, default=0.2)
    parser.add_argument("--reward_score_norm", type=float, default=64.0)
    parser.add_argument("--reward_potential_norm", type=float, default=5.0)
    parser.add_argument("--reward_w_score", type=float, default=0.15)
    parser.add_argument("--reward_w_potential", type=float, default=0.35)
    parser.add_argument("--reward_w_expert", type=float, default=0.40)
    parser.add_argument("--reward_w_risk", type=float, default=0.10)
    parser.add_argument("--reward_expert_depth", type=int, default=2)
    parser.add_argument("--reward_expert_max_empty", type=int, default=8)

    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument(
        "--monitor_backend",
        type=str,
        default=None,
        choices=["wandb", "tensorboard", "none"],
        help="监控后端 (wandb/tensorboard/none)",
    )
    parser.add_argument("--no_thinking", action="store_true")

    # Backward-compatible legacy args from old grpo.py / scripts.
    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument("--group_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--update_freq", type=int, default=None)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--no_unsloth", action="store_true")
    parser.add_argument("--no_4bit", action="store_true")
    parser.add_argument("--use_8bit", action="store_true")
    parser.add_argument("--no_flash_attn", action="store_true")

    args = parser.parse_args()

    if args.eval_only:
        raise SystemExit("`--eval_only` is removed from TRL GRPO entrypoint. Use src.eval.evaluator.")

    model_name_or_path = args.base_model or args.model
    monitor_backend = normalize_monitor_backend(
        args.monitor_backend,
        no_wandb=args.no_wandb,
    )

    if args.learning_rate is not None:
        args.lr = args.learning_rate
    if args.num_episodes is not None and args.num_samples == 1200:
        # Backward compatibility: old CLI used num_episodes as training scale.
        args.num_samples = max(100, int(args.num_episodes) * 10)

    _validate_batch_generation_compatibility(
        batch_size=args.batch_size,
        num_generations=args.num_generations,
    )

    builder = GRPO2048DatasetBuilder(
        model_name=model_name_or_path,
        use_thinking=not args.no_thinking,
    )

    ds_cfg = GRPO2048DataConfig(
        source=args.data_source,
        input_dir=args.input_dir,
        num_samples=args.num_samples,
        min_score=args.min_score,
        seed=args.seed,
    )

    train_dataset = builder.build(ds_cfg)

    trainer = TRLGRPO2048Trainer(
        model_name_or_path=model_name_or_path,
        output_dir=args.output_dir,
        use_wandb=not args.no_wandb,
        monitor_backend=monitor_backend,
    )

    reward_cfg = ExpertRewardConfig(
        reward_mode=args.reward_mode,
        illegal_penalty=args.reward_illegal_penalty,
        legal_bonus=args.reward_legal_bonus,
        score_norm=args.reward_score_norm,
        potential_norm=args.reward_potential_norm,
        w_score=args.reward_w_score,
        w_potential=args.reward_w_potential,
        w_expert=args.reward_w_expert,
        w_risk=args.reward_w_risk,
        expert_depth=args.reward_expert_depth,
        expert_max_empty=args.reward_expert_max_empty,
    )
    GRPO2048Rewards.configure(reward_cfg)

    trainer.train(
        dataset=train_dataset,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        reward_mode=args.reward_mode,
    )


if __name__ == "__main__":
    main()
