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
import re
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

from src.data_gen.prompting import format_inference_prompt
from src.envs.game_2048 import ACTION_MAP, Game2048
from src.utils.action_stats import ActionWindowStats
from src.utils.monitoring import normalize_monitor_backend, report_to_list


_TRAILING_TEMPLATE_TOKEN_RE = re.compile(
    r"(?:\s*(?:<\|[^>\n]+\|>|</s>|<\s*/s\s*>))+\s*$"
)
_LEADING_TEMPLATE_TOKEN_RE = re.compile(
    r"^(?:\s*(?:<\|[^>\n]+\|>|</s>|<\s*/s\s*>))+"
)
_STRICT_ACTION_OUTPUT_RE = re.compile(
    r"^\s*(?:<think>[\s\S]*?</think>\s*)?([上右下左])\s*$"
)
_ACTION_CHAR_HINT_RE = re.compile(r"(?:动作|action)\s*[:：]?\s*([上右下左])", flags=re.I)
_ACTION_ID_HINT_RE = re.compile(r"(?:动作|action)\s*[:：]?\s*([0-3])", flags=re.I)
_TRAILING_ACTION_CHAR_RE = re.compile(r"([上右下左])\s*$")
_TRAILING_ACTION_ID_RE = re.compile(r"([0-3])\s*$")
_ACTION_WORD_RE = re.compile(r"\b(up|right|down|left)\b", flags=re.I)
_MAX_TILE_CLAIM_RE = re.compile(r"(?:最大(?:数字|块|值)?|max(?:\s*tile)?)\s*[:：]?\s*(\d+)", flags=re.I)
_EMPTY_CLAIM_RE = re.compile(r"(?:空位|空格|空白|empty(?:\s*cells?)?)\s*[:：]?\s*(\d+)", flags=re.I)
_VALID_ACTIONS_SEGMENT_RE = re.compile(r"(?:可行动作|合法动作|可选动作)[^。\n]*", flags=re.I)
_FINAL_ACTION_CLAIM_RE = re.compile(
    r"(?:最终|最后|选择|动作|着法|action|move)\s*(?:为|是|向|:|：)?\s*([上右下左]|[0-3]|up|right|down|left)",
    flags=re.I,
)
_ACTION_CHAR_TO_ID = {v: k for k, v in ACTION_MAP.items()}
_ACTION_WORD_TO_ID = {
    "up": 0,
    "right": 1,
    "down": 2,
    "left": 3,
}


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


def _patch_trainer_action_window_logging(trainer: Any, stats: ActionWindowStats) -> None:
    """Patch trainer.log to emit window stats on the existing logging cadence."""
    original_log = trainer.log

    def _wrapped(self, logs, *args, **kwargs):
        merged = dict(logs) if isinstance(logs, dict) else {}
        merged.update(stats.flush())
        return original_log(merged, *args, **kwargs)

    trainer.log = types.MethodType(_wrapped, trainer)


def _build_grpo_config(
    *,
    GRPOConfig: Any,
    output_dir: str,
    learning_rate: float,
    warmup_ratio: float,
    lr_scheduler_type: str,
    num_train_epochs: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    num_generations: int,
    max_prompt_length: int,
    max_completion_length: int,
    save_steps: int,
    logging_steps: int,
    report_to: List[str],
    clip_eps: float,
    kl_beta: float,
):
    """Build GRPOConfig with runtime compatibility across TRL versions."""
    params = set(inspect.signature(GRPOConfig.__init__).parameters.keys())

    kwargs: Dict[str, Any] = {
        "output_dir": output_dir,
        "learning_rate": learning_rate,
        "warmup_ratio": warmup_ratio,
        "lr_scheduler_type": lr_scheduler_type,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_generations": num_generations,
        "max_prompt_length": max_prompt_length,
        "max_completion_length": max_completion_length,
        "save_steps": save_steps,
        "disable_tqdm": True,
        "logging_steps": logging_steps,
        "report_to": report_to,
        "use_vllm": True,
        "vllm_mode": "colocate",
        "vllm_gpu_memory_utilization" : 0.3,
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

    for key in ("epsilon", "clip_eps", "clip_range"):
        if key in params:
            filtered[key] = clip_eps
            break
    for key in ("beta", "kl_beta"):
        if key in params:
            filtered[key] = kl_beta
            break

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

    source: str = "raw"  # raw only
    input_dir: str = "data/raw"
    num_samples: int = 5000
    min_score: int = 0
    seed: int = 42


@dataclass
class ExpertRewardConfig:
    """Config for expert-shaped GRPO reward."""

    format_penalty: float = -1.2
    illegal_penalty: float = -0.6
    legal_bonus: float = 0.45
    format_quality_weight: float = 0.05
    potential_norm: float = 5.0
    w_score: float = 0.20
    w_potential: float = 0.25
    w_expert: float = 0.35
    w_cot_facts: float = 0.10
    w_cot_consistency: float = 0.08
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
        if cfg.source != "raw":
            raise ValueError(f"Unsupported source: {cfg.source}. GRPO now supports raw only.")
        rows = self._from_raw(Path(cfg.input_dir), cfg.min_score)

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
    _window_stats: Optional[ActionWindowStats] = None
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

    @classmethod
    def attach_window_stats(cls, stats: Optional[ActionWindowStats]) -> None:
        cls._window_stats = stats

    @classmethod
    def _record_window_stats(cls, *, parsed: bool, legal: bool, correct: bool) -> None:
        if cls._window_stats is None:
            return
        cls._window_stats.update(parsed=parsed, legal=legal, correct=correct)

    @classmethod
    def expert_shaped(cls, completions: List[Any], **kwargs) -> List[float]:
        state_col = kwargs.get("state_text")
        valid_actions_col = kwargs.get("valid_actions")
        target_action_col = kwargs.get("target_action")
        rewards: List[float] = []

        for i, completion in enumerate(completions):
            text = _completion_to_text(completion)
            think_text = _extract_think_text(text)
            action_id, format_quality = _parse_action_id_with_quality(text)
            strict_format_bonus = 0.1 if action_id is not None else 0.0
            target_action_id = _coerce_action_id(
                target_action_col[i] if target_action_col and i < len(target_action_col) else None
            )
            parsed = action_id is not None
            legal = False
            correct = bool(parsed and target_action_id is not None and action_id == target_action_id)

            if action_id is None:
                rewards.append(float(cls._cfg.format_penalty))
                cls._record_window_stats(parsed=parsed, legal=legal, correct=correct)
                continue

            state_text = state_col[i] if state_col else ""
            game = _game_from_state_text(state_text)
            if game is None:
                rewards.append(float(strict_format_bonus))
                cls._record_window_stats(parsed=parsed, legal=legal, correct=correct)
                continue

            valid_actions = _coerce_valid_actions(valid_actions_col[i] if valid_actions_col else None)
            if not valid_actions:
                valid_actions = game.get_valid_actions()
            if not valid_actions:
                rewards.append(float(strict_format_bonus))
                cls._record_window_stats(parsed=parsed, legal=legal, correct=correct)
                continue

            if action_id not in valid_actions:
                rewards.append(float(cls._cfg.illegal_penalty + strict_format_bonus))
                cls._record_window_stats(parsed=parsed, legal=legal, correct=correct)
                continue

            prev_grid = np.array(game.grid, copy=True)
            prev_potential = cls._potential_model.score(prev_grid, valid_actions)

            # Use deterministic transition (move without random tile spawn) to keep reward stable.
            next_grid, moved, score_gain = cls._expert_scorer._simulate_move(prev_grid, int(action_id))
            if not moved:
                rewards.append(float(cls._cfg.illegal_penalty + strict_format_bonus))
                cls._record_window_stats(parsed=parsed, legal=legal, correct=correct)
                continue
            score_gain = float(score_gain)
            legal = True
            next_valid_actions = cls._expert_scorer._valid_actions(next_grid)
            next_potential = cls._potential_model.score(next_grid, next_valid_actions)

            if score_gain > 0:
                score_term = float(1.0 + 0.02 * np.log(score_gain))
            else:
                score_term = 0.0
            potential_term = float(
                np.tanh((next_potential - prev_potential) / max(cls._cfg.potential_norm, 1e-6))
            )
            expert_term = cls._expert_term(prev_grid, valid_actions, action_id)
            cot_fact_term = cls._cot_fact_term(think_text, prev_grid, valid_actions)
            cot_consistency_term = cls._cot_consistency_term(
                think_text=think_text,
                action_id=action_id,
                prev_grid=prev_grid,
                next_grid=next_grid,
            )
            format_quality_term = cls._cfg.format_quality_weight * float(np.clip(format_quality, 0.0, 1.0))

            total_reward = (
                cls._cfg.legal_bonus
                + cls._cfg.w_score * score_term
                + cls._cfg.w_potential * potential_term
                + cls._cfg.w_expert * expert_term
                + cls._cfg.w_cot_facts * cot_fact_term
                + cls._cfg.w_cot_consistency * cot_consistency_term
                + format_quality_term
                + strict_format_bonus
            )
            rewards.append(float(total_reward))
            cls._record_window_stats(parsed=parsed, legal=legal, correct=correct)

        return rewards

    @classmethod
    def get_reward_funcs(cls) -> List[Any]:
        return [cls.expert_shaped]

    @classmethod
    def _expert_term(cls, grid: np.ndarray, valid_actions: List[int], action_id: int) -> float:
        values = cls._expert_scorer.action_values(grid, valid_actions)
        if action_id not in values or not values:
            return 0.0

        arr = np.array([values[a] for a in valid_actions if a in values], dtype=float)
        if arr.size == 0:
            return 0.0

        best = float(arr.max())
        worst = float(arr.min())
        span = best - worst
        if span < 1e-6:
            return 0.5

        val = float(values[action_id])
        percentile = (val - worst) / span  # [0, 1]

        std = float(arr.std())
        if std < 1e-6:
            confidence = percentile
        else:
            z = (val - float(arr.mean())) / std
            confidence = 0.5 * (float(np.tanh(z)) + 1.0)  # [0, 1]

        aligned = 0.7 * percentile + 0.3 * confidence
        return float(np.clip(aligned, 0.0, 1.0))

    @classmethod
    def _cot_fact_term(cls, think_text: str, grid: np.ndarray, valid_actions: List[int]) -> float:
        if not think_text:
            return 0.0

        scores: List[float] = []
        true_max_tile = int(np.max(grid))
        true_empty = int(np.sum(grid == 0))
        true_valid_set = set(int(a) for a in valid_actions)

        for m in _MAX_TILE_CLAIM_RE.finditer(think_text):
            claim = int(m.group(1))
            scores.append(1.0 if claim == true_max_tile else -1.0)

        for m in _EMPTY_CLAIM_RE.finditer(think_text):
            claim = int(m.group(1))
            scores.append(1.0 if claim == true_empty else -1.0)

        claimed_actions = _extract_claimed_action_set(think_text)
        if claimed_actions is not None:
            scores.append(1.0 if set(claimed_actions) == true_valid_set else -1.0)

        if not scores:
            return 0.0
        return float(np.clip(float(np.mean(scores)), -1.0, 1.0))

    @classmethod
    def _cot_consistency_term(
        cls,
        think_text: str,
        action_id: int,
        prev_grid: np.ndarray,
        next_grid: np.ndarray,
    ) -> float:
        if not think_text:
            return 0.0

        scores: List[float] = []
        claimed_final = _extract_claimed_final_action(think_text)
        if claimed_final is not None:
            scores.append(1.0 if int(claimed_final) == int(action_id) else -1.0)

        if _mentions_corner_preserve(think_text) and cls._potential_model.max_tile_in_corner(prev_grid):
            next_corner_locked = cls._potential_model.max_tile_in_corner(next_grid)
            scores.append(1.0 if next_corner_locked else -1.0)

        if not scores:
            return 0.0
        return float(np.clip(float(np.mean(scores)), -1.0, 1.0))

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
        max_prompt_length: int = 1024,
        max_completion_length: int = 128,
        save_steps: int = 200,
        logging_steps: int = 4,
    ):
        GRPOConfig, GRPOTrainer = _load_trl_grpo_symbols()

        report_to = report_to_list(self.monitor_backend)

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
            save_steps=save_steps,
            logging_steps=logging_steps,
            report_to=report_to,
            clip_eps=clip_eps,
            kl_beta=kl_beta,
        )

        reward_funcs = GRPO2048Rewards.get_reward_funcs()
        model_input = _resolve_grpo_model_input(self.model_name_or_path)
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
        action_stats = ActionWindowStats()
        GRPO2048Rewards.attach_window_stats(action_stats)
        _patch_trainer_action_window_logging(trainer, action_stats)

        try:
            trainer.train()
            tail_metrics = action_stats.flush()
            if tail_metrics:
                trainer.log(tail_metrics)
        finally:
            GRPO2048Rewards.attach_window_stats(None)
        trainer.save_model(self.output_dir)

        # Save metadata for consistent checkpoint contract.
        meta = {
            "trainer": "trl_grpo",
            "model_name_or_path": self.model_name_or_path,
            "num_train_samples": len(dataset),
            "reward_mode": "expert_shaped",
            "reward_config": vars(GRPO2048Rewards._cfg),
        }
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(self.output_dir) / "training_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        return trainer


def _normalize_completion_text(text: str) -> str:
    normalized = text.rstrip()
    while True:
        stripped = _TRAILING_TEMPLATE_TOKEN_RE.sub("", normalized)
        if stripped == normalized:
            break
        normalized = stripped.rstrip()
    while True:
        stripped = _LEADING_TEMPLATE_TOKEN_RE.sub("", normalized)
        if stripped == normalized:
            break
        normalized = stripped.lstrip()
    return normalized


def _parse_strict_action_id(text: str) -> Optional[int]:
    """Parse action from strict output: `动作` or `<think>...</think> + 动作`."""
    normalized = _normalize_completion_text(text)
    match = _STRICT_ACTION_OUTPUT_RE.fullmatch(normalized)
    if not match:
        return None
    return _ACTION_CHAR_TO_ID.get(match.group(1))


def _parse_action_id(text: str) -> Optional[int]:
    """Parse action with strict format only."""
    action_id, _ = _parse_action_id_with_quality(text)
    return action_id


def _parse_action_id_with_quality(text: str) -> Tuple[Optional[int], float]:
    """Parse action with strict format only and return (action_id, format_quality in [0,1])."""
    normalized = _normalize_completion_text(text)

    strict = _parse_strict_action_id(normalized)
    if strict is not None:
        return strict, 1.0

    return None, 0.0


def _decode_action_token(token: str) -> Optional[int]:
    token_norm = token.strip().lower()
    if not token_norm:
        return None
    if token_norm in _ACTION_WORD_TO_ID:
        return _ACTION_WORD_TO_ID[token_norm]
    if token_norm in {"上", "右", "下", "左"}:
        return _ACTION_CHAR_TO_ID.get(token_norm)
    if token_norm in {"0", "1", "2", "3"}:
        return int(token_norm)
    return None


def _extract_think_text(text: str) -> str:
    normalized = _normalize_completion_text(text)
    match = re.search(r"<think>\s*([\s\S]*?)\s*</think>", normalized, flags=re.I)
    if match:
        return match.group(1).strip()
    return normalized


def _extract_claimed_action_set(text: str) -> Optional[List[int]]:
    segments = _VALID_ACTIONS_SEGMENT_RE.findall(text)
    for seg in segments:
        actions: List[int] = []
        for ch in seg:
            if ch in _ACTION_CHAR_TO_ID:
                actions.append(int(_ACTION_CHAR_TO_ID[ch]))
        for d in re.findall(r"[0-3]", seg):
            actions.append(int(d))
        for word in _ACTION_WORD_RE.findall(seg):
            action_id = _ACTION_WORD_TO_ID.get(word.lower())
            if action_id is not None:
                actions.append(int(action_id))
        if actions:
            return sorted(set(actions))
    return None


def _extract_claimed_final_action(text: str) -> Optional[int]:
    matches = _FINAL_ACTION_CLAIM_RE.findall(text)
    if not matches:
        return None
    return _decode_action_token(matches[-1])


def _mentions_corner_preserve(text: str) -> bool:
    if "角" not in text:
        return False
    keywords = ["保持", "固定", "锁", "不要", "避免", "留在", "别把", "不能"]
    return any(k in text for k in keywords)


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


def _coerce_action_id(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        action_id = int(value)
        return action_id if 0 <= action_id <= 3 else None
    if isinstance(value, str):
        decoded = _decode_action_token(value)
        if decoded is not None and 0 <= int(decoded) <= 3:
            return int(decoded)
    return None


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
    prompts_per_step = global_batch // k
    if prompts_per_step < 2:
        print(
            "[GRPO] 警告: 每步仅有 1 个prompt组 "
            f"(global_batch={global_batch}, num_generations={k})。"
            "这会放大 reward zero-std 风险，建议降低 --num_generations 或提升 --batch_size。"
        )


def main():
    parser = argparse.ArgumentParser(description="TRL GRPO training for 2048")

    parser.add_argument("--model", type=str, default="./checkpoints/sft", help="Model path")
    parser.add_argument("--base_model", type=str, default=None, help="Alias of --model")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/grpo")

    parser.add_argument("--data_source", type=str, default="raw", choices=["raw"])
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
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--clip_eps", type=float, default=0.28)
    parser.add_argument("--kl_beta", type=float, default=0.0)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_completion_length", type=int, default=768)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument(
        "--reward_mode",
        type=str,
        default="expert_shaped",
        choices=["expert_shaped"],
        help="GRPO reward mode (expert_shaped only)",
    )
    parser.add_argument("--reward_format_penalty", type=float, default=-1.2)
    parser.add_argument("--reward_illegal_penalty", type=float, default=-0.6)
    parser.add_argument("--reward_legal_bonus", type=float, default=0.45)
    parser.add_argument("--reward_format_quality_weight", type=float, default=0.05)
    parser.add_argument("--reward_potential_norm", type=float, default=5.0)
    parser.add_argument("--reward_w_score", type=float, default=0.20)
    parser.add_argument("--reward_w_potential", type=float, default=0.25)
    parser.add_argument("--reward_w_expert", type=float, default=0.35)
    parser.add_argument("--reward_w_cot_facts", type=float, default=0.10)
    parser.add_argument("--reward_w_cot_consistency", type=float, default=0.08)
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
        format_penalty=args.reward_format_penalty,
        illegal_penalty=args.reward_illegal_penalty,
        legal_bonus=args.reward_legal_bonus,
        format_quality_weight=args.reward_format_quality_weight,
        potential_norm=args.reward_potential_norm,
        w_score=args.reward_w_score,
        w_potential=args.reward_w_potential,
        w_expert=args.reward_w_expert,
        w_cot_facts=args.reward_w_cot_facts,
        w_cot_consistency=args.reward_w_cot_consistency,
        expert_depth=args.reward_expert_depth,
        expert_max_empty=args.reward_expert_max_empty,
    )
    GRPO2048Rewards.configure(reward_cfg)

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
    )


if __name__ == "__main__":
    main()
