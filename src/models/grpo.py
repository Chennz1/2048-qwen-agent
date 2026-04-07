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
import math
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
from src.envs.game_2048 import ACTION_MAP, ACTION_MAP_ENGLISH, Game2048, parse_action_from_non_think_text
from src.utils.action_stats import ActionWindowStats
from src.utils.monitoring import normalize_monitor_backend, report_to_list


_TRAILING_TEMPLATE_TOKEN_RE = re.compile(
    r"(?:\s*(?:<\|[^>\n]+\|>|</s>|<\s*/s\s*>))+\s*$"
)
_LEADING_TEMPLATE_TOKEN_RE = re.compile(
    r"^(?:\s*(?:<\|[^>\n]+\|>|</s>|<\s*/s\s*>))+"
)
_THINK_BLOCK_RE = re.compile(r"<think>\s*([\s\S]*?)\s*</think>", flags=re.I)
_THINK_CLOSE_RE = re.compile(r"</think>", flags=re.I)
_ACTION_CHAR_TO_ID = {v: k for k, v in ACTION_MAP.items()}
_ACTION_WORD_TO_ID = {
    "up": 0,
    "right": 1,
    "down": 2,
    "left": 3,
}
_TOP_LEVEL_KEYS_EN = {"board", "judgment", "choice"}
_SITUATION_KEYS_EN = {"max_tile", "positions", "in_corner"}
_JUDGMENT_KEYS_EN = {"UP", "RIGHT", "DOWN", "LEFT"}
_TOP_LEVEL_KEYS_ZH = {"局面", "判断", "选择"}
_SITUATION_KEYS_ZH = {"最大数字", "位置", "在角落"}
_JUDGMENT_KEYS_ZH = {"上", "右", "下", "左"}


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
    disable_tqdm: bool,
    report_to: List[str],
    clip_eps: float,
    kl_beta: float,
):
    """Build GRPOConfig with runtime compatibility across TRL versions."""
    params = set(inspect.signature(GRPOConfig.__init__).parameters.keys())
    # Prevent vLLM from allocating KV cache for the model's full native context
    # (e.g. 40k), which can OOM in colocate mode.
    vllm_max_model_len = int(max_prompt_length + max_completion_length + 64)

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
        "disable_tqdm": bool(disable_tqdm),
        "logging_steps": logging_steps,
        "report_to": report_to,
        "use_vllm": False,  # 关闭 vLLM 以省下预分配显存
        "gradient_checkpointing": True,  # 开启梯度检查点，极大降低反向传播时的显存峰值
        
        # "use_vllm": True,
        # "vllm_mode": "colocate",
        # "vllm_gpu_memory_utilization": 0.25,
        # "max_model_len": vllm_max_model_len,
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


def resolve_save_steps(
    *,
    save_steps: float,
    dataset_size: int,
    num_train_epochs: int,
    batch_size: int,
    gradient_accumulation_steps: int,
) -> int:
    """Resolve save_steps from either an absolute step count or a total-step ratio."""
    value = float(save_steps)
    if value <= 0:
        raise ValueError(f"save_steps must be > 0, got {save_steps}")
    if value >= 1:
        return max(1, int(round(value)))

    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    effective_batch = max(1, int(batch_size) * int(gradient_accumulation_steps) * world_size)
    steps_per_epoch = max(1, int(math.ceil(float(dataset_size) / float(effective_batch))))
    total_steps = max(1, int(math.ceil(float(num_train_epochs) * float(steps_per_epoch))))
    resolved = max(1, int(math.ceil(total_steps * value)))
    print(
        f"[GRPO] Resolved save_steps ratio {value:.4f} "
        f"-> every {resolved} update steps (estimated total steps: {total_steps})"
    )
    return resolved


@dataclass
class GRPO2048DataConfig:
    """Prompt dataset config for GRPO."""

    source: str = "raw"  # raw only
    input_dir: str = "data/raw"
    num_samples: int = 5000
    min_score: int = 0
    seed: int = 42


@dataclass
class JsonRewardConfig:
    """Config for fixed-score JSON-focused GRPO reward."""

    json_invalid_penalty: float = -20.0

    top_level_ok_bonus: float = 1.0
    top_level_bad_penalty: float = -1.0
    situation_schema_ok_bonus: float = 1.0
    situation_schema_bad_penalty: float = -1.0
    judgment_schema_ok_bonus: float = 1.0
    judgment_schema_bad_penalty: float = -1.0
    choice_schema_ok_bonus: float = 1.0
    choice_schema_bad_penalty: float = -2.0

    max_tile_correct_bonus: float = 1.0
    max_tile_wrong_penalty: float = -1.0
    positions_correct_bonus: float = 2.0
    positions_wrong_penalty: float = -2.0
    corner_correct_bonus: float = 1.0
    corner_wrong_penalty: float = -1.0

    judgment_match_bonus: float = 1.0
    judgment_mismatch_penalty: float = -0.5

    choice_legal_bonus: float = 1.0
    choice_illegal_penalty: float = -0.5

    expert_match_bonus: float = 3.0
    all_correct_bonus: float = 2.0
    cot_length_threshold: int = 100
    cot_length_bonus: float = 0.5

    expert_depth: int = 2
    expert_max_empty: int = 8


# Backward-compatible alias for older imports.
ExpertRewardConfig = JsonRewardConfig


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

    _cfg: JsonRewardConfig = JsonRewardConfig()
    _window_stats: Optional[ActionWindowStats] = None
    _potential_model: BoardPotentialModel = BoardPotentialModel()
    _expert_scorer: ExpectimaxActionScorer = ExpectimaxActionScorer(
        potential_model=_potential_model,
        depth=_cfg.expert_depth,
        max_empty_branches=_cfg.expert_max_empty,
    )

    @classmethod
    def configure(cls, cfg: JsonRewardConfig) -> None:
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
    def _record_window_stats(
        cls,
        *,
        parsed: bool,
        legal: bool,
        correct: bool,
        reward_format: Optional[float] = None,
        reward_legal: Optional[float] = None,
        reward_facts: Optional[float] = None,
        reward_expert: Optional[float] = None,
        reward_all_correct: Optional[float] = None,
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

    @staticmethod
    def _is_plain_int(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool)

    @classmethod
    def _extract_non_think_region(cls, text: str) -> str:
        normalized = _normalize_completion_text(text)
        closes = list(_THINK_CLOSE_RE.finditer(normalized))
        if closes:
            return normalized[closes[-1].end():].strip()
        return normalized.strip()

    @classmethod
    def _parse_non_think_json_object(cls, text: str) -> Optional[Dict[str, Any]]:
        candidate = cls._extract_non_think_region(text)
        if not candidate:
            return None
        try:
            parsed = json.loads(candidate)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        return parsed if isinstance(parsed, dict) else None

    @classmethod
    def _extract_think_region(cls, text: str) -> str:
        normalized = _normalize_completion_text(text)
        match = _THINK_BLOCK_RE.search(normalized)
        if match is None:
            return ""
        return str(match.group(1) or "").strip()

    @classmethod
    def _coerce_position_set(cls, value: Any) -> Optional[set[tuple[int, int]]]:
        if not isinstance(value, list):
            return None
        result: set[tuple[int, int]] = set()
        for coord in value:
            if not isinstance(coord, list) or len(coord) != 2:
                return None
            r, c = coord[0], coord[1]
            if not (cls._is_plain_int(r) and cls._is_plain_int(c)):
                return None
            if not (0 <= int(r) <= 3 and 0 <= int(c) <= 3):
                return None
            result.add((int(r), int(c)))
        return result

    @classmethod
    def _expert_best_actions(cls, grid: np.ndarray, valid_actions: List[int]) -> set[int]:
        values = cls._expert_scorer.action_values(grid, valid_actions)
        if not values:
            return set()
        best = float(max(values.values()))
        return {int(a) for a, v in values.items() if abs(float(v) - best) <= 1e-6}

    @classmethod
    def json_focused(cls, completions: List[Any], **kwargs) -> List[float]:
        state_col = kwargs.get("state_text")
        valid_actions_col = kwargs.get("valid_actions")
        target_action_col = kwargs.get("target_action")
        rewards: List[float] = []

        for i, completion in enumerate(completions):
            text = _completion_to_text(completion)
            think_text = cls._extract_think_region(text)
            predicted = cls._parse_non_think_json_object(text)
            parsed = predicted is not None
            legal = False
            correct = False
            format_score = 0.0
            legal_score = 0.0
            facts_score = 0.0
            expert_score = 0.0
            all_correct_score = 0.0
            cot_length_score = 0.0
            format_full = False
            legal_full = False
            facts_full = False
            if len(think_text) > int(cls._cfg.cot_length_threshold):
                cot_length_score += float(cls._cfg.cot_length_bonus)

            if predicted is None:
                format_score = float(cls._cfg.json_invalid_penalty)
                total_score = float(format_score + cot_length_score)
                rewards.append(total_score)
                cls._record_window_stats(
                    parsed=parsed,
                    legal=legal,
                    correct=correct,
                    reward_format=format_score,
                    reward_legal=legal_score,
                    reward_facts=facts_score,
                    reward_expert=expert_score,
                    reward_all_correct=all_correct_score,
                )
                continue

            state_text = state_col[i] if state_col else ""
            game = _game_from_state_text(state_text)
            if game is None:
                format_score = float(cls._cfg.json_invalid_penalty)
                total_score = float(format_score + cot_length_score)
                rewards.append(total_score)
                cls._record_window_stats(
                    parsed=parsed,
                    legal=legal,
                    correct=correct,
                    reward_format=format_score,
                    reward_legal=legal_score,
                    reward_facts=facts_score,
                    reward_expert=expert_score,
                    reward_all_correct=all_correct_score,
                )
                continue

            valid_actions = _coerce_valid_actions(valid_actions_col[i] if valid_actions_col else None)
            if not valid_actions:
                valid_actions = game.get_valid_actions()
            prev_grid = np.array(game.grid, copy=True)
            valid_set = set(int(a) for a in valid_actions)
            true_max_tile = int(np.max(prev_grid))
            true_positions = {
                (int(r), int(c))
                for r, c in zip(*np.where(prev_grid == true_max_tile))
            }
            true_corner = any((r, c) in {(0, 0), (0, 3), (3, 0), (3, 3)} for r, c in true_positions)
            true_judgment_en = {ACTION_MAP_ENGLISH[a]: (a in valid_set) for a in range(4)}
            true_judgment_zh = {ACTION_MAP[a]: (a in valid_set) for a in range(4)}
            is_english_schema = set(predicted.keys()) == _TOP_LEVEL_KEYS_EN
            top_level_ok = is_english_schema or set(predicted.keys()) == _TOP_LEVEL_KEYS_ZH
            format_score += (
                float(cls._cfg.top_level_ok_bonus)
                if top_level_ok
                else float(cls._cfg.top_level_bad_penalty)
            )

            situation = predicted.get("board") if is_english_schema else predicted.get("局面")
            pred_pos_set: Optional[set[tuple[int, int]]] = None
            if isinstance(situation, dict):
                pred_pos_set = cls._coerce_position_set(
                    situation.get("positions") if is_english_schema else situation.get("位置")
                )
            situation_schema_ok = (
                isinstance(situation, dict)
                and set(situation.keys()) == (_SITUATION_KEYS_EN if is_english_schema else _SITUATION_KEYS_ZH)
                and cls._is_plain_int(situation.get("max_tile") if is_english_schema else situation.get("最大数字"))
                and pred_pos_set is not None
                and isinstance(situation.get("in_corner") if is_english_schema else situation.get("在角落"), bool)
            )
            format_score += (
                float(cls._cfg.situation_schema_ok_bonus)
                if situation_schema_ok
                else float(cls._cfg.situation_schema_bad_penalty)
            )

            judgment = predicted.get("judgment") if is_english_schema else predicted.get("判断")
            judgment_schema_ok = (
                isinstance(judgment, dict)
                and set(judgment.keys()) == (_JUDGMENT_KEYS_EN if is_english_schema else _JUDGMENT_KEYS_ZH)
                and all(isinstance(v, bool) for v in judgment.values())
            )
            format_score += (
                float(cls._cfg.judgment_schema_ok_bonus)
                if judgment_schema_ok
                else float(cls._cfg.judgment_schema_bad_penalty)
            )

            choice_raw = predicted.get("choice") if is_english_schema else predicted.get("选择")
            choice_name = choice_raw.strip() if isinstance(choice_raw, str) else None
            choice_id = _decode_action_token(choice_name or "")
            choice_schema_ok = choice_id is not None
            format_score += (
                float(cls._cfg.choice_schema_ok_bonus)
                if choice_schema_ok
                else float(cls._cfg.choice_schema_bad_penalty)
            )
            format_full = bool(
                top_level_ok and situation_schema_ok and judgment_schema_ok and choice_schema_ok
            )

            pred_max_tile = (
                situation.get("max_tile") if is_english_schema and isinstance(situation, dict)
                else situation.get("最大数字") if isinstance(situation, dict)
                else None
            )
            if cls._is_plain_int(pred_max_tile) and int(pred_max_tile) == true_max_tile:
                facts_score += float(cls._cfg.max_tile_correct_bonus)
                max_tile_ok = True
            else:
                facts_score += float(cls._cfg.max_tile_wrong_penalty)
                max_tile_ok = False

            if pred_pos_set is not None and pred_pos_set == true_positions:
                facts_score += float(cls._cfg.positions_correct_bonus)
                positions_ok = True
            else:
                facts_score += float(cls._cfg.positions_wrong_penalty)
                positions_ok = False

            pred_corner = (
                situation.get("in_corner") if is_english_schema and isinstance(situation, dict)
                else situation.get("在角落") if isinstance(situation, dict)
                else None
            )
            if isinstance(pred_corner, bool) and bool(pred_corner) == bool(true_corner):
                facts_score += float(cls._cfg.corner_correct_bonus)
                corner_ok = True
            else:
                facts_score += float(cls._cfg.corner_wrong_penalty)
                corner_ok = False
            facts_full = bool(max_tile_ok and positions_ok and corner_ok)

            all_judgment_ok = True
            for action_id in range(4):
                action_name = ACTION_MAP_ENGLISH[action_id] if is_english_schema else ACTION_MAP[action_id]
                expected = true_judgment_en[action_name] if is_english_schema else true_judgment_zh[action_name]
                pred_val = judgment.get(action_name) if isinstance(judgment, dict) else None
                if isinstance(pred_val, bool) and pred_val == expected:
                    legal_score += float(cls._cfg.judgment_match_bonus)
                else:
                    legal_score += float(cls._cfg.judgment_mismatch_penalty)
                    all_judgment_ok = False

            if choice_id is not None and int(choice_id) in valid_set:
                legal = True
                legal_score += float(cls._cfg.choice_legal_bonus)
            else:
                legal_score += float(cls._cfg.choice_illegal_penalty)
            legal_full = bool(all_judgment_ok and legal)

            if choice_id is not None:
                expert_best = cls._expert_best_actions(prev_grid, valid_actions)
                if int(choice_id) in expert_best:
                    expert_score += float(cls._cfg.expert_match_bonus)

            target_action_id = _coerce_action_id(
                target_action_col[i] if target_action_col and i < len(target_action_col) else None
            )
            correct = bool(choice_id is not None and target_action_id is not None and int(choice_id) == target_action_id)
            if bool(format_full and legal_full and facts_full and correct):
                all_correct_score += float(cls._cfg.all_correct_bonus)

            total_score = float(
                format_score
                + legal_score
                + facts_score
                + expert_score
                + all_correct_score
                + cot_length_score
            )
            rewards.append(total_score)
            cls._record_window_stats(
                parsed=parsed,
                legal=legal,
                correct=correct,
                reward_format=format_score,
                reward_legal=legal_score,
                reward_facts=facts_score,
                reward_expert=expert_score,
                reward_all_correct=all_correct_score,
            )

        return rewards

    @classmethod
    def expert_shaped(cls, completions: List[Any], **kwargs) -> List[float]:
        """Backward-compatible alias; now uses fixed-score JSON-focused reward."""
        return cls.json_focused(completions, **kwargs)

    @classmethod
    def get_reward_funcs(cls) -> List[Any]:
        return [cls.json_focused]

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
        max_prompt_length: int = 600,
        max_completion_length: int = 768,
        save_steps: float = 200,
        logging_steps: int = 4,
        disable_tqdm: bool = False,
    ):
        GRPOConfig, GRPOTrainer = _load_trl_grpo_symbols()

        report_to = report_to_list(self.monitor_backend)
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
            "reward_mode": "json_focused",
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
    """Parse action from strict output JSON in non-think region only."""
    normalized = _normalize_completion_text(text)
    parsed = parse_action_from_non_think_text(normalized)
    return int(parsed) if parsed is not None else None


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
    parser.add_argument("--max_prompt_length", type=int, default=600)
    parser.add_argument("--max_completion_length", type=int, default=768)
    parser.add_argument("--save_steps", type=float, default=1000)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--disable_tqdm", action="store_true")
    parser.add_argument(
        "--reward_mode",
        type=str,
        default="json_focused",
        choices=["json_focused"],
        help="GRPO reward mode (json_focused only)",
    )
    parser.add_argument("--reward_json_invalid_penalty", type=float, default=-20.0)
    parser.add_argument("--reward_expert_depth", type=int, default=2)
    parser.add_argument("--reward_expert_max_empty", type=int, default=8)

    # Deprecated legacy args kept for CLI compatibility; no longer used.
    parser.add_argument("--reward_format_penalty", type=float, default=None)
    parser.add_argument("--reward_illegal_penalty", type=float, default=None)
    parser.add_argument("--reward_legal_bonus", type=float, default=None)
    parser.add_argument("--reward_format_quality_weight", type=float, default=None)
    parser.add_argument("--reward_potential_norm", type=float, default=None)
    parser.add_argument("--reward_w_score", type=float, default=None)
    parser.add_argument("--reward_w_potential", type=float, default=None)
    parser.add_argument("--reward_w_expert", type=float, default=None)
    parser.add_argument("--reward_w_cot_facts", type=float, default=None)
    parser.add_argument("--reward_w_cot_consistency", type=float, default=None)

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

    reward_cfg = JsonRewardConfig(
        json_invalid_penalty=args.reward_json_invalid_penalty,
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
        disable_tqdm=args.disable_tqdm,
    )


if __name__ == "__main__":
    main()
