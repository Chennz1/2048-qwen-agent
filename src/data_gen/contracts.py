"""
Data contracts for raw game trajectories and processed training samples.

This module is the single source of truth for dataset schemas used by:
- data generation
- data processing
- SFT/ReST/GRPO training
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List, Optional, Tuple

SCHEMA_VERSION = "1.0"
VALID_ACTIONS = ["上", "右", "下", "左"]
VALID_ACTION_IDS = [0, 1, 2, 3]
_TRAILING_TEMPLATE_TOKEN_RE = re.compile(
    r"(?:\s*(?:<\|[^>\n]+\|>|</s>|<\s*/s\s*>))+\s*$"
)


@dataclass
class ValidationResult:
    is_valid: bool
    error: Optional[str] = None



def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0



def validate_raw_game(game: Dict) -> ValidationResult:
    """Validate one raw game trajectory object."""
    required_fields = [
        "schema_version",
        "game_id",
        "difficulty",
        "states",
        "final_score",
        "max_tile",
        "total_steps",
    ]

    for field in required_fields:
        if field not in game:
            return ValidationResult(False, f"Missing required field: {field}")

    if game["schema_version"] != SCHEMA_VERSION:
        return ValidationResult(
            False,
            f"Unsupported schema_version: {game['schema_version']} (expected {SCHEMA_VERSION})",
        )

    states = game["states"]
    if not isinstance(states, list) or not states:
        return ValidationResult(False, "states must be a non-empty list")

    if game["total_steps"] != len(states):
        return ValidationResult(
            False,
            f"total_steps mismatch: {game['total_steps']} vs {len(states)}",
        )

    if game["final_score"] < 0:
        return ValidationResult(False, f"Negative final_score: {game['final_score']}")

    max_tile = game["max_tile"]
    if not isinstance(max_tile, int) or not _is_power_of_two(max_tile):
        return ValidationResult(False, f"max_tile must be a power of 2, got: {max_tile}")

    prev_step = -1
    prev_score = -1

    for idx, state in enumerate(states):
        required_state_fields = ["state", "action", "action_id", "score", "step"]
        for field in required_state_fields:
            if field not in state:
                return ValidationResult(False, f"State {idx}: missing field {field}")

        action = state["action"]
        action_id = state["action_id"]

        if action not in VALID_ACTIONS:
            return ValidationResult(False, f"State {idx}: invalid action '{action}'")
        if action_id not in VALID_ACTION_IDS:
            return ValidationResult(False, f"State {idx}: invalid action_id {action_id}")
        if VALID_ACTIONS[action_id] != action:
            return ValidationResult(
                False,
                f"State {idx}: action/action_id mismatch ({action}, {action_id})",
            )

        step = state["step"]
        score = state["score"]

        if not isinstance(step, int) or step != prev_step + 1:
            return ValidationResult(
                False,
                f"State {idx}: step should be contiguous from 0, got {step}",
            )

        if not isinstance(score, int) or score < 0:
            return ValidationResult(False, f"State {idx}: invalid score {score}")

        # Score should not go backward in recorded trajectory.
        if score < prev_score:
            return ValidationResult(
                False,
                f"State {idx}: score regressed {score} < {prev_score}",
            )

        prev_step = step
        prev_score = score

        if "thinking" in state and not isinstance(state["thinking"], str):
            return ValidationResult(False, f"State {idx}: thinking must be string if provided")

    return ValidationResult(True)



def validate_processed_sample(sample: Dict, require_thinking: bool = False) -> ValidationResult:
    """Validate one processed training sample."""
    prompt = sample.get("prompt")
    completion = sample.get("completion")
    if not isinstance(prompt, list) or not prompt:
        return ValidationResult(False, "Sample must contain non-empty `prompt` message list")
    if not isinstance(completion, list) or not completion:
        return ValidationResult(False, "Sample must contain non-empty `completion` message list")

    for name, messages in (("prompt", prompt), ("completion", completion)):
        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                return ValidationResult(False, f"{name}[{i}] must be dict")
            role = message.get("role")
            content = message.get("content")
            if role not in {"system", "user", "assistant"}:
                return ValidationResult(False, f"{name}[{i}] has invalid role: {role}")
            if not isinstance(content, str) or not content.strip():
                return ValidationResult(False, f"{name}[{i}].content must be non-empty string")

    assistant_msgs = [
        m for m in completion
        if isinstance(m, dict) and m.get("role") == "assistant" and isinstance(m.get("content"), str)
    ]
    if not assistant_msgs:
        return ValidationResult(False, "completion must include at least one assistant message")
    target_text = assistant_msgs[-1]["content"]

    normalized = target_text.rstrip()
    while True:
        stripped = _TRAILING_TEMPLATE_TOKEN_RE.sub("", normalized)
        if stripped == normalized:
            break
        normalized = stripped.rstrip()

    if not any(normalized.endswith(action) for action in VALID_ACTIONS):
        return ValidationResult(False, "assistant response must end with one legal action token")

    has_think_tag = "<think>" in target_text and "</think>" in target_text
    if require_thinking and not has_think_tag:
        return ValidationResult(False, "CoT sample must contain <think>...</think>")

    return ValidationResult(True)



def summarize_validation_failures(errors: List[str], max_items: int = 10) -> str:
    preview = "\n".join(f"  - {item}" for item in errors[:max_items])
    suffix = "" if len(errors) <= max_items else f"\n  ... and {len(errors) - max_items} more"
    return f"Found {len(errors)} validation issues:\n{preview}{suffix}"
