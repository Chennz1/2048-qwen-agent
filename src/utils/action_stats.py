"""Windowed action-stat metrics for periodic training logs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class ActionWindowStats:
    """Collect action metrics and flush/reset as one logging window."""

    prefix: str = ""
    total: int = 0
    parsed: int = 0
    legal: int = 0
    correct: int = 0
    reward_count: int = 0
    reward_format_sum: float = 0.0
    reward_legal_sum: float = 0.0
    reward_facts_sum: float = 0.0
    reward_expert_sum: float = 0.0
    reward_all_correct_sum: float = 0.0

    def update(
        self,
        *,
        parsed: bool,
        legal: bool,
        correct: bool,
        reward_format: float | None = None,
        reward_legal: float | None = None,
        reward_facts: float | None = None,
        reward_expert: float | None = None,
        reward_all_correct: float | None = None,
    ) -> None:
        self.total += 1
        self.parsed += int(bool(parsed))
        self.legal += int(bool(legal))
        self.correct += int(bool(correct))
        has_reward_item = any(
            v is not None
            for v in (
                reward_format,
                reward_legal,
                reward_facts,
                reward_expert,
                reward_all_correct,
            )
        )
        if not has_reward_item:
            return

        self.reward_count += 1
        self.reward_format_sum += float(reward_format or 0.0)
        self.reward_legal_sum += float(reward_legal or 0.0)
        self.reward_facts_sum += float(reward_facts or 0.0)
        self.reward_expert_sum += float(reward_expert or 0.0)
        self.reward_all_correct_sum += float(reward_all_correct or 0.0)

    def flush(self) -> Dict[str, float]:
        if self.total <= 0:
            return {}

        denom = float(self.total)
        def _k(group: str, name: str) -> str:
            return f"{group}/{name}"

        metrics: Dict[str, float] = {
            _k("action", "parse_rate"): float(self.parsed / denom),
            _k("action", "legal_rate"): float(self.legal / denom),
            _k("action", "action_acc"): float(self.correct / denom),
        }
        if self.reward_count > 0:
            reward_denom = float(self.reward_count)
            metrics.update(
                {
                    _k("reward", "format_mean"): float(self.reward_format_sum / reward_denom),
                    _k("reward", "legal_mean"): float(self.reward_legal_sum / reward_denom),
                    _k("reward", "facts_mean"): float(self.reward_facts_sum / reward_denom),
                    _k("reward", "expert_mean"): float(self.reward_expert_sum / reward_denom),
                    _k("reward", "all_correct_mean"): float(
                        self.reward_all_correct_sum / reward_denom
                    ),
                }
            )
        self.reset()
        return metrics

    def reset(self) -> None:
        self.total = 0
        self.parsed = 0
        self.legal = 0
        self.correct = 0
        self.reward_count = 0
        self.reward_format_sum = 0.0
        self.reward_legal_sum = 0.0
        self.reward_facts_sum = 0.0
        self.reward_expert_sum = 0.0
        self.reward_all_correct_sum = 0.0
