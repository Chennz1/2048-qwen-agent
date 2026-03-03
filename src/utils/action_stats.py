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

    def update(self, *, parsed: bool, legal: bool, correct: bool) -> None:
        self.total += 1
        self.parsed += int(bool(parsed))
        self.legal += int(bool(legal))
        self.correct += int(bool(correct))

    def flush(self) -> Dict[str, float]:
        if self.total <= 0:
            return {}

        denom = float(self.total)
        def _k(name: str) -> str:
            return f"{self.prefix}/{name}" if self.prefix else name

        metrics: Dict[str, float] = {
            _k("parse_rate"): float(self.parsed / denom),
            _k("legal_rate"): float(self.legal / denom),
            _k("action_acc"): float(self.correct / denom),
        }
        self.reset()
        return metrics

    def reset(self) -> None:
        self.total = 0
        self.parsed = 0
        self.legal = 0
        self.correct = 0
