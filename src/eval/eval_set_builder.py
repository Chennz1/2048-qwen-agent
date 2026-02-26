"""Build fixed evaluation sets (easy/medium/hard) from raw trajectories."""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class EvalSetConfig:
    raw_dir: str = "data/raw"
    output_dir: str = "data/eval_sets"
    total_size: int = 1500
    easy_ratio: float = 0.2
    medium_ratio: float = 0.4
    hard_ratio: float = 0.4
    seed: int = 42



def _safe_parse_state(state_text: str):
    try:
        grid = np.array(ast.literal_eval(state_text), dtype=int)
        if grid.shape != (4, 4):
            return None
        return grid
    except Exception:
        return None



def _bucket_state(grid: np.ndarray, max_tile: int) -> str:
    empty = int(np.sum(grid == 0))

    if empty >= 8 and max_tile <= 64:
        return "easy"
    if empty <= 3 or max_tile >= 512:
        return "hard"
    return "medium"



def build_eval_sets(cfg: EvalSetConfig) -> Dict[str, int]:
    raw_dir = Path(cfg.raw_dir)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pools: Dict[str, List[Dict]] = {"easy": [], "medium": [], "hard": []}

    for json_file in sorted(raw_dir.glob("*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            game = json.load(f)

        states = game.get("states", [])
        for step in states:
            state_text = step.get("state", "")
            grid = _safe_parse_state(state_text)
            if grid is None:
                continue

            max_tile = int(np.max(grid))
            bucket = _bucket_state(grid, max_tile)
            pools[bucket].append(
                {
                    "bucket": bucket,
                    "state": state_text,
                    "start_score": int(step.get("score", 0)),
                    "source_game_id": game.get("game_id", json_file.stem),
                    "source_step": int(step.get("step", 0)),
                    "difficulty": game.get("difficulty", "unknown"),
                    "max_tile": max_tile,
                    "empty_cells": int(np.sum(grid == 0)),
                }
            )

    rng = np.random.default_rng(cfg.seed)
    counts: Dict[str, int] = {}

    ratio_sum = cfg.easy_ratio + cfg.medium_ratio + cfg.hard_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {ratio_sum}")

    targets = {
        "easy": int(cfg.total_size * cfg.easy_ratio),
        "medium": int(cfg.total_size * cfg.medium_ratio),
        "hard": int(cfg.total_size * cfg.hard_ratio),
    }
    # Allocate remainder to hard by default.
    remainder = cfg.total_size - sum(targets.values())
    targets["hard"] += remainder

    for bucket in ["easy", "medium", "hard"]:
        items = pools[bucket]
        if not items:
            counts[bucket] = 0
            continue

        target_n = max(0, targets[bucket])
        if len(items) > target_n:
            idx = rng.choice(len(items), size=target_n, replace=False)
            sampled = [items[i] for i in idx]
        else:
            sampled = items

        sampled = sorted(sampled, key=lambda x: (x["source_game_id"], x["source_step"]))

        path = out_dir / f"{bucket}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for row in sampled:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        counts[bucket] = len(sampled)

    manifest = {
        "manifest_version": "1.0",
        "raw_dir": cfg.raw_dir,
        "output_dir": cfg.output_dir,
        "total_size": cfg.total_size,
        "ratios": {
            "easy": cfg.easy_ratio,
            "medium": cfg.medium_ratio,
            "hard": cfg.hard_ratio,
        },
        "targets": targets,
        "seed": cfg.seed,
        "counts": counts,
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return counts



def main():
    parser = argparse.ArgumentParser(description="Build fixed eval sets for 2048")
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--output_dir", type=str, default="data/eval_sets")
    parser.add_argument("--total_size", type=int, default=1500)
    parser.add_argument("--easy_ratio", type=float, default=0.2)
    parser.add_argument("--medium_ratio", type=float, default=0.4)
    parser.add_argument("--hard_ratio", type=float, default=0.4)
    parser.add_argument("--per_set", type=int, default=None, help="旧参数兼容：总大小=per_set*3")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    total_size = args.total_size
    if args.per_set is not None:
        total_size = int(args.per_set) * 3

    cfg = EvalSetConfig(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        total_size=total_size,
        easy_ratio=args.easy_ratio,
        medium_ratio=args.medium_ratio,
        hard_ratio=args.hard_ratio,
        seed=args.seed,
    )

    counts = build_eval_sets(cfg)
    print("Eval sets built:")
    for k in ["easy", "medium", "hard"]:
        print(f"  {k}: {counts.get(k, 0)}")


if __name__ == "__main__":
    main()
