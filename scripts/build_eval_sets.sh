#!/bin/bash
# Build fixed easy/medium/hard eval sets from raw data.
# ======================= 参数说明区 =======================
# 用法:
#   bash scripts/build_eval_sets.sh [参数]
#
# 参数:
#   --raw_dir <dir>
#     原始对局目录。默认: data/raw
#   --output_dir <dir>
#     固定评测集输出目录。默认: data/eval_sets
#   --per_set <int>
#     每个难度(easy/medium/hard)的样本数。默认: 500
#   --total_size <int>
#     三个难度总样本数；设置后会覆盖 --per_set 的行为。
#   --easy_ratio <float>
#   --medium_ratio <float>
#   --hard_ratio <float>
#     三个难度占比，通常三者之和应为 1.0。默认: 0.2/0.4/0.4
#   --seed <int>
#     随机种子。默认: 42
#
# 示例:
#   bash scripts/build_eval_sets.sh --raw_dir data/raw --output_dir data/eval_sets --total_size 1500 --seed 42
# ========================================================

set -e

RAW_DIR="data/raw"
OUTPUT_DIR="data/eval_sets"
PER_SET=500
TOTAL_SIZE=""
EASY_RATIO=0.2
MEDIUM_RATIO=0.4
HARD_RATIO=0.4
SEED=42

while [[ $# -gt 0 ]]; do
  case $1 in
    --raw_dir)
      RAW_DIR="$2"; shift 2 ;;
    --output_dir)
      OUTPUT_DIR="$2"; shift 2 ;;
    --per_set)
      PER_SET="$2"; shift 2 ;;
    --total_size)
      TOTAL_SIZE="$2"; shift 2 ;;
    --easy_ratio)
      EASY_RATIO="$2"; shift 2 ;;
    --medium_ratio)
      MEDIUM_RATIO="$2"; shift 2 ;;
    --hard_ratio)
      HARD_RATIO="$2"; shift 2 ;;
    --seed)
      SEED="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1"; exit 1 ;;
  esac
done

CMD=(python -m src.eval.eval_set_builder
  --raw_dir "$RAW_DIR"
  --output_dir "$OUTPUT_DIR"
  --easy_ratio "$EASY_RATIO"
  --medium_ratio "$MEDIUM_RATIO"
  --hard_ratio "$HARD_RATIO"
  --seed "$SEED"
)

if [ -n "$TOTAL_SIZE" ]; then
  CMD+=(--total_size "$TOTAL_SIZE")
else
  CMD+=(--per_set "$PER_SET")
fi

"${CMD[@]}"
