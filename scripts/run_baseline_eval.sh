#!/bin/bash
# Run baseline eval on fixed easy/medium/hard sets and aggregate results.
# ======================= 参数说明区 =======================
# 用法:
#   bash scripts/run_baseline_eval.sh --model_path <path> [参数]
#
# 参数:
#   --model_path <path>            (必填) 待评测模型目录
#   --base_model <name>            基座模型名。默认: Qwen/Qwen3-1.7B
#   --eval_set_dir <dir>           easy/medium/hard jsonl 所在目录。默认: data/eval_sets
#   --output_dir <dir>             输出目录。默认: data/eval/baseline
#   --max_steps <int>              单局最大步数。默认: 1000
#   --temperature <float>          采样温度。默认: 0.1
#   --seed <int>                   随机种子。默认: 不固定
#   --no_vllm                      禁用 vLLM
#   --vllm_quant <str>             vLLM 量化方式。默认: int8
#                                  可选值: int8, awq, gptq
#   --no_visualize                 关闭终端可视化输出
#
# 输出:
#   1) <output_dir>/easy|medium|hard/eval_results.json
#   2) <output_dir>/baseline_results.json (聚合结果)
#
# 示例:
#   bash scripts/run_baseline_eval.sh --model_path checkpoints/sft --eval_set_dir data/eval_sets --seed 42
# ========================================================

set -e

MODEL_PATH=""
BASE_MODEL="Qwen/Qwen3-1.7B"
EVAL_SET_DIR="data/eval_sets"
OUTPUT_DIR="data/eval/baseline"
MAX_STEPS=1000
TEMP=0.1
SEED=""
USE_VLLM=true
VLLM_QUANT="int8"
NO_VISUALIZE=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --model_path)
      MODEL_PATH="$2"; shift 2 ;;
    --base_model)
      BASE_MODEL="$2"; shift 2 ;;
    --eval_set_dir)
      EVAL_SET_DIR="$2"; shift 2 ;;
    --output_dir)
      OUTPUT_DIR="$2"; shift 2 ;;
    --max_steps)
      MAX_STEPS="$2"; shift 2 ;;
    --temperature)
      TEMP="$2"; shift 2 ;;
    --seed)
      SEED="$2"; shift 2 ;;
    --no_vllm)
      USE_VLLM=false; shift ;;
    --vllm_quant)
      VLLM_QUANT="$2"; shift 2 ;;
    --no_visualize)
      NO_VISUALIZE=true; shift ;;
    *)
      echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [ -z "$MODEL_PATH" ]; then
  echo "Error: --model_path is required"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

for split in easy medium hard; do
  set_path="$EVAL_SET_DIR/$split.jsonl"
  if [ ! -f "$set_path" ]; then
    echo "Missing eval set: $set_path"
    exit 1
  fi

  bash scripts/evaluate.sh \
    --model_path "$MODEL_PATH" \
    --base_model "$BASE_MODEL" \
    --output_dir "$OUTPUT_DIR/$split" \
    --eval_set "$set_path" \
    --num_games 1 \
    --max_steps "$MAX_STEPS" \
    --temperature "$TEMP" \
    $( [ -n "$SEED" ] && echo "--seed $SEED" ) \
    $( [ "$NO_VISUALIZE" = true ] && echo "--no_visualize" ) \
    $( [ "$USE_VLLM" = false ] && echo "--no_vllm" ) \
    $( [ "$USE_VLLM" = true ] && echo "--vllm_quant $VLLM_QUANT" )
done

export BASELINE_OUT="$OUTPUT_DIR"

python - <<'PY'
import json
from pathlib import Path

out = Path("data/eval/baseline")
# if OUTPUT_DIR changed, infer from env fallback
import os
out = Path(os.environ.get("BASELINE_OUT", str(out)))

summary = {"splits": {}}
for split in ["easy", "medium", "hard"]:
    p = out / split / "eval_results.json"
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    summary["splits"][split] = {
        "mean_score": data["mean_score"],
        "mean_max_tile": data["mean_max_tile"],
        "legal_move_rate": data["legal_move_rate"],
        "num_games": data["num_games"],
    }

total_games = sum(v["num_games"] for v in summary["splits"].values())
if total_games == 0:
    raise ValueError("No games found in split results")

summary["overall"] = {
    "mean_score": sum(v["mean_score"] * v["num_games"] for v in summary["splits"].values()) / total_games,
    "mean_max_tile": sum(v["mean_max_tile"] * v["num_games"] for v in summary["splits"].values()) / total_games,
    "legal_move_rate": sum(v["legal_move_rate"] * v["num_games"] for v in summary["splits"].values()) / total_games,
    "num_games": total_games,
}

with open(out / "baseline_results.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("Saved baseline summary:", out / "baseline_results.json")
PY
