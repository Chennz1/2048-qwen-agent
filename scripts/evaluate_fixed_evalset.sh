#!/usr/bin/env bash
set -euo pipefail

# Compare base / SFT / GRPO on a predefined fixed eval set.
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

BASE_MODEL="Qwen/Qwen3-1.7B"
SFT_MODEL_PATH=""
GRPO_MODEL_PATH=""

EVAL_SET="data/eval_sets/all.jsonl"
MAX_STEPS=1000
BATCH_SIZE=8
TEMPERATURE="0.1"
PRESENCE_PENALTY="0.0"
SEED="42"

USE_THINKING=false
USE_VLLM=false
VLLM_QUANT=""
OUTPUT_ROOT="data/eval/fixed_compare_$(date +%Y%m%d_%H%M%S)"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/evaluate_fixed_evalset.sh --sft_model_path <path> --grpo_model_path <path> [options]

Required:
  --sft_model_path <path>       SFT checkpoint path
  --grpo_model_path <path>      GRPO checkpoint path

Options:
  --base_model <name>           Base model name (default: Qwen/Qwen3-1.7B)
  --eval_set <path>             Fixed eval set jsonl (default: data/eval_sets/all.jsonl)
  --max_steps <int>             Max steps per game (default: 1000)
  --batch_size <int>            Parallel inference batch size (default: 8)
  --temperature <float>         Sampling temperature (default: 0.1)
  --presence_penalty <float>    Presence penalty (default: 0.0)
  --seed <int>                  Seed (default: 42)
  --use_thinking                Enable thinking mode
  --no_thinking                 Disable thinking mode (default)
  --use_vllm                    Enable vLLM
  --no_vllm                     Disable vLLM (default)
  --vllm_quant <str>            vLLM quantization (default: none)
  --output_root <dir>           Output root dir
  -h, --help                    Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base_model)
      BASE_MODEL="$2"; shift 2 ;;
    --sft_model_path)
      SFT_MODEL_PATH="$2"; shift 2 ;;
    --grpo_model_path)
      GRPO_MODEL_PATH="$2"; shift 2 ;;
    --eval_set)
      EVAL_SET="$2"; shift 2 ;;
    --max_steps)
      MAX_STEPS="$2"; shift 2 ;;
    --batch_size)
      BATCH_SIZE="$2"; shift 2 ;;
    --temperature)
      TEMPERATURE="$2"; shift 2 ;;
    --presence_penalty)
      PRESENCE_PENALTY="$2"; shift 2 ;;
    --seed)
      SEED="$2"; shift 2 ;;
    --use_thinking)
      USE_THINKING=true; shift ;;
    --no_thinking)
      USE_THINKING=false; shift ;;
    --use_vllm)
      USE_VLLM=true; shift ;;
    --no_vllm)
      USE_VLLM=false; shift ;;
    --vllm_quant)
      VLLM_QUANT="$2"; shift 2 ;;
    --output_root)
      OUTPUT_ROOT="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1 ;;
  esac
done

if [[ -z "$SFT_MODEL_PATH" ]]; then
  echo "Error: --sft_model_path is required"
  exit 1
fi
if [[ -z "$GRPO_MODEL_PATH" ]]; then
  echo "Error: --grpo_model_path is required"
  exit 1
fi
if [[ ! -f "$EVAL_SET" ]]; then
  echo "Error: eval set not found: $EVAL_SET"
  echo "Tip: build one by running:"
  echo "  bash scripts/build_eval_sets.sh --raw_dir data/raw --output_dir data/eval_sets --total_size 1500 --seed 42"
  echo "  cat data/eval_sets/easy.jsonl data/eval_sets/medium.jsonl data/eval_sets/hard.jsonl > data/eval_sets/all.jsonl"
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"

COMMON_ARGS=(
  --base_model "$BASE_MODEL"
  --eval_set "$EVAL_SET"
  --max_steps "$MAX_STEPS"
  --batch_size "$BATCH_SIZE"
  --temperature "$TEMPERATURE"
  --presence_penalty "$PRESENCE_PENALTY"
  --seed "$SEED"
  --no_visualize
)

if [[ "$USE_THINKING" == true ]]; then
  COMMON_ARGS+=(--use_thinking)
else
  COMMON_ARGS+=(--no_thinking)
fi

if [[ "$USE_VLLM" == true ]]; then
  COMMON_ARGS+=(--use_vllm)
  if [[ -n "$VLLM_QUANT" ]]; then
    COMMON_ARGS+=(--vllm_quantization "$VLLM_QUANT")
  fi
fi

run_eval() {
  local model_name="$1"
  shift
  local out_dir="$OUTPUT_ROOT/$model_name"
  mkdir -p "$out_dir"
  echo ""
  echo "==> Evaluating $model_name on fixed eval set"
  python -m src.eval.evaluator \
    "${COMMON_ARGS[@]}" \
    --output_dir "$out_dir" \
    "$@"
}

run_eval "base" --is_base_model
run_eval "sft" --model_path "$SFT_MODEL_PATH"
run_eval "grpo" --model_path "$GRPO_MODEL_PATH"

python - "$OUTPUT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
models = ["base", "sft", "grpo"]
keys = ["mean_score", "mean_max_tile", "legal_move_rate", "max_tile_reached", "mean_steps"]

print("\n=== Fixed Eval Set Summary ===")
for m in models:
    p = root / m / "eval_results.json"
    if not p.exists():
        print(f"{m:>5}: missing ({p})")
        continue
    data = json.loads(p.read_text(encoding="utf-8"))
    row = ", ".join(f"{k}={data.get(k)}" for k in keys)
    print(f"{m:>5}: {row}")
PY

echo ""
echo "Done. Outputs in: $OUTPUT_ROOT"
