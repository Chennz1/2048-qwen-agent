#!/bin/bash
# Run required three evaluation capabilities:
# 1) rule baseline
# 2) random baseline
# 3) LLM thinking ON/OFF
# ======================= 参数说明区 =======================
# 用法:
#   bash scripts/evaluate_three_agents.sh [参数]
#
# 参数:
#   --model_path <path>            LLM 评测模型路径（评测 LoRA/SFT/RL 模型时使用）
#   --base_model <name>            基座模型名。默认: Qwen/Qwen3-1.7B
#   --is_base_model                将 LLM 直接作为 base model 评测（无需 --model_path）
#   --seed <int>                   随机种子。默认: 42
#   --out_root <dir>               输出根目录。默认: data/eval_agents
#   --no_quick                     关闭 quick 模式（默认 quick=true）
#   --visualize_game               打开单局可视化（会降低速度）
#   --skip_llm                     仅跑 rule/random baseline，不跑 LLM
#
# 说明:
#   1) 默认会跑 rule + random + LLM(think on/off)
#   2) 若未设置 --skip_llm，则必须提供 --model_path 或 --is_base_model
#
# 示例:
#   bash scripts/evaluate_three_agents.sh --model_path checkpoints/sft --seed 42
# ========================================================

set -e

# Hugging Face mirror defaults (can be overridden by pre-set env vars).
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_HOME"

MODEL_PATH=""
BASE_MODEL="Qwen/Qwen3-1.7B"
IS_BASE_MODEL=false
SEED=42
OUT_ROOT="data/eval_agents"
QUICK=true
VISUALIZE=false
SKIP_LLM=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --model_path)
      MODEL_PATH="$2"; shift 2 ;;
    --base_model)
      BASE_MODEL="$2"; shift 2 ;;
    --is_base_model)
      IS_BASE_MODEL=true; shift ;;
    --seed)
      SEED="$2"; shift 2 ;;
    --out_root)
      OUT_ROOT="$2"; shift 2 ;;
    --no_quick)
      QUICK=false; shift ;;
    --visualize_game)
      VISUALIZE=true; shift ;;
    --skip_llm)
      SKIP_LLM=true; shift ;;
    *)
      echo "Unknown arg: $1"; exit 1 ;;
  esac
done

mkdir -p "$OUT_ROOT"

COMMON_ARGS=(--seed "$SEED")
if [ "$QUICK" = true ]; then
  COMMON_ARGS+=(--quick)
fi
if [ "$VISUALIZE" = true ]; then
  COMMON_ARGS+=(--visualize_game --visualize_delay 0.05)
fi

echo "=== 1) Rule baseline eval ==="
python -m src.eval.agent_eval \
  --agent_type rule \
  --rule_difficulty advanced \
  --rule_with_thinking \
  --output_dir "$OUT_ROOT/rule_baseline" \
  "${COMMON_ARGS[@]}"

echo "=== 2) Random baseline eval ==="
python -m src.eval.agent_eval \
  --agent_type random \
  --output_dir "$OUT_ROOT/random_baseline" \
  "${COMMON_ARGS[@]}"

if [ "$SKIP_LLM" = true ]; then
  echo "=== 3) LLM eval skipped (--skip_llm) ==="
  exit 0
fi

if [ "$IS_BASE_MODEL" = false ] && [ -z "$MODEL_PATH" ]; then
  echo "Error: LLM eval requires --model_path or --is_base_model"
  exit 1
fi

LLM_COMMON=(--agent_type llm --base_model "$BASE_MODEL" --output_dir "$OUT_ROOT/llm_thinking_on" "${COMMON_ARGS[@]}")
if [ -n "$MODEL_PATH" ]; then
  LLM_COMMON+=(--model_path "$MODEL_PATH")
fi
if [ "$IS_BASE_MODEL" = true ]; then
  LLM_COMMON+=(--is_base_model)
fi

echo "=== 3a) LLM eval (thinking ON) ==="
python -m src.eval.agent_eval \
  "${LLM_COMMON[@]}" \
  --use_thinking

LLM_OFF_COMMON=(--agent_type llm --base_model "$BASE_MODEL" --output_dir "$OUT_ROOT/llm_thinking_off" "${COMMON_ARGS[@]}")
if [ -n "$MODEL_PATH" ]; then
  LLM_OFF_COMMON+=(--model_path "$MODEL_PATH")
fi
if [ "$IS_BASE_MODEL" = true ]; then
  LLM_OFF_COMMON+=(--is_base_model)
fi

echo "=== 3b) LLM eval (thinking OFF) ==="
python -m src.eval.agent_eval \
  "${LLM_OFF_COMMON[@]}" \
  --no_thinking

echo "=== All requested evaluations finished ==="
