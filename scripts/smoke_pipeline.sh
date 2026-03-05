#!/bin/bash
# Minimal SFT+GRPO smoke pipeline aligned with train_sft_grpo.sh.
# ======================= 参数说明区 =======================
# 用法:
#   bash scripts/smoke_pipeline.sh [参数]
#
# 参数:
#   --base_model <name>            基座模型名。默认: Qwen/Qwen3-1.7B
#   --monitor_backend <str>        监控后端。默认: none
#                                  可选值: wandb, tensorboard, none
#   --num_games <int>              expert CoT 原始对局数量。默认: 20
#   --sft_samples <int>            SFT 训练子集条数。默认: 128
#   --grpo_samples <int>           GRPO 样本条数。默认: 256
#   --grpo_num_generations <int>   GRPO每个prompt采样条数K。默认: 2
#                                  约束: global_train_batch_size(单卡即batch_size)必须能被K整除
#   --eval_games <int>             SFT/GRPO 各自评测局数。默认: 2
#   --use_unsloth                  启用 Unsloth（默认关闭）
#   --no_unsloth                   禁用 Unsloth
#   --no_4bit                      禁用 4-bit 量化（默认开启）
#   --no_flash_attn                禁用 Flash Attention（默认关闭）
#   --use_vllm                     评测阶段启用 vLLM（默认关闭）
#   --no_vllm                      评测阶段禁用 vLLM
#
# 说明:
#   该脚本与 train_sft_grpo.sh 的训练设置保持一致（SFT/GRPO核心超参），
#   仅把数据规模压缩到“可行性验证”级别。
# ========================================================

set -e

# Hugging Face mirror defaults (can be overridden by pre-set env vars).
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_HOME"

BASE_MODEL="Qwen/Qwen3-1.7B"
MONITOR_BACKEND="none"

RAW_DIR="data/raw_expert_smoke"
PROCESSED_DIR="data/processed_expert_smoke"
SFT_TRAIN_DIR="data/processed_expert_smoke_sft/train"
SFT_DIR="checkpoints/sft_smoke"
GRPO_DIR="checkpoints/grpo_smoke"
EVAL_ROOT="data/eval/smoke_sft_grpo"

NUM_GAMES=20
SFT_TRAIN_SAMPLES=512
SFT_SUBSET_SEED=42
EXPERT_DEPTH=2
EXPERT_MAX_EMPTY=8

SFT_EPOCHS=1
SFT_BATCH_SIZE=16
SFT_GRAD_ACCUM=1

GRPO_EPOCHS=1
GRPO_NUM_SAMPLES=512
GRPO_GRAD_ACCUM=4
GRPO_BATCH_SIZE=8
GRPO_NUM_GENERATIONS=8
GRPO_MAX_PROMPT_LENGTH=600
GRPO_MAX_COMPLETION_LENGTH=768

EVAL_GAMES=32

USE_UNSLOTH=false
LOAD_IN_4BIT=true
USE_FLASH_ATTN=false
USE_VLLM=true

while [[ $# -gt 0 ]]; do
  case $1 in
    --base_model)
      BASE_MODEL="$2"
      shift 2
      ;;
    --monitor_backend)
      MONITOR_BACKEND="$2"
      shift 2
      ;;
    --num_games)
      NUM_GAMES="$2"
      shift 2
      ;;
    --sft_samples)
      SFT_TRAIN_SAMPLES="$2"
      shift 2
      ;;
    --grpo_samples)
      GRPO_NUM_SAMPLES="$2"
      shift 2
      ;;
    --grpo_num_generations)
      GRPO_NUM_GENERATIONS="$2"
      shift 2
      ;;
    --eval_games)
      EVAL_GAMES="$2"
      shift 2
      ;;
    --use_unsloth)
      USE_UNSLOTH=true
      shift
      ;;
    --no_unsloth)
      USE_UNSLOTH=false
      shift
      ;;
    --no_4bit)
      LOAD_IN_4BIT=false
      shift
      ;;
    --no_flash_attn)
      USE_FLASH_ATTN=false
      shift
      ;;
    --use_vllm)
      USE_VLLM=true
      shift
      ;;
    --no_vllm)
      USE_VLLM=false
      shift
      ;;
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
done

echo "=== Smoke SFT+GRPO Pipeline ==="
echo "HF_ENDPOINT=$HF_ENDPOINT"
echo "HF_HOME=$HF_HOME"
echo "base_model=$BASE_MODEL"
echo "monitor_backend=$MONITOR_BACKEND"
echo "num_games=$NUM_GAMES"
echo "sft_samples=$SFT_TRAIN_SAMPLES"
echo "grpo_samples=$GRPO_NUM_SAMPLES"
echo "grpo_num_generations=$GRPO_NUM_GENERATIONS"
echo "eval_games=$EVAL_GAMES"

# rm -rf "$RAW_DIR" "$PROCESSED_DIR" "$SFT_TRAIN_DIR" "$SFT_DIR" "$GRPO_DIR" "$EVAL_ROOT"

echo "==> [1/7] generate expert CoT raw data"
bash scripts/generate_expert_cot_data.sh \
  --num_games "$NUM_GAMES" \
  --expert_depth "$EXPERT_DEPTH" \
  --expert_max_empty "$EXPERT_MAX_EMPTY" \
  --output_dir "$RAW_DIR"

echo "==> [2/7] process raw -> processed"
python -m src.data_gen.processor \
  --input_dir "$RAW_DIR" \
  --output_dir "$PROCESSED_DIR" \
  --use_thinking \
  --validate

echo "==> [3/7] build fixed SFT subset"
python - <<PY
from datasets import load_from_disk
from pathlib import Path
import shutil

src = Path("${PROCESSED_DIR}/train")
dst = Path("${SFT_TRAIN_DIR}")
n = int("${SFT_TRAIN_SAMPLES}")
seed = int("${SFT_SUBSET_SEED}")

ds = load_from_disk(str(src))
orig = len(ds)
if len(ds) > n:
    ds = ds.shuffle(seed=seed).select(range(n))

if dst.exists():
    shutil.rmtree(dst)
dst.parent.mkdir(parents=True, exist_ok=True)
ds.save_to_disk(str(dst))
print(f"SFT subset saved: {dst} | original={orig}, used={len(ds)}")
PY

echo "==> [4/7] train SFT"
python -m src.models.trl_train \
  --mode sft \
  --model "$BASE_MODEL" \
  --train_data "$SFT_TRAIN_DIR" \
  --val_data "$PROCESSED_DIR/val" \
  --output_dir "$SFT_DIR" \
  --epochs "$SFT_EPOCHS" \
  --batch_size "$SFT_BATCH_SIZE" \
  --grad_accum "$SFT_GRAD_ACCUM" \
  --monitor_backend "$MONITOR_BACKEND" \
  $( [ "$USE_UNSLOTH" = true ] && echo "--use_unsloth" ) \
  $( [ "$LOAD_IN_4BIT" = true ] && echo "--load_in_4bit" ) \
  $( [ "$USE_FLASH_ATTN" = true ] && echo "--use_flash_attn" )


echo "==> [7/8] train GRPO (raw -> prompt-only)"
python -m src.models.grpo \
  --model "$SFT_DIR" \
  --output_dir "$GRPO_DIR" \
  --input_dir "$RAW_DIR" \
  --num_samples "$GRPO_NUM_SAMPLES" \
  --epochs "$GRPO_EPOCHS" \
  --batch_size "$GRPO_BATCH_SIZE" \
  --grad_accum "$GRPO_GRAD_ACCUM" \
  --num_generations "$GRPO_NUM_GENERATIONS" \
  --max_prompt_length "$GRPO_MAX_PROMPT_LENGTH" \
  --max_completion_length "$GRPO_MAX_COMPLETION_LENGTH" \
  --monitor_backend "$MONITOR_BACKEND"

echo "==> [8/8] eval GRPO"
bash scripts/evaluate.sh \
  --model_path "$GRPO_DIR" \
  --base_model "$BASE_MODEL" \
  --num_games "$EVAL_GAMES" \
  --output_dir "$EVAL_ROOT/grpo" \
  $( [ "$USE_VLLM" = false ] && echo "--no_vllm" )

echo "Smoke SFT+GRPO pipeline completed successfully."
