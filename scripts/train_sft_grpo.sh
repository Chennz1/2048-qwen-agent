#!/bin/bash
# SFT + GRPO 训练Pipeline
# 11GB 显存友好默认配置（可通过参数覆盖）
# ======================= 参数说明区 =======================
# 用法:
#   bash scripts/train_sft_grpo.sh [参数]
#
# 参数:
#   --monitor_backend <str>        监控后端。默认: wandb
#                                  可选值: wandb, tensorboard, none
#   --grpo_num_generations <int>   GRPO每个prompt采样条数K。默认: 2
#                                  约束: global_train_batch_size(单卡即batch_size)必须能被K整除
#                                  单卡11GB建议: batch_size=2, K=2
#   --use_unsloth                  启用 Unsloth（默认关闭，11GB更稳）
#   --no_unsloth                   禁用 Unsloth
#   --no_4bit                      禁用 4-bit 量化（默认开启）
#   --no_flash_attn                禁用 Flash Attention（默认本就关闭）
#   --use_vllm                     评测阶段启用 vLLM（默认关闭，11GB更稳）
#   --no_vllm                      评测阶段禁用 vLLM
# 说明:
#   默认配置优先保证 11GB 显存可跑通 SFT+GRPO，而不是追求速度或最佳效果。
#   且采用“轻SFT + 重RL”策略：
#   1) SFT: expert CoT + 固定抽样 1000 条，仅做冷启动
#   2) GRPO: 使用更多样本做策略优化
#   其余训练规模参数当前仍是脚本内固定值（见下方变量区）。
#
# 示例:
#   bash scripts/train_sft_grpo.sh --monitor_backend none --no_vllm
# ========================================================

set -e

# Hugging Face mirror defaults (can be overridden by pre-set env vars).
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_HOME"

echo "=========================================="
echo "  SFT + GRPO 训练"
echo "=========================================="
echo ""
echo "GRPO = Group Relative Policy Optimization"
echo "现代RL算法 (2020年代)"
echo ""
echo "⚡ 默认使用 11GB 显存友好配置"
echo "HF_ENDPOINT=$HF_ENDPOINT"
echo "HF_HOME=$HF_HOME"
echo ""

# 配置
BASE_MODEL="Qwen/Qwen3-1.7B"
# 数据策略：默认使用 expert CoT 数据
NUM_GAMES=100
RAW_DIR="data/raw_expert"
PROCESSED_DIR="data/processed_expert"
SFT_TRAIN_SAMPLES=4096
SFT_SUBSET_SEED=42
SFT_TRAIN_DIR="data/processed_expert_sft10k/train"
EXPERT_DEPTH=2
EXPERT_MAX_EMPTY=8

SFT_EPOCHS=1
SFT_BATCH_SIZE=16
SFT_GRAD_ACCUM=4
SFT_OUTPUT="./checkpoints/sft"

# GRPO配置（TRL GRPOTrainer）
GRPO_EPOCHS=1
GRPO_NUM_SAMPLES=8192
GRPO_BATCH_SIZE=8
GRPO_GRAD_ACCUM=4
GRPO_NUM_GENERATIONS=4
GRPO_MAX_PROMPT_LENGTH=512
GRPO_MAX_COMPLETION_LENGTH=512
GRPO_OUTPUT="./checkpoints/grpo"

EVAL_GAMES=32

# 优化选项（11GB 友好默认）
USE_UNSLOTH=false
LOAD_IN_4BIT=true
USE_FLASH_ATTN=false
USE_VLLM=true
VLLM_QUANT=""
MONITOR_BACKEND="wandb"

# 解析优化参数
while [[ $# -gt 0 ]]; do
    case $1 in
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
        --monitor_backend)
            MONITOR_BACKEND="$2"
            shift 2
            ;;
        --grpo_num_generations)
            GRPO_NUM_GENERATIONS="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# ===== 阶段0: 数据准备 =====
echo "=========================================="
echo "阶段 0: 数据准备"
echo "=========================================="

if [ ! -d "$PROCESSED_DIR/train" ]; then
    echo "生成 expert CoT SFT 冷启动数据..."
    bash scripts/generate_expert_cot_data.sh \
        --num_games "$NUM_GAMES" \
        --expert_depth "$EXPERT_DEPTH" \
        --expert_max_empty "$EXPERT_MAX_EMPTY" \
        --output_dir "$RAW_DIR"
    python -m src.data_gen.processor \
        --input_dir "$RAW_DIR" \
        --output_dir "$PROCESSED_DIR" \
        --use_thinking \
        --validate \
        --analyze
else
    echo "✓ 数据已存在"
fi

if [ ! -d "$RAW_DIR" ]; then
    echo "raw 数据不存在，补充生成..."
    bash scripts/generate_expert_cot_data.sh \
        --num_games "$NUM_GAMES" \
        --expert_depth "$EXPERT_DEPTH" \
        --expert_max_empty "$EXPERT_MAX_EMPTY" \
        --output_dir "$RAW_DIR"
fi

echo "构建固定 SFT 训练子集: ${SFT_TRAIN_SAMPLES} 条..."
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
print(f"SFT train subset saved: {dst} | original={orig}, used={len(ds)}")
PY

# ===== 阶段1: SFT训练 =====
echo ""
echo "=========================================="
echo "阶段 1: SFT训练（初始化）"
echo "=========================================="
echo "优化: Unsloth=$USE_UNSLOTH, 4bit=$LOAD_IN_4BIT, FlashAttn=$USE_FLASH_ATTN"
echo "监控后端: $MONITOR_BACKEND"

if [ ! -d "$SFT_OUTPUT" ]; then
    python -m src.models.trl_train \
        --mode sft \
        --model "$BASE_MODEL" \
        --train_data "$SFT_TRAIN_DIR" \
        --output_dir "$SFT_OUTPUT" \
        --epochs "$SFT_EPOCHS" \
        --batch_size "$SFT_BATCH_SIZE" \
        --grad_accum "$SFT_GRAD_ACCUM" \
        --monitor_backend "$MONITOR_BACKEND" \
        $( [ "$USE_UNSLOTH" = true ] && echo "--use_unsloth" ) \
        $( [ "$LOAD_IN_4BIT" = true ] && echo "--load_in_4bit" ) \
        $( [ "$USE_FLASH_ATTN" = true ] && echo "--use_flash_attn" )

    echo "✅ SFT训练完成"
else
    echo "✓ SFT模型已存在"
fi

# 评估SFT
echo ""
echo "评估SFT模型..."
bash scripts/evaluate.sh \
    --model_path "$SFT_OUTPUT" \
    --base_model "$BASE_MODEL" \
    --num_games "$EVAL_GAMES" \
    --output_dir "data/eval/sft" \
    $( [ "$USE_VLLM" = false ] && echo "--no_vllm" )

# ===== 阶段2: GRPO训练 =====
echo ""
echo "=========================================="
echo "阶段 2: GRPO训练（在线RL）"
echo "=========================================="
echo ""
echo "GRPO算法特点:"
echo "  1. Group采样: 每个状态采样N个动作"
echo "  2. Relative优势: 相对于Group平均"
echo "  3. 无需Critic: 不需要价值函数"
echo ""

echo "继续执行 GRPO 训练（非交互模式）..."

echo "预检查 TRL GRPO API 可用性..."
python - <<'PY'
import importlib
from importlib.metadata import PackageNotFoundError, version

try:
    importlib.import_module("trl").GRPOConfig  # type: ignore[attr-defined]
    importlib.import_module("trl").GRPOTrainer  # type: ignore[attr-defined]
    print("TRL GRPO symbols found in trl top-level.")
except Exception:
    try:
        importlib.import_module("trl.trainer.grpo_config").GRPOConfig
        importlib.import_module("trl.trainer.grpo_trainer").GRPOTrainer
        print("TRL GRPO symbols found in trl.trainer submodules.")
    except Exception as exc:
        try:
            v = version("trl")
        except (PackageNotFoundError, Exception):
            v = "not_installed"
        raise SystemExit(
            f"GRPO preflight failed: trl={v} does not expose GRPO API. "
            "Please run: pip install -U \"trl>=0.15.0\". "
            f"Inner error: {type(exc).__name__}: {exc}"
        )
PY

# GRPO 训练固定从 raw 轨迹在线构造 prompt-only 数据集。
python -m src.models.grpo \
    --model "$SFT_OUTPUT" \
    --output_dir "$GRPO_OUTPUT" \
    --input_dir "$RAW_DIR" \
    --num_samples "$GRPO_NUM_SAMPLES" \
    --epochs "$GRPO_EPOCHS" \
    --batch_size "$GRPO_BATCH_SIZE" \
    --grad_accum "$GRPO_GRAD_ACCUM" \
    --num_generations "$GRPO_NUM_GENERATIONS" \
    --max_prompt_length "$GRPO_MAX_PROMPT_LENGTH" \
    --max_completion_length "$GRPO_MAX_COMPLETION_LENGTH" \
    --monitor_backend "$MONITOR_BACKEND"

echo ""
echo "✅ GRPO训练完成"

# 评估GRPO
echo ""
echo "评估GRPO最终模型..."
bash scripts/evaluate.sh \
    --model_path "$GRPO_OUTPUT" \
    --base_model "$BASE_MODEL" \
    --num_games "$EVAL_GAMES" \
    --output_dir "data/eval/grpo" \
    $( [ "$USE_VLLM" = false ] && echo "--no_vllm" )

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo ""
echo "📊 结果对比:"
echo "  SFT:  data/eval/sft/"
echo "  GRPO: data/eval/grpo/"
echo ""
echo "💡 与ReST对比:"
echo "  bash scripts/train_sft_rest.sh"
echo ""
echo "📈 查看实操说明:"
echo "  cat docs/PRACTICAL_DATA_AND_TOKENIZER_GUIDE.md"
