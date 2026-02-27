#!/bin/bash
# 使用TRL库训练2048游戏模型
# 默认启用优化: Unsloth + 4-bit + Flash Attention 2
# ======================= 参数说明区 =======================
# 用法:
#   bash scripts/train.sh [参数]
#
# 参数:
#   --mode <str>                   训练模式。默认: sft
#                                  可选值: sft, rl
#   --base_model <name>            基座模型名。默认: Qwen/Qwen3-1.7B
#   --train_data <dir>             训练数据目录。默认: data/processed/train
#   --val_data <dir>               验证数据目录。默认: data/processed/val
#   --output_dir <dir>             模型输出目录。默认: ./checkpoints/sft
#   --epochs <int>                 训练轮数。默认: 3
#   --batch_size <int>             批次大小。默认: 8
#   --grad_accum <int>             梯度累计步数。默认: 4
#   --learning_rate <float>        学习率。默认: 5e-5
#   --monitor_backend <str>        监控后端。默认: wandb
#                                  可选值: wandb, tensorboard, none
#   --no_wandb                     关闭 wandb（并将 monitor_backend 置为 none）
#   --no_unsloth                   禁用 Unsloth 优化（默认开启）
#   --no_4bit                      禁用 4-bit 量化（默认开启）
#   --no_flash_attn                禁用 Flash Attention（默认本就关闭）
#
# 示例:
#   bash scripts/train.sh --mode sft --train_data data/processed/train --val_data data/processed/val --monitor_backend none
# ========================================================

set -e

# Hugging Face mirror defaults (can be overridden by pre-set env vars).
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_HOME"

echo "=========================================="
echo "  TRL训练 - 2048游戏"
echo "=========================================="

# 默认配置
BASE_MODEL="Qwen/Qwen3-1.7B"
MODE="sft"  # sft 或 rl
TRAIN_DATA="data/processed/train"
VAL_DATA="data/processed/val"
OUTPUT_DIR="./checkpoints/sft"
NUM_EPOCHS=3
BATCH_SIZE=8
GRAD_ACCUM=4
LR=5e-5
USE_WANDB=true
MONITOR_BACKEND="wandb"

# 优化选项 (默认开启)
USE_UNSLOTH=true
LOAD_IN_4BIT=true
USE_FLASH_ATTN=false

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --base_model)
            BASE_MODEL="$2"
            shift 2
            ;;
        --train_data)
            TRAIN_DATA="$2"
            shift 2
            ;;
        --val_data)
            VAL_DATA="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --grad_accum)
            GRAD_ACCUM="$2"
            shift 2
            ;;
        --learning_rate)
            LR="$2"
            shift 2
            ;;
        --no_wandb)
            USE_WANDB=false
            MONITOR_BACKEND="none"
            shift
            ;;
        --monitor_backend)
            MONITOR_BACKEND="$2"
            if [ "$MONITOR_BACKEND" = "none" ]; then
                USE_WANDB=false
            fi
            shift
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
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

echo ""
echo "训练配置:"
echo "  模式: $MODE"
echo "  基座模型: $BASE_MODEL"
echo "  训练数据: $TRAIN_DATA"
echo "  验证数据: $VAL_DATA"
echo "  输出目录: $OUTPUT_DIR"
echo "  训练轮数: $NUM_EPOCHS"
echo "  批次大小: $BATCH_SIZE"
echo "  梯度累计: $GRAD_ACCUM"
echo "  学习率: $LR"
echo "  WandB: $USE_WANDB"
echo "  监控后端: $MONITOR_BACKEND"
echo ""
echo "⚡ 优化配置:"
echo "  Unsloth: $USE_UNSLOTH"
echo "  4-bit量化: $LOAD_IN_4BIT"
echo "  Flash Attention 2: $USE_FLASH_ATTN"
echo ""

# 检查数据
if [ ! -d "$TRAIN_DATA" ]; then
    echo "❌ 错误: 找不到训练数据 $TRAIN_DATA"
    echo "请先运行: python -m src.data_gen.processor"
    exit 1
fi

# 训练
echo "🚀 开始训练..."
python -m src.models.trl_train \
    --mode "$MODE" \
    --model "$BASE_MODEL" \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --grad_accum "$GRAD_ACCUM" \
    --lr "$LR" \
    --monitor_backend "$MONITOR_BACKEND" \
    $( [ "$USE_WANDB" = false ] && echo "--no_wandb" ) \
    $( [ "$USE_UNSLOTH" = true ] && echo "--use_unsloth" ) \
    $( [ "$LOAD_IN_4BIT" = true ] && echo "--load_in_4bit" ) \
    $( [ "$USE_FLASH_ATTN" = true ] && echo "--use_flash_attn" )

echo ""
echo "✅ 训练完成！"
echo "📍 模型保存在: $OUTPUT_DIR"
