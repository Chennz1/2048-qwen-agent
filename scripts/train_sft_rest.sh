#!/bin/bash
# 现代RL训练Pipeline - SFT + ReST
# 默认启用优化: Unsloth + 4-bit + Flash Attention 2
# ======================= 参数说明区 =======================
# 用法:
#   bash scripts/train_sft_rest.sh [参数]
#
# 参数:
#   --monitor_backend <str>        监控后端。默认: wandb
#                                  可选值: wandb, tensorboard, none
#   --no_unsloth                   禁用 Unsloth（默认开启）
#   --no_4bit                      禁用 4-bit 量化（默认开启）
#   --no_flash_attn                禁用 Flash Attention（默认本就关闭）
#   --no_vllm                      评测阶段禁用 vLLM（默认开启）
#
# 说明:
#   本脚本当前将以下业务参数写死在脚本内部（需改脚本变量）:
#   BASE_MODEL/NUM_GAMES/DIFFICULTY/SFT_EPOCHS/REST_ITERATIONS/REST_GAMES_PER_ITER/EVAL_GAMES
#
# 示例:
#   bash scripts/train_sft_rest.sh --monitor_backend none --no_vllm
# ========================================================

set -e

echo "=========================================="
echo "  SFT + ReST 现代训练流程"
echo "=========================================="
echo ""
echo "ReST = Reinforcement Learning via Self-Play + 自训练"
echo "类似AlphaGo的方法，迭代改进"
echo ""
echo "⚡ 默认启用优化加速"
echo ""

# 配置
BASE_MODEL="Qwen/Qwen3-1.7B"
NUM_GAMES=10000
DIFFICULTY="mixed"
SFT_EPOCHS=3
SFT_OUTPUT="./checkpoints/sft"

# ReST配置
REST_ITERATIONS=5
REST_GAMES_PER_ITER=1000
REST_OUTPUT="./checkpoints/rest"

EVAL_GAMES=100

# 优化选项 (默认开启)
USE_UNSLOTH=true
LOAD_IN_4BIT=true
USE_FLASH_ATTN=false
USE_VLLM=true
VLLM_QUANT="int8"
MONITOR_BACKEND="wandb"

# 解析优化参数
while [[ $# -gt 0 ]]; do
    case $1 in
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
        --no_vllm)
            USE_VLLM=false
            shift
            ;;
        --monitor_backend)
            MONITOR_BACKEND="$2"
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

if [ ! -d "data/processed/train" ]; then
    echo "生成SFT训练数据..."
    bash scripts/generate_data.sh --num_games "$NUM_GAMES" --difficulty "$DIFFICULTY"
    python -m src.data.processor --use_thinking --validate --analyze
else
    echo "✓ 数据已存在"
fi

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
        --train_data "data/processed/train" \
        --val_data "data/processed/val" \
        --output_dir "$SFT_OUTPUT" \
        --epochs "$SFT_EPOCHS" \
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

# ===== 阶段2: ReST训练 =====
echo ""
echo "=========================================="
echo "阶段 2: ReST训练（迭代改进）"
echo "=========================================="
echo ""
echo "ReST算法:"
echo "  1. 用当前模型玩游戏"
echo "  2. 筛选高质量对局"
echo "  3. 用高质量对局微调"
echo "  4. 重复，模型越来越强"
echo ""

echo "继续执行 ReST 训练（非交互模式）..."

python -m src.models.modern_rl_trainer \
    --base_model "$SFT_OUTPUT" \
    --output_dir "$REST_OUTPUT" \
    --num_iterations "$REST_ITERATIONS" \
    --games_per_iteration "$REST_GAMES_PER_ITER" \
    --monitor_backend "$MONITOR_BACKEND" \
    $( [ "$USE_UNSLOTH" = true ] && echo "--use_unsloth" ) \
    $( [ "$LOAD_IN_4BIT" = true ] && echo "--load_in_4bit" ) \
    $( [ "$USE_FLASH_ATTN" = true ] && echo "--use_flash_attn" )

echo ""
echo "✅ ReST训练完成"

# 评估最终模型
echo ""
echo "评估ReST最终模型..."
FINAL_MODEL="$REST_OUTPUT/iteration_$REST_ITERATIONS"
bash scripts/evaluate.sh \
    --model_path "$FINAL_MODEL" \
    --base_model "$BASE_MODEL" \
    --num_games "$EVAL_GAMES" \
    --output_dir "data/eval/rest" \
    $( [ "$USE_VLLM" = false ] && echo "--no_vllm" )

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo ""
echo "📊 结果对比:"
echo "  SFT:  data/eval/sft/"
echo "  ReST: data/eval/rest/"
echo ""
echo "📈 训练历史:"
echo "  cat $REST_OUTPUT/training_history.json"
