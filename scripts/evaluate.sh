#!/bin/bash
# Evaluate 2048 game model
# 默认使用 BF16（不使用 bitsandbytes 量化）
# ======================= 参数说明区 =======================
# 用法:
#   bash scripts/evaluate.sh --model_path <path> [参数]
#
# 参数:
#   --model_path <path>            (必填) 待评测模型目录
#   --base_model <name>            基座模型名。默认: Qwen/Qwen3-1.7B
#   --num_games <int>              评测局数。默认: 100
#   --max_steps <int>              单局最大步数。默认: 1000
#   --batch_size <int>             并行推理批大小。默认: min(num_games, 8)
#   --temperature <float>          采样温度覆盖（默认自动: thinking=0.6, non-thinking=0.7）
#   --presence_penalty <float>     存在惩罚（支持框架可用）。默认: 0（建议范围 0~2）
#   --use_thinking                 使用 thinking 模式（默认开启）
#   --no_thinking                  关闭 thinking 模式
#   --seed <int>                   随机种子。默认: 不固定
#   --output_dir <dir>             输出目录。默认: data/eval
#   --eval_set <file>              固定评测集(jsonl)路径；不填则随机开局评测
#   --no_visualize                 关闭终端可视化输出
#   --no_vllm                      禁用 vLLM，回退到标准 BF16 推理路径
#   --vllm_quant <str>             vLLM 量化方式。默认: 不使用量化
#                                  常用值: awq, gptq（不建议 bitsandbytes）
#
# 示例:
#   bash scripts/evaluate.sh --model_path checkpoints/sft --num_games 100 --seed 42
# ========================================================

set -e

# Hugging Face mirror defaults (can be overridden by pre-set env vars).
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
mkdir -p "$HF_HOME"

echo "=== Evaluating 2048 Game Model ==="

# Default values
MODEL_PATH=""
BASE_MODEL="Qwen/Qwen3-1.7B"
NUM_GAMES=100
MAX_STEPS=1000
BATCH_SIZE=""
TEMPERATURE=""
PRESENCE_PENALTY=0
USE_THINKING=true
SEED=""
OUTPUT_DIR="data/eval"
EVAL_SET=""
NO_VISUALIZE=false

# 优化选项（默认关闭 vLLM，走标准 BF16）
USE_VLLM=true
VLLM_QUANT=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --base_model)
            BASE_MODEL="$2"
            shift 2
            ;;
        --num_games)
            NUM_GAMES="$2"
            shift 2
            ;;
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --presence_penalty)
            PRESENCE_PENALTY="$2"
            shift 2
            ;;
        --use_thinking)
            USE_THINKING=true
            shift
            ;;
        --no_thinking)
            USE_THINKING=false
            shift
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --eval_set)
            EVAL_SET="$2"
            shift 2
            ;;
        --no_visualize)
            NO_VISUALIZE=true
            shift
            ;;
        --no_vllm)
            USE_VLLM=false
            shift
            ;;
        --vllm_quant)
            VLLM_QUANT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Auto batch size when user does not specify --batch_size.
if [ -z "$BATCH_SIZE" ]; then
    if [ "$NUM_GAMES" -lt 32 ]; then
        BATCH_SIZE="$NUM_GAMES"
    else
        BATCH_SIZE=32
    fi
fi
if [ "$BATCH_SIZE" -lt 1 ]; then
    BATCH_SIZE=1
fi

# Check if model path is provided
if [ -z "$MODEL_PATH" ]; then
    echo "Error: --model_path is required"
    echo "Usage: $0 --model_path <path> [--base_model <name>] [--num_games <n>]"
    exit 1
fi

echo "Configuration:"
echo "  Model path: $MODEL_PATH"
echo "  Base model: $BASE_MODEL"
echo "  Number of games: $NUM_GAMES"
echo "  Max steps: $MAX_STEPS"
echo "  Batch size: $BATCH_SIZE"
if [ -n "$TEMPERATURE" ]; then
    echo "  Temperature: $TEMPERATURE"
else
    if [ "$USE_THINKING" = true ]; then
        echo "  Temperature: auto (0.6)"
    else
        echo "  Temperature: auto (0.7)"
    fi
fi
echo "  Use thinking: $USE_THINKING"
echo "  Presence penalty: $PRESENCE_PENALTY"
if [ -n "$SEED" ]; then
    echo "  Seed: $SEED"
fi
echo "  Output directory: $OUTPUT_DIR"
if [ -n "$EVAL_SET" ]; then
    echo "  Eval set: $EVAL_SET"
fi
echo ""
echo "⚡ 优化配置:"
echo "  vLLM: $USE_VLLM"
if [ "$USE_VLLM" = true ] && [ -n "$VLLM_QUANT" ]; then
    echo "  量化: $VLLM_QUANT"
fi
echo ""

# Evaluate
python -m src.eval.evaluator \
    --model_path "$MODEL_PATH" \
    --base_model "$BASE_MODEL" \
    --num_games "$NUM_GAMES" \
    --max_steps "$MAX_STEPS" \
    --batch_size "$BATCH_SIZE" \
    $( [ -n "$TEMPERATURE" ] && echo "--temperature $TEMPERATURE" ) \
    --presence_penalty "$PRESENCE_PENALTY" \
    $( [ "$USE_THINKING" = true ] && echo "--use_thinking" ) \
    $( [ "$USE_THINKING" = false ] && echo "--no_thinking" ) \
    $( [ -n "$SEED" ] && echo "--seed $SEED" ) \
    --output_dir "$OUTPUT_DIR" \
    $( [ -n "$EVAL_SET" ] && echo "--eval_set $EVAL_SET" ) \
    $( [ "$NO_VISUALIZE" = true ] && echo "--no_visualize" ) \
    $( [ "$USE_VLLM" = true ] && echo "--use_vllm" ) \
    $( [ "$USE_VLLM" = true ] && [ -n "$VLLM_QUANT" ] && echo "--vllm_quantization $VLLM_QUANT" )

echo ""
echo "=== Evaluation complete ==="
