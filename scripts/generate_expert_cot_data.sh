#!/bin/bash
# Generate 100% expert (expectimax) CoT training data for 2048.
# ======================= 参数说明区 =======================
# 用法:
#   bash scripts/generate_expert_cot_data.sh [参数]
#
# 参数:
#   --num_games <int>              生成对局数量。默认: 10000
#   --output_dir <dir>             输出目录。默认: data/raw_expert
#   --seed <int>                   随机种子。默认: 42
#   --expert_depth <int>           expectimax 搜索深度。默认: 2
#   --expert_max_empty <int>       expectimax 空位分支上限。默认: 8
#   --no_diversity                 关闭策略多样性增强（默认开启）
#
# 说明:
#   本脚本固定使用:
#   1) --difficulty expert
#   2) --with_thinking
#
# 示例:
#   bash scripts/generate_expert_cot_data.sh --num_games 2000 --expert_depth 2 --expert_max_empty 8
# ========================================================

set -e

echo "=== Generating 100% Expert CoT Data ==="

NUM_GAMES=1
OUTPUT_DIR="data/raw_expert"
SEED=42
EXPERT_DEPTH=2
EXPERT_MAX_EMPTY=8
ENABLE_DIVERSITY=true

while [[ $# -gt 0 ]]; do
  case $1 in
    --num_games)
      NUM_GAMES="$2"; shift 2 ;;
    --output_dir)
      OUTPUT_DIR="$2"; shift 2 ;;
    --seed)
      SEED="$2"; shift 2 ;;
    --expert_depth)
      EXPERT_DEPTH="$2"; shift 2 ;;
    --expert_max_empty)
      EXPERT_MAX_EMPTY="$2"; shift 2 ;;
    --no_diversity)
      ENABLE_DIVERSITY=false; shift ;;
    *)
      echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "Configuration:"
echo "  num_games: $NUM_GAMES"
echo "  difficulty: expert (100%)"
echo "  with_thinking: true"
echo "  output_dir: $OUTPUT_DIR"
echo "  seed: $SEED"
echo "  expert_depth: $EXPERT_DEPTH"
echo "  expert_max_empty: $EXPERT_MAX_EMPTY"
echo "  enable_diversity: $ENABLE_DIVERSITY"
echo ""

python -m src.data_gen.generator \
  --num_games "$NUM_GAMES" \
  --difficulty expert \
  --with_thinking \
  --output_dir "$OUTPUT_DIR" \
  --seed "$SEED" \
  --expert_depth "$EXPERT_DEPTH" \
  --expert_max_empty "$EXPERT_MAX_EMPTY" \
  $( [ "$ENABLE_DIVERSITY" = false ] && echo "--no_enable_diversity" )

echo ""
echo "=== Expert CoT data generation complete ==="
