#!/bin/bash
# Generate 2048 raw data (default: with thinking + gap filtering enabled in generator).

set -e

NUM_GAMES=2000
DIFFICULTY="mixed"
OUTPUT_DIR="data/raw"
SEED=42
EXPERT_DEPTH=2
EXPERT_MAX_EMPTY=8
WITH_THINKING=true
ENABLE_DIVERSITY=true
GAP_FILTER=true

while [[ $# -gt 0 ]]; do
  case $1 in
    --num_games)
      NUM_GAMES="$2"; shift 2 ;;
    --difficulty)
      DIFFICULTY="$2"; shift 2 ;;
    --output_dir)
      OUTPUT_DIR="$2"; shift 2 ;;
    --seed)
      SEED="$2"; shift 2 ;;
    --expert_depth)
      EXPERT_DEPTH="$2"; shift 2 ;;
    --expert_max_empty)
      EXPERT_MAX_EMPTY="$2"; shift 2 ;;
    --no_thinking)
      WITH_THINKING=false; shift ;;
    --no_diversity)
      ENABLE_DIVERSITY=false; shift ;;
    --no_gap_filter)
      GAP_FILTER=false; shift ;;
    *)
      echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "=== Generating 2048 Data ==="
echo "  num_games: $NUM_GAMES"
echo "  difficulty: $DIFFICULTY"
echo "  output_dir: $OUTPUT_DIR"
echo "  seed: $SEED"
echo "  with_thinking: $WITH_THINKING"
echo "  expert_depth: $EXPERT_DEPTH"
echo "  expert_max_empty: $EXPERT_MAX_EMPTY"
echo "  enable_diversity: $ENABLE_DIVERSITY"
echo "  gap_filter: $GAP_FILTER"
echo ""

python -m src.data_gen.generator \
  --num_games "$NUM_GAMES" \
  --difficulty "$DIFFICULTY" \
  --output_dir "$OUTPUT_DIR" \
  --seed "$SEED" \
  --expert_depth "$EXPERT_DEPTH" \
  --expert_max_empty "$EXPERT_MAX_EMPTY" \
  $( [ "$WITH_THINKING" = true ] && echo "--with_thinking" ) \
  $( [ "$ENABLE_DIVERSITY" = false ] && echo "--no_enable_diversity" ) \
  $( [ "$GAP_FILTER" = false ] && echo "--no_gap_filter" )

echo ""
echo "=== Data generation complete ==="
