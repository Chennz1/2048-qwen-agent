#!/bin/bash
# Local CI-style gate: lint + typecheck + tests + smoke pipeline.
# ======================= 参数说明区 =======================
# 用法:
#   bash scripts/ci_local.sh [--skip-smoke] [pytest参数...]
#   bash scripts/ci_local.sh [--skip-smoke] -- [pytest参数...]
#
# 参数:
#   --skip-smoke
#     跳过 smoke pipeline，仅执行 lint/typecheck/tests。
#   -- [pytest参数...]
#     将后续参数原样透传给 scripts/test.sh -> pytest。
#     若不写 --，未识别参数也会被当作 pytest 参数透传。
#
# 示例:
#   bash scripts/ci_local.sh --skip-smoke -k baseline
#   bash scripts/ci_local.sh -- --maxfail=1 -q
# ========================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SKIP_SMOKE=false
TEST_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-smoke)
      SKIP_SMOKE=true
      shift
      ;;
    --)
      shift
      TEST_ARGS=("$@")
      break
      ;;
    *)
      TEST_ARGS+=("$1")
      shift
      ;;
  esac
done

echo "==> [1/4] lint"
bash scripts/lint.sh

echo "==> [2/4] typecheck"
bash scripts/typecheck.sh

echo "==> [3/4] tests"
bash scripts/test.sh "${TEST_ARGS[@]}"

if [ "$SKIP_SMOKE" = false ]; then
  echo "==> [4/4] smoke pipeline"
  bash scripts/smoke_pipeline.sh \
    --monitor_backend none \
    --num_games 20 \
    --epochs 1 \
    --eval_games 5
else
  echo "==> [4/4] smoke pipeline skipped"
fi

echo "Local CI checks completed."
