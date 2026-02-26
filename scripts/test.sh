#!/bin/bash
# Unified test entrypoint for offline unit tests.
# ======================= 参数说明区 =======================
# 用法:
#   bash scripts/test.sh [pytest参数...]
#
# 参数:
#   [pytest参数...]
#     所有参数会原样透传给:
#     pytest -q tests "$@"
#     可用于 -k/-m/--maxfail 等 pytest 过滤与控制参数。
#
# 示例:
#   bash scripts/test.sh
#   bash scripts/test.sh -k evaluator --maxfail=1
# ========================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v pytest >/dev/null 2>&1; then
  echo "Error: pytest is not installed."
  echo "Install dependencies first: pip install -r requirements.txt"
  exit 1
fi

echo "==> Running tests: pytest -q tests $*"
pytest -q tests "$@"
