#!/bin/bash
# Unified type-check entrypoint.
# ======================= 参数说明区 =======================
# 用法:
#   bash scripts/typecheck.sh [mypy参数...]
#
# 参数:
#   [mypy参数...]
#     所有参数会追加到默认 mypy 参数后，并透传给:
#     mypy --pretty --show-error-codes --ignore-missing-imports src "$@"
#
# 示例:
#   bash scripts/typecheck.sh
#   bash scripts/typecheck.sh --strict
# ========================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v mypy >/dev/null 2>&1; then
  echo "Error: mypy is not installed."
  echo "Install dependencies (or dev deps) before type checking."
  exit 1
fi

TARGETS=(src)
DEFAULT_FLAGS=(
  --pretty
  --show-error-codes
  --ignore-missing-imports
)

echo "==> Type check: mypy ${TARGETS[*]} $*"
mypy "${DEFAULT_FLAGS[@]}" "${TARGETS[@]}" "$@"
