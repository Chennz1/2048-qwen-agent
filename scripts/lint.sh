#!/bin/bash
# Unified lint entrypoint for Python and shell scripts.
# ======================= 参数说明区 =======================
# 用法:
#   bash scripts/lint.sh
#
# 参数:
#   本脚本不接收命令行参数。
#   固定检查内容:
#   1) ruff check src tests scripts
#   2) black --check src tests scripts
#   3) isort --check-only src tests scripts
#   4) shellcheck scripts/*.sh (若 shellcheck 已安装)
#
# 示例:
#   bash scripts/lint.sh
# ========================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY_PATHS=(src tests scripts)
SH_FILES=(scripts/*.sh)

need_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Error: required tool '$cmd' is not installed."
    return 1
  fi
}

echo "==> Lint: ruff"
need_cmd ruff
ruff check "${PY_PATHS[@]}"

echo "==> Format check: black"
need_cmd black
black --check "${PY_PATHS[@]}"

echo "==> Import order check: isort"
need_cmd isort
isort --check-only "${PY_PATHS[@]}"

if command -v shellcheck >/dev/null 2>&1; then
  if [ -e "${SH_FILES[0]}" ]; then
    echo "==> Shell lint: shellcheck"
    shellcheck "${SH_FILES[@]}"
  fi
else
  echo "Warning: shellcheck not found; skipping shell script lint."
fi

echo "Lint completed."
